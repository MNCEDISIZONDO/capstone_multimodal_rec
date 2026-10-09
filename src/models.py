"""Model architectures for the multimodal recommender 
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MultimodalRecommender(nn.Module):
    """Scores user-item pairs from any enabled combination of three signals.

    Parameters
    ----------
    n_users, n_items
        Sizes of the embedding tables. Identifiers must be contiguous integers
        starting at zero.
    image_features, text_features
        Precomputed encoder outputs, one row per item, in item-index order.
    image_available, text_available
        Per-item boolean masks. False marks a placeholder row that must never be
        consumed as data.
    use_interaction, use_image, use_text
        Signal switches. At least one must be enabled.
    fusion_mode
        'attention' for cross-attention fusion, 'concat' for concatenation
        fusion. Immaterial when only one signal is enabled, since there is
        nothing to fuse.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        image_features: np.ndarray,
        text_features: np.ndarray,
        image_available: np.ndarray,
        text_available: np.ndarray,
        cfg: dict,
        use_interaction: bool = True,
        use_image: bool = True,
        use_text: bool = True,
        fusion_mode: str = "attention",
    ):
        super().__init__()

        if not (use_interaction or use_image or use_text):
            raise ValueError("at least one signal must be enabled")
        if fusion_mode not in {"attention", "concat"}:
            raise ValueError(f"unknown fusion mode: {fusion_mode}")

        model_cfg = cfg["model"]
        self.embedding_dim = model_cfg["embedding_dim"]
        self.fusion_dim = model_cfg["fusion_dim"]
        self.use_interaction = use_interaction
        self.use_image = use_image
        self.use_text = use_text
        self.fusion_mode = fusion_mode

        # Suppressing the collaborative signal during training simulates the
        # cold-start condition, which is only instructive when content signals
        # remain for the model to learn from. A collaborative-only model has no
        # fallback, and structurally cannot serve cold-start items in any case.
        self.collaborative_dropout = (
            model_cfg.get("collaborative_dropout", 0.0)
            if (use_interaction and (use_image or use_text)) else 0.0
        )

        # Precomputed features are held as buffers: they move with the model
        # between devices and are saved with its state, but are never updated by
        # the optimiser, which is what a frozen encoder means in practice.
        self.register_buffer("image_features",
                             torch.as_tensor(image_features, dtype=torch.float32))
        self.register_buffer("text_features",
                             torch.as_tensor(text_features, dtype=torch.float32))
        self.register_buffer("image_available",
                             torch.as_tensor(image_available, dtype=torch.bool))
        self.register_buffer("text_available",
                             torch.as_tensor(text_available, dtype=torch.bool))

        self.user_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.user_projection = nn.Linear(self.embedding_dim, self.fusion_dim)

        # Each enabled signal is projected into a shared space of width
        # fusion_dim, so that signals are directly comparable and the fusion
        # operation is independent of their native dimensionalities.
        if use_interaction:
            self.item_embedding = nn.Embedding(n_items, self.embedding_dim)
            self.collaborative_projection = nn.Linear(self.embedding_dim, self.fusion_dim)

        normalisation = model_cfg.get("modality_normalisation", "layernorm")
        if use_image:
            image_dim = cfg["features"]["clip_dim"]
            self.image_normalisation = self._build_normalisation(normalisation, image_dim)
            self.image_projection = nn.Linear(image_dim, self.fusion_dim)
        if use_text:
            text_dim = cfg["features"]["sbert_dim"]
            self.text_normalisation = self._build_normalisation(normalisation, text_dim)
            self.text_projection = nn.Linear(text_dim, self.fusion_dim)

        self.n_signals = sum([use_interaction, use_image, use_text])

        # Attention can exclude a signal from its keys, so absence needs no
        # representation there. Every other path has a fixed input width and
        # therefore requires one.
        self.uses_attention = (self.n_signals > 1 and fusion_mode == "attention")

        # A learned representation of absence. Distinct from a zero vector: the
        # model observes this parameter during training and learns what it
        # denotes, whereas the origin is a point training never populates.
        if not self.uses_attention:
            self.absent_signal = nn.Parameter(torch.zeros(self.n_signals, self.fusion_dim))
            nn.init.normal_(self.absent_signal, std=0.01)
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.fusion_dim,
                num_heads=model_cfg["attention_heads"],
                dropout=model_cfg["dropout"],
                batch_first=True,
            )

        layers: list[nn.Module] = []
        width = self._head_input_width()
        for hidden in model_cfg["mlp_layers"]:
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(model_cfg["dropout"])]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*layers)

        self._initialise()

    @staticmethod
    def _build_normalisation(kind: str, dim: int) -> nn.Module:
        if kind == "layernorm":
            return nn.LayerNorm(dim)
        if kind == "none":
            return nn.Identity()
        raise ValueError(f"unknown normalisation: {kind}")

    def _head_input_width(self) -> int:
        if self.uses_attention:
            return 2 * self.fusion_dim          # the user and the attended summary
        return (1 + self.n_signals) * self.fusion_dim

    def _initialise(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _signal_tokens(
        self,
        item_idx: torch.Tensor,
        collaborative_available: torch.Tensor | None,
        image_enabled: bool,
        text_enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-signal representations and their availability mask.

        Shapes are (batch, n_signals, fusion_dim) and (batch, n_signals), with
        signals ordered collaborative, image, text among those enabled.
        """
        tokens: list[torch.Tensor] = []
        available: list[torch.Tensor] = []
        batch = item_idx.shape[0]

        if self.use_interaction:
            token = self.collaborative_projection(self.item_embedding(item_idx))
            if collaborative_available is None:
                present = torch.ones(batch, dtype=torch.bool, device=item_idx.device)
            else:
                present = collaborative_available.to(item_idx.device)
            if self.training and self.collaborative_dropout > 0:
                keep = torch.rand(batch, device=item_idx.device) >= self.collaborative_dropout
                present = present & keep
            tokens.append(token)
            available.append(present)

        if self.use_image:
            token = self.image_projection(
                self.image_normalisation(self.image_features[item_idx]))
            present = self.image_available[item_idx]
            if not image_enabled:
                present = torch.zeros_like(present)
            tokens.append(token)
            available.append(present)

        if self.use_text:
            token = self.text_projection(
                self.text_normalisation(self.text_features[item_idx]))
            present = self.text_available[item_idx]
            if not text_enabled:
                present = torch.zeros_like(present)
            tokens.append(token)
            available.append(present)

        return torch.stack(tokens, dim=1), torch.stack(available, dim=1)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        collaborative_available: torch.Tensor | None = None,
        image_enabled: bool = True,
        text_enabled: bool = True,
    ) -> torch.Tensor:
        """Score each user-item pair.

        collaborative_available marks, per pair, whether the item has a learned
        collaborative representation; it is False for items withheld from
        training. image_enabled and text_enabled suppress a modality across the
        whole batch, which is how modality dependence is measured at evaluation
        without altering the trained weights.
        """
        user = self.user_projection(self.user_embedding(user_idx))
        tokens, available = self._signal_tokens(
            item_idx, collaborative_available, image_enabled, text_enabled)

        if not self.uses_attention:
            # Absent signals occupy their slot with a learned representation of
            # absence rather than zeros, keeping the input width fixed while
            # remaining distinguishable from a genuine all-zero feature.
            absent = self.absent_signal.unsqueeze(0).expand_as(tokens)
            fused = torch.where(available.unsqueeze(-1), tokens, absent).flatten(start_dim=1)
        else:
            # Attention over an empty key set is undefined, so the case the
            # content floor exists to prevent is checked explicitly.
            if not available.any(dim=1).all():
                raise ValueError("an item in this batch has no available signal")
            # The user representation queries the available signals, so the
            # weighting of image, text and collaborative evidence is decided per
            # pair rather than fixed in advance. Absent signals are excluded from
            # the keys and contribute nothing to the weighted sum.
            attended, _ = self.attention(
                query=user.unsqueeze(1),
                key=tokens,
                value=tokens,
                key_padding_mask=~available,
                need_weights=False,
            )
            fused = attended.squeeze(1)

        return self.head(torch.cat([user, fused], dim=1)).squeeze(-1)

    def attention_weights(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        collaborative_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the attention placed on each signal, for interpretability.

        Shape is (batch, n_signals), ordered as signal_names() reports.
        """
        if not self.uses_attention:
            raise RuntimeError("attention weights exist only for attention fusion")

        user = self.user_projection(self.user_embedding(user_idx))
        tokens, available = self._signal_tokens(item_idx, collaborative_available, True, True)
        _, weights = self.attention(
            query=user.unsqueeze(1),
            key=tokens,
            value=tokens,
            key_padding_mask=~available,
            need_weights=True,
            average_attn_weights=True,
        )
        return weights.squeeze(1)

    def signal_names(self) -> list[str]:
        names = []
        if self.use_interaction:
            names.append("collaborative")
        if self.use_image:
            names.append("image")
        if self.use_text:
            names.append("text")
        return names


class PopularityBaseline:
    """Scores items by their training interaction count.

    Included because popularity is a strong recommender that a learned model
    should be required to beat, and because it cannot rank a withheld item above
    zero: an item with no interactions has no popularity. That structural
    inability is the clearest statement of the problem this project addresses.
    """

    def __init__(self, n_items: int, train_item_indices: np.ndarray):
        counts = np.zeros(n_items, dtype=np.float32)
        unique, frequency = np.unique(train_item_indices, return_counts=True)
        counts[unique] = frequency
        self.scores = counts

    def score(self, item_idx: np.ndarray) -> np.ndarray:
        return self.scores[item_idx]


class RandomBaseline:
    """Assigns uniform random scores, establishing the floor for every metric."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def score(self, item_idx: np.ndarray) -> np.ndarray:
        return self.rng.random(len(item_idx)).astype(np.float32)


# The named systems in the comparison, each a configuration of one class.
SYSTEMS = {
    "ncf":               dict(use_interaction=True,  use_image=False, use_text=False, fusion_mode="concat"),
    "image_only":        dict(use_interaction=False, use_image=True,  use_text=False, fusion_mode="concat"),
    "text_only":         dict(use_interaction=False, use_image=False, use_text=True,  fusion_mode="concat"),
    "content_only":      dict(use_interaction=False, use_image=True,  use_text=True,  fusion_mode="concat"),
    "interaction_image": dict(use_interaction=True,  use_image=True,  use_text=False, fusion_mode="concat"),
    "interaction_text":  dict(use_interaction=True,  use_image=False, use_text=True,  fusion_mode="concat"),
    "concat_fusion":     dict(use_interaction=True,  use_image=True,  use_text=True,  fusion_mode="concat"),
    "attention_fusion":  dict(use_interaction=True,  use_image=True,  use_text=True,  fusion_mode="attention"),
}


def build_model(cfg: dict, n_users: int, n_items: int,
                image_features: np.ndarray, text_features: np.ndarray,
                image_available: np.ndarray, text_available: np.ndarray,
                system: str) -> MultimodalRecommender:
    """Construct one of the named systems in the comparison."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}; expected one of {sorted(SYSTEMS)}")
    return MultimodalRecommender(
        n_users=n_users,
        n_items=n_items,
        image_features=image_features,
        text_features=text_features,
        image_available=image_available,
        text_available=text_available,
        cfg=cfg,
        **SYSTEMS[system],
    )