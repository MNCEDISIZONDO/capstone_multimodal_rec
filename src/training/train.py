"""Train one system on the development seed and report the best validation NDCG@10 and coverage at that epoch.
"""
from __future__ import annotations

# Allow this module to import the shared modules at the top of src/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import copy
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import load_config
from device_utils import clear_gpu_memory, get_device, vram_report
from models import SYSTEMS, build_model

BASE_CFG = load_config()
PROCESSED = Path(BASE_CFG["paths"]["processed_data"])
FEATURES = Path(BASE_CFG["paths"]["features"])
CHECKPOINTS = Path(BASE_CFG["paths"]["checkpoints"])
LOGS = Path(BASE_CFG["paths"]["logs"])
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

K = BASE_CFG["evaluation"]["k"]

_CACHE: dict | None = None


def load_everything() -> dict:
    """Load the frozen split, the index mappings and the precomputed features.

    Cached across calls so that a sweep of configurations does not repeatedly
    re-read the same immutable inputs.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    items = pd.read_parquet(PROCESSED / "item_index.parquet").sort_values("item_idx")
    users = pd.read_parquet(PROCESSED / "user_index.parquet")

    item_of = dict(zip(items["parent_asin"], items["item_idx"]))
    user_of = dict(zip(users["user_id"], users["user_idx"]))

    def to_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (frame["user_id"].map(user_of).to_numpy(dtype=np.int64),
                frame["parent_asin"].map(item_of).to_numpy(dtype=np.int64))

    train_users, train_items = to_indices(pd.read_parquet(PROCESSED / "split_train.parquet"))
    val_users, val_items = to_indices(pd.read_parquet(PROCESSED / "split_validation.parquet"))

    with h5py.File(FEATURES / "image_embeddings.h5", "r") as store:
        image_features, image_available = store["embeddings"][:], store["available"][:]
    with h5py.File(FEATURES / "text_embeddings.h5", "r") as store:
        text_features, text_available = store["embeddings"][:], store["available"][:]

    _CACHE = dict(
        n_users=len(users), n_items=len(items),
        train_users=train_users, train_items=train_items,
        val_users=val_users, val_items=val_items,
        image_features=image_features, image_available=image_available,
        text_features=text_features, text_available=text_available,
        is_ghost=items["is_ghost"].to_numpy().copy(),
    )
    return _CACHE


class NegativeSampler:
    """Draws negatives from items the model can represent, avoiding false ones.

    Withheld cold-start items are excluded because they have no collaborative
    representation during training. Items the user actually interacted with are
    excluded because presenting a chosen item as a rejection is a labelling
    error rather than a hard negative.
    """

    def __init__(self, n_items: int, is_ghost: np.ndarray,
                 train_users: np.ndarray, train_items: np.ndarray, rng):
        self.pool = np.flatnonzero(~is_ghost).astype(np.int64)
        self.n_items = n_items
        self.rng = rng
        # Interactions are encoded as single integers so that membership can be
        # tested with a sorted search rather than a per-sample lookup.
        self.observed = np.sort(train_users.astype(np.int64) * n_items + train_items)

    def _is_observed(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        keys = users * self.n_items + items
        position = np.clip(np.searchsorted(self.observed, keys),
                           0, len(self.observed) - 1)
        return self.observed[position] == keys

    def sample(self, users: np.ndarray, n_negatives: int) -> np.ndarray:
        repeated = np.repeat(users, n_negatives)
        negatives = self.rng.choice(self.pool, size=len(repeated))
        for _ in range(10):
            collision = self._is_observed(repeated, negatives)
            if not collision.any():
                break
            negatives[collision] = self.rng.choice(self.pool, size=int(collision.sum()))
        return negatives


def evaluate(model, device, val_users: np.ndarray, val_items: np.ndarray,
             candidates: np.ndarray, seen: dict[int, np.ndarray],
             chunk_size: int) -> tuple[float, float, int]:
    """Rank each user's held-out item against the full candidate catalogue.

    Returns NDCG@K, hit rate@K, and the number of distinct items appearing in
    any user's top-K list.

    Items the user interacted with during training are removed from the
    candidate list, since scoring the model on items it correctly learned the
    user already chose would penalise it for being right.

    Ranks are computed as tensor operations over a whole chunk of users rather
    than user by user, because reading a scalar back from the accelerator forces
    a synchronisation, and one synchronisation per user dominates the cost of
    the scoring itself.
    """
    model.eval()
    candidate_tensor = torch.as_tensor(candidates, device=device)
    position_of = {int(item): index for index, item in enumerate(candidates)}
    n_candidates = len(candidates)

    ndcg_total = hit_total = 0.0
    counted = 0
    recommended: set[int] = set()

    with torch.no_grad():
        for start in range(0, len(val_users), chunk_size):
            chunk_users = val_users[start:start + chunk_size]
            chunk_items = val_items[start:start + chunk_size]
            n_chunk = len(chunk_users)

            positions = np.array([position_of.get(int(i), -1) for i in chunk_items])
            valid = positions >= 0
            if not valid.any():
                continue

            user_tensor = torch.as_tensor(chunk_users, device=device)
            scores = model(user_tensor.repeat_interleave(n_candidates),
                           candidate_tensor.repeat(n_chunk)).view(n_chunk, n_candidates)

            # Exclusion indices are assembled on the host and applied in a
            # single scattered write, avoiding a device round trip per user.
            rows, columns = [], []
            for row, user in enumerate(chunk_users):
                already_seen = seen.get(int(user))
                if already_seen is not None and len(already_seen):
                    rows.append(np.full(len(already_seen), row, dtype=np.int64))
                    columns.append(already_seen)
            if rows:
                scores[torch.as_tensor(np.concatenate(rows), device=device),
                       torch.as_tensor(np.concatenate(columns), device=device)] = float("-inf")

            recommended.update(scores.topk(K, dim=1).indices.cpu().numpy().reshape(-1).tolist())

            target_positions = torch.as_tensor(np.where(valid, positions, 0), device=device)
            target_scores = scores.gather(1, target_positions.unsqueeze(1))
            ranks = (scores > target_scores).sum(dim=1) + 1

            within_k = (ranks <= K) & torch.as_tensor(valid, device=device)
            ndcg_total += float(torch.where(
                within_k, 1.0 / torch.log2(ranks.float() + 1.0),
                torch.zeros_like(ranks, dtype=torch.float32)).sum())
            hit_total += float(within_k.sum())
            counted += int(valid.sum())

    return (ndcg_total / counted if counted else 0.0,
            hit_total / counted if counted else 0.0,
            len(recommended))


def train(system: str, seed: int, smoke: bool = False, tag: str = "",
          overrides: dict | None = None, quiet: bool = False) -> dict:
    """Train one system and return a summary of the run."""
    cfg = copy.deepcopy(BASE_CFG)
    for path, value in (overrides or {}).items():
        section, key = path.split(".")
        cfg[section][key] = value

    train_cfg = cfg["training"]
    data = load_everything()
    device = get_device()

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = build_model(cfg, data["n_users"], data["n_items"],
                        data["image_features"], data["text_features"],
                        data["image_available"], data["text_available"],
                        system).to(device)

    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=train_cfg["learning_rate"],
                                 weight_decay=train_cfg["weight_decay"])

    sampler = NegativeSampler(data["n_items"], data["is_ghost"],
                              data["train_users"], data["train_items"], rng)

    # Withheld items are not candidates during warm-start validation; they are
    # evaluated separately under the cold-start condition.
    candidates = np.flatnonzero(~data["is_ghost"]).astype(np.int64)
    position_of = {int(item): index for index, item in enumerate(candidates)}

    seen: dict[int, np.ndarray] = {}
    frame = pd.DataFrame({"user": data["train_users"], "item": data["train_items"]})
    for user, group in frame.groupby("user"):
        positions = [position_of[int(i)] for i in group["item"] if int(i) in position_of]
        if positions:
            seen[int(user)] = np.array(positions, dtype=np.int64)

    # The validation sample is drawn once, so the early-stopping signal is
    # comparable across epochs and across runs.
    n_validation = min(train_cfg["validation_users"], len(data["val_users"]))
    chosen = rng.choice(len(data["val_users"]), size=n_validation, replace=False)
    val_users = data["val_users"][chosen]
    val_items = data["val_items"][chosen]

    max_epochs = 2 if smoke else train_cfg["max_epochs"]
    batch_size = train_cfg["batch_size"]
    n_negatives = train_cfg["num_negatives"]
    n_train = len(data["train_users"])
    name = f"{system}{('_' + tag) if tag else ''}_seed{seed}"

    if not quiet:
        print(f"system            : {system}")
        print(f"run               : {name}")
        print(f"collab. dropout   : {cfg['model']['collaborative_dropout']}")
        print(f"embedding dim     : {cfg['model']['embedding_dim']}")
        print(f"training pairs    : {n_train:,}")
        print(f"candidate pool    : {len(candidates):,} items")
        print(f"parameters        : "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"{vram_report()}\n")

    history = []
    best_ndcg = -1.0
    best_epoch = -1
    best_coverage = 0
    without_improvement = 0
    checkpoint_path = CHECKPOINTS / f"{name}.pt"

    for epoch in range(1, max_epochs + 1):
        started = time.time()
        model.train()
        order = rng.permutation(n_train)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            index = order[start:start + batch_size]
            users = data["train_users"][index]
            positives = data["train_items"][index]
            negatives = sampler.sample(users, n_negatives)

            user_tensor = torch.as_tensor(users, device=device)
            positive_scores = model(user_tensor, torch.as_tensor(positives, device=device))
            negative_scores = model(user_tensor.repeat_interleave(n_negatives),
                                    torch.as_tensor(negatives, device=device))

            loss = -F.logsigmoid(
                positive_scores.repeat_interleave(n_negatives) - negative_scores).mean()

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            n_batches += 1

        train_time = time.time() - started
        validation_started = time.time()
        ndcg, hit_rate, coverage = evaluate(
            model, device, val_users, val_items, candidates, seen,
            train_cfg["validation_chunk_users"])
        validation_time = time.time() - validation_started

        mean_loss = total_loss / n_batches
        improved = ndcg > best_ndcg
        if improved:
            best_ndcg, best_epoch, best_coverage = ndcg, epoch, coverage
            without_improvement = 0
            torch.save({"state_dict": model.state_dict(), "system": system,
                        "seed": seed, "epoch": epoch, "val_ndcg": ndcg,
                        "coverage": coverage,
                        "collaborative_dropout": cfg["model"]["collaborative_dropout"],
                        "embedding_dim": cfg["model"]["embedding_dim"]},
                       checkpoint_path)
        else:
            without_improvement += 1

        history.append({"epoch": epoch, "train_loss": round(mean_loss, 5),
                        "val_ndcg": round(ndcg, 5), "val_hit_rate": round(hit_rate, 5),
                        "coverage": coverage,
                        "train_seconds": round(train_time, 1),
                        "validation_seconds": round(validation_time, 1)})

        if not quiet:
            print(f"epoch {epoch:>3}  loss {mean_loss:.4f}  "
                  f"NDCG@{K} {ndcg:.4f}  HR@{K} {hit_rate:.4f}  "
                  f"cov {coverage:>5,}  {train_time:.0f}s + {validation_time:.0f}s"
                  f"{'  *' if improved else ''}")

        if without_improvement >= train_cfg["early_stopping_patience"]:
            if not quiet:
                print(f"\nstopped early: no improvement for "
                      f"{train_cfg['early_stopping_patience']} epochs")
            break

    pd.DataFrame(history).to_csv(LOGS / f"{name}.csv", index=False)

    summary = {"system": system, "run": name, "seed": seed,
               "collaborative_dropout": cfg["model"]["collaborative_dropout"],
               "embedding_dim": cfg["model"]["embedding_dim"],
               "best_epoch": best_epoch, "best_val_ndcg": round(best_ndcg, 5),
               "coverage_at_best": best_coverage, "epochs_run": len(history),
               "checkpoint": str(checkpoint_path)}
    (LOGS / f"{name}.json").write_text(json.dumps(summary, indent=2))

    if not quiet:
        print(f"\nbest NDCG@{K} {best_ndcg:.4f} at epoch {best_epoch}, "
              f"coverage {best_coverage:,}")
        print(f"Wrote {checkpoint_path}")

    clear_gpu_memory()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one system.")
    parser.add_argument("system", choices=sorted(SYSTEMS))
    parser.add_argument("--seed", type=int, default=BASE_CFG["seeds"]["development"])
    parser.add_argument("--smoke", action="store_true",
                        help="run two epochs only, to verify the pipeline")
    parser.add_argument("--tag", default="", help="suffix for output filenames")
    parser.add_argument("--collaborative-dropout", type=float, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    arguments = parser.parse_args()

    overrides = {}
    if arguments.collaborative_dropout is not None:
        overrides["model.collaborative_dropout"] = arguments.collaborative_dropout
    if arguments.embedding_dim is not None:
        overrides["model.embedding_dim"] = arguments.embedding_dim
    if arguments.max_epochs is not None:
        overrides["training.max_epochs"] = arguments.max_epochs
    if arguments.patience is not None:
        overrides["training.early_stopping_patience"] = arguments.patience

    train(arguments.system, arguments.seed, arguments.smoke,
          arguments.tag, overrides)


if __name__ == "__main__":
    main()