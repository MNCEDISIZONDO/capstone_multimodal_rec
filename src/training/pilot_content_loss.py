
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import load_config
from device_utils import clear_gpu_memory, get_device
from models import build_model
from train import NegativeSampler, evaluate, load_everything

cfg = load_config()
SYSTEM = "concat_fusion"
SEED = cfg["seeds"]["development"]
EPOCHS = 25
VALIDATION_USERS = 500


def run(content_weight: float) -> pd.DataFrame:
    data = load_everything()
    device = get_device()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    model = build_model(cfg, data["n_users"], data["n_items"],
                        data["image_features"], data["text_features"],
                        data["image_available"], data["text_available"],
                        SYSTEM).to(device)

    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=cfg["training"]["learning_rate"],
                                 weight_decay=cfg["training"]["weight_decay"])

    sampler = NegativeSampler(data["n_items"], data["is_ghost"],
                              data["train_users"], data["train_items"], rng)
    candidates = np.flatnonzero(~data["is_ghost"]).astype(np.int64)
    position_of = {int(item): index for index, item in enumerate(candidates)}

    seen: dict[int, np.ndarray] = {}
    frame = pd.DataFrame({"user": data["train_users"], "item": data["train_items"]})
    for user, group in frame.groupby("user"):
        positions = [position_of[int(i)] for i in group["item"] if int(i) in position_of]
        if positions:
            seen[int(user)] = np.array(positions, dtype=np.int64)

    chosen = rng.choice(len(data["val_users"]), size=VALIDATION_USERS, replace=False)
    val_users = data["val_users"][chosen]
    val_items = data["val_items"][chosen]

    batch_size = cfg["training"]["batch_size"]
    n_negatives = cfg["training"]["num_negatives"]
    n_train = len(data["train_users"])

    history = []
    for epoch in range(1, EPOCHS + 1):
        started = time.time()
        model.train()
        order = rng.permutation(n_train)
        total, n_batches = 0.0, 0

        for start in range(0, n_train, batch_size):
            index = order[start:start + batch_size]
            users = torch.as_tensor(data["train_users"][index], device=device)
            positives = torch.as_tensor(data["train_items"][index], device=device)
            negatives = torch.as_tensor(
                sampler.sample(data["train_users"][index], n_negatives), device=device)
            expanded = users.repeat_interleave(n_negatives)

            def ranking_loss(available: torch.Tensor | None) -> torch.Tensor:
                positive_available = available
                negative_available = (None if available is None
                                      else available.repeat_interleave(n_negatives))
                positive = model(users, positives,
                                 collaborative_available=positive_available)
                negative = model(expanded, negatives,
                                 collaborative_available=negative_available)
                return -F.logsigmoid(
                    positive.repeat_interleave(n_negatives) - negative).mean()

            loss = ranking_loss(None)
            if content_weight > 0:
                absent = torch.zeros(len(index), dtype=torch.bool, device=device)
                loss = loss + content_weight * ranking_loss(absent)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += loss.item()
            n_batches += 1

        warm_ndcg, _, _ = evaluate(model, device, val_users, val_items,
                                   candidates, seen, 64)
        cold_ndcg = content_only_validation(model, device, val_users, val_items,
                                            candidates, seen)

        history.append({"epoch": epoch, "loss": total / n_batches,
                        "warm_ndcg": warm_ndcg, "content_ndcg": cold_ndcg})
        print(f"  epoch {epoch:>3}  loss {total / n_batches:.4f}  "
              f"warm {warm_ndcg:.4f}  content-only {cold_ndcg:.4f}  "
              f"{time.time() - started:.0f}s", flush=True)

    clear_gpu_memory()
    return pd.DataFrame(history)


def content_only_validation(model, device, val_users, val_items,
                            candidates, seen) -> float:
    """Validation NDCG with the collaborative signal masked for every item.

    This measures the content pathway on items the model has otherwise seen,
    which is a proxy for cold-start ability that does not consult the withheld
    split.
    """
    model.eval()
    candidate_tensor = torch.as_tensor(candidates, device=device)
    position_of = {int(item): index for index, item in enumerate(candidates)}
    n_candidates = len(candidates)
    absent = torch.zeros(n_candidates, dtype=torch.bool, device=device)

    total, counted = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(val_users), 64):
            chunk_users = val_users[start:start + 64]
            chunk_items = val_items[start:start + 64]
            n_chunk = len(chunk_users)

            positions = np.array([position_of.get(int(i), -1) for i in chunk_items])
            valid = positions >= 0
            if not valid.any():
                continue

            user_tensor = torch.as_tensor(chunk_users, device=device)
            scores = model(user_tensor.repeat_interleave(n_candidates),
                           candidate_tensor.repeat(n_chunk),
                           collaborative_available=absent.repeat(n_chunk)
                           ).view(n_chunk, n_candidates)

            for row, user in enumerate(chunk_users):
                already = seen.get(int(user))
                if already is not None and len(already):
                    scores[row, torch.as_tensor(already, device=device)] = float("-inf")

            target_positions = torch.as_tensor(np.where(valid, positions, 0),
                                               device=device)
            target = scores.gather(1, target_positions.unsqueeze(1))
            ranks = ((scores > target).sum(dim=1) + 1).cpu().numpy()
            within = (ranks <= 10) & valid
            total += float(np.where(within, 1.0 / np.log2(ranks + 1.0), 0.0).sum())
            counted += int(valid.sum())

    return total / counted if counted else 0.0


def main() -> None:
    print(f"system : {SYSTEM}   seed {SEED}   {EPOCHS} epochs\n")

    print("WITHOUT auxiliary content loss")
    baseline = run(content_weight=0.0)

    print("\nWITH auxiliary content loss (weight 1.0)")
    augmented = run(content_weight=1.0)

    print("\nBEST VALIDATION SCORES")
    print(f"  {'':<22}{'warm':>10}{'content-only':>15}")
    print(f"  {'without auxiliary':<22}{baseline['warm_ndcg'].max():>10.4f}"
          f"{baseline['content_ndcg'].max():>15.4f}")
    print(f"  {'with auxiliary':<22}{augmented['warm_ndcg'].max():>10.4f}"
          f"{augmented['content_ndcg'].max():>15.4f}")

    warm_change = ((augmented["warm_ndcg"].max() / baseline["warm_ndcg"].max() - 1)
                   * 100)
    content_change = ((augmented["content_ndcg"].max()
                       / max(baseline["content_ndcg"].max(), 1e-9) - 1) * 100)
    print(f"\n  warm         {warm_change:+.1f}%")
    print(f"  content-only {content_change:+.1f}%")

    output = Path(cfg["paths"]["reports"]) / "pilot_content_loss.csv"
    pd.concat([baseline.assign(auxiliary=False),
               augmented.assign(auxiliary=True)]).to_csv(output, index=False)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    from pathlib import Path
    main()