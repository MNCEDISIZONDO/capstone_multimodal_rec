"""Compare each model's content-only performance on seen and unseen items.

The pilot showed a fusion model scoring higher than the text-only model when the
collaborative signal was masked on validation items, yet lower on the withheld
items. Both measurements exercise the same pathway, so the difference lies in
generalisation rather than in the pathway's quality.

This measures both under one protocol. Seen items are scored from the validation
split with the collaborative signal masked, so the model must rank them from
content alone despite having trained on them. Unseen items are the withheld
split. The ratio between the two is what separates a model that learned
transferable structure from one that fitted the items it was shown.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from config import load_config
from device_utils import clear_gpu_memory, get_device
from models import build_model

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
FEATURES = Path(cfg["paths"]["features"])
CHECKPOINTS = Path(cfg["paths"]["checkpoints"])
REPORTS = Path(cfg["paths"]["reports"])

K = cfg["evaluation"]["k"]
CHUNK = 64
SEEDS = cfg["seeds"]["statistical"]
SYSTEMS = ["text_only", "image_only", "content_only",
           "interaction_text", "interaction_image",
           "concat_fusion", "attention_fusion"]


def ndcg_from_ranks(ranks: np.ndarray) -> float:
    if len(ranks) == 0:
        return 0.0
    return float(np.where(ranks <= K, 1.0 / np.log2(ranks + 1.0), 0.0).mean())


def rank_content_only(model, device, eval_users, eval_items,
                      candidates, seen) -> np.ndarray:
    """Rank held-out items with the collaborative signal masked throughout."""
    model.eval()
    candidate_tensor = torch.as_tensor(candidates, device=device)
    position_of = {int(item): index for index, item in enumerate(candidates)}
    n_candidates = len(candidates)
    absent = torch.zeros(n_candidates, dtype=torch.bool, device=device)

    collected = []
    with torch.no_grad():
        for start in range(0, len(eval_users), CHUNK):
            chunk_users = eval_users[start:start + CHUNK]
            chunk_items = eval_items[start:start + CHUNK]
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

            rows, columns = [], []
            for row, user in enumerate(chunk_users):
                already = seen.get(int(user))
                if already is not None and len(already):
                    rows.append(np.full(len(already), row, dtype=np.int64))
                    columns.append(already)
            if rows:
                scores[torch.as_tensor(np.concatenate(rows), device=device),
                       torch.as_tensor(np.concatenate(columns), device=device)] = float("-inf")

            target_positions = torch.as_tensor(np.where(valid, positions, 0),
                                               device=device)
            target = scores.gather(1, target_positions.unsqueeze(1))
            greater = (scores > target).sum(dim=1).float()
            tied = (scores == target).sum(dim=1).float()
            collected.append((greater + (tied + 1.0) / 2.0).cpu().numpy()[valid])

    return np.concatenate(collected) if collected else np.array([])


def main() -> None:
    device = get_device()

    items = pd.read_parquet(PROCESSED / "item_index.parquet").sort_values("item_idx")
    users = pd.read_parquet(PROCESSED / "user_index.parquet")
    item_of = dict(zip(items["parent_asin"], items["item_idx"]))
    user_of = dict(zip(users["user_id"], users["user_idx"]))

    def to_indices(name):
        frame = pd.read_parquet(PROCESSED / name)
        return (frame["user_id"].map(user_of).to_numpy(dtype=np.int64),
                frame["parent_asin"].map(item_of).to_numpy(dtype=np.int64))

    train_users, train_items = to_indices("split_train.parquet")
    val_users, val_items = to_indices("split_validation.parquet")
    cold_users, cold_items = to_indices("split_ghost_evaluation.parquet")

    with h5py.File(FEATURES / "image_embeddings.h5", "r") as store:
        image_features, image_available = store["embeddings"][:], store["available"][:]
    with h5py.File(FEATURES / "text_embeddings.h5", "r") as store:
        text_features, text_available = store["embeddings"][:], store["available"][:]

    is_ghost = items["is_ghost"].to_numpy().copy()
    seen_candidates = np.flatnonzero(~is_ghost).astype(np.int64)
    unseen_candidates = np.flatnonzero(is_ghost).astype(np.int64)

    position_of = {int(i): p for p, i in enumerate(seen_candidates)}
    seen_map: dict[int, np.ndarray] = {}
    frame = pd.DataFrame({"user": train_users, "item": train_items})
    for user, group in frame.groupby("user"):
        positions = [position_of[int(i)] for i in group["item"] if int(i) in position_of]
        if positions:
            seen_map[int(user)] = np.array(positions, dtype=np.int64)

    print(f"seen items evaluated   : {len(seen_candidates):,} candidates, "
          f"{len(val_users):,} interactions")
    print(f"unseen items evaluated : {len(unseen_candidates):,} candidates, "
          f"{len(cold_users):,} interactions\n")

    print(f"  {'system':<20}{'seed':>5}{'seen':>10}{'unseen':>10}{'retention':>12}")
    rows = []

    for system in SYSTEMS:
        for seed in SEEDS:
            path = CHECKPOINTS / f"{system}_seed{seed}.pt"
            if not path.exists():
                continue

            model = build_model(cfg, len(users), len(items),
                                image_features, text_features,
                                image_available, text_available, system).to(device)
            model.load_state_dict(torch.load(path, map_location=device)["state_dict"])

            seen_score = ndcg_from_ranks(rank_content_only(
                model, device, val_users, val_items, seen_candidates, seen_map))
            unseen_score = ndcg_from_ranks(rank_content_only(
                model, device, cold_users, cold_items, unseen_candidates, {}))

            retention = unseen_score / seen_score if seen_score else 0.0
            print(f"  {system:<20}{seed:>5}{seen_score:>10.4f}"
                  f"{unseen_score:>10.4f}{retention:>11.1%}")
            rows.append({"system": system, "seed": seed,
                         "seen_ndcg": seen_score, "unseen_ndcg": unseen_score,
                         "retention": retention})
            clear_gpu_memory()

    frame = pd.DataFrame(rows)
    summary = (frame.groupby("system")
               .agg(seen=("seen_ndcg", "mean"), seen_sd=("seen_ndcg", "std"),
                    unseen=("unseen_ndcg", "mean"), unseen_sd=("unseen_ndcg", "std"),
                    retention=("retention", "mean"))
               .sort_values("retention", ascending=False))

    print("\nMEAN ACROSS SEEDS, ordered by retention")
    print(f"  {'system':<20}{'seen':>10}{'unseen':>10}{'retention':>12}")
    for system, row in summary.iterrows():
        print(f"  {system:<20}{row['seen']:>10.4f}{row['unseen']:>10.4f}"
              f"{row['retention']:>11.1%}")

    frame.to_csv(REPORTS / "generalisation_gap_per_seed.csv", index=False)
    summary.round(5).to_csv(REPORTS / "generalisation_gap.csv")
    print(f"\nWrote {REPORTS / 'generalisation_gap.csv'}")


if __name__ == "__main__":
    main()