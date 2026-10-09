"""Check whether ranking performance comes from personalisation or popularity.

Compare learned models with the popularity baseline and measure item coverage.
Higher coverage shows that recommendations are spread across more products,
while low coverage may indicate that the model repeatedly recommends the same
popular items.
"""
from __future__ import annotations

# Allow this module to import the shared modules at the top of src/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from config import load_config
from device_utils import get_device
from models import PopularityBaseline, build_model

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
FEATURES = Path(cfg["paths"]["features"])
CHECKPOINTS = Path(cfg["paths"]["checkpoints"])
REPORTS = Path(cfg["paths"]["reports"])

K = cfg["evaluation"]["k"]
N_USERS = 1000
CHUNK = 64


def main() -> None:
    device = get_device()

    items = pd.read_parquet(PROCESSED / "item_index.parquet").sort_values("item_idx")
    users = pd.read_parquet(PROCESSED / "user_index.parquet")
    item_of = dict(zip(items["parent_asin"], items["item_idx"]))
    user_of = dict(zip(users["user_id"], users["user_idx"]))

    train = pd.read_parquet(PROCESSED / "split_train.parquet")
    train_users = train["user_id"].map(user_of).to_numpy(dtype=np.int64)
    train_items = train["parent_asin"].map(item_of).to_numpy(dtype=np.int64)

    validation = pd.read_parquet(PROCESSED / "split_validation.parquet")
    val_users = validation["user_id"].map(user_of).to_numpy(dtype=np.int64)
    val_items = validation["parent_asin"].map(item_of).to_numpy(dtype=np.int64)

    with h5py.File(FEATURES / "image_embeddings.h5", "r") as store:
        image_features, image_available = store["embeddings"][:], store["available"][:]
    with h5py.File(FEATURES / "text_embeddings.h5", "r") as store:
        text_features, text_available = store["embeddings"][:], store["available"][:]

    is_ghost = items["is_ghost"].to_numpy().copy()
    candidates = np.flatnonzero(~is_ghost).astype(np.int64)
    position_of = {int(item): index for index, item in enumerate(candidates)}
    n_candidates = len(candidates)

    seen: dict[int, np.ndarray] = {}
    frame = pd.DataFrame({"user": train_users, "item": train_items})
    for user, group in frame.groupby("user"):
        positions = [position_of[int(i)] for i in group["item"] if int(i) in position_of]
        if positions:
            seen[int(user)] = np.array(positions, dtype=np.int64)

    rng = np.random.default_rng(cfg["seeds"]["development"])
    chosen = rng.choice(len(val_users), size=min(N_USERS, len(val_users)), replace=False)
    val_users, val_items = val_users[chosen], val_items[chosen]

    print(f"users evaluated   : {len(val_users):,}")
    print(f"candidate items   : {n_candidates:,}\n")
    print(f"  {'system':<20}{'NDCG@10':>9}{'HR@10':>8}{'coverage':>10}"
          f"{'top item share':>16}")

    rows = []

    def summarise(name: str, top_k_lists: np.ndarray, ndcg: float, hit: float) -> None:
        flat = top_k_lists.reshape(-1)
        distinct = len(np.unique(flat))
        counts = np.bincount(flat, minlength=n_candidates)
        share = counts.max() / len(top_k_lists)
        print(f"  {name:<20}{ndcg:>9.4f}{hit:>8.4f}"
              f"{distinct:>7,}/{n_candidates:,}{share:>15.1%}")
        rows.append({"system": name, "ndcg": round(ndcg, 5), "hit_rate": round(hit, 5),
                     "distinct_items": distinct, "candidates": n_candidates,
                     "coverage": round(distinct / n_candidates, 5),
                     "most_recommended_item_share": round(float(share), 5)})

    # Popularity reference: the same ranking for every user, by construction.
    popularity = PopularityBaseline(len(items), train_items)
    scores = popularity.score(candidates)
    ndcg_total = hit_total = 0.0
    top_lists = []
    for user, target in zip(val_users, val_items):
        target_position = position_of.get(int(target))
        row = scores.copy()
        already_seen = seen.get(int(user))
        if already_seen is not None:
            row[already_seen] = -np.inf
        order = np.argpartition(-row, K)[:K]
        top_lists.append(order)
        if target_position is not None:
            rank = int((row > row[target_position]).sum()) + 1
            if rank <= K:
                ndcg_total += 1.0 / np.log2(rank + 1)
                hit_total += 1.0
    summarise("popularity", np.array(top_lists), ndcg_total / len(val_users),
              hit_total / len(val_users))

    for system in ["ncf", "image_only", "text_only", "content_only",
                   "concat_fusion", "attention_fusion"]:
        path = CHECKPOINTS / f"{system}_seed{cfg['seeds']['development']}.pt"
        if not path.exists():
            continue
        model = build_model(cfg, len(users), len(items), image_features, text_features,
                            image_available, text_available, system).to(device)
        model.load_state_dict(torch.load(path, map_location=device)["state_dict"])
        model.eval()

        candidate_tensor = torch.as_tensor(candidates, device=device)
        ndcg_total = hit_total = 0.0
        top_lists = []

        with torch.no_grad():
            for start in range(0, len(val_users), CHUNK):
                chunk_users = val_users[start:start + CHUNK]
                chunk_items = val_items[start:start + CHUNK]
                n_chunk = len(chunk_users)

                user_tensor = torch.as_tensor(chunk_users, device=device)
                scores = model(user_tensor.repeat_interleave(n_candidates),
                               candidate_tensor.repeat(n_chunk)).view(n_chunk, n_candidates)

                rows_index, columns_index = [], []
                for row, user in enumerate(chunk_users):
                    already_seen = seen.get(int(user))
                    if already_seen is not None and len(already_seen):
                        rows_index.append(np.full(len(already_seen), row, dtype=np.int64))
                        columns_index.append(already_seen)
                if rows_index:
                    scores[torch.as_tensor(np.concatenate(rows_index), device=device),
                           torch.as_tensor(np.concatenate(columns_index), device=device)] = float("-inf")

                top_lists.append(scores.topk(K, dim=1).indices.cpu().numpy())

                positions = np.array([position_of.get(int(i), -1) for i in chunk_items])
                valid = positions >= 0
                target_positions = torch.as_tensor(np.where(valid, positions, 0), device=device)
                target_scores = scores.gather(1, target_positions.unsqueeze(1))
                ranks = (scores > target_scores).sum(dim=1) + 1
                within = (ranks <= K) & torch.as_tensor(valid, device=device)
                ndcg_total += float(torch.where(
                    within, 1.0 / torch.log2(ranks.float() + 1.0),
                    torch.zeros_like(ranks, dtype=torch.float32)).sum())
                hit_total += float(within.sum())

        summarise(system, np.concatenate(top_lists),
                  ndcg_total / len(val_users), hit_total / len(val_users))

    output = REPORTS / "diagnostic_coverage.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()