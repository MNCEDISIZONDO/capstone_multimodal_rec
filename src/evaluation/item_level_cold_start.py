"""Per-item cold-start retrieval quality, for error analysis.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
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
RESULTS = Path(cfg["paths"]["results"])
TABLES = RESULTS / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

K = cfg["evaluation"]["k"]
CHUNK = cfg["training"]["validation_chunk_users"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", default="concat_fusion")
    parser.add_argument("--seed", type=int, default=cfg["seeds"]["statistical"][0])
    arguments = parser.parse_args()

    device = get_device()

    items = pd.read_parquet(PROCESSED / "item_index.parquet").sort_values("item_idx")
    users = pd.read_parquet(PROCESSED / "user_index.parquet")
    item_of = dict(zip(items["parent_asin"], items["item_idx"]))
    user_of = dict(zip(users["user_id"], users["user_idx"]))

    cold = pd.read_parquet(PROCESSED / "split_ghost_evaluation.parquet")
    cold_users = cold["user_id"].map(user_of).to_numpy(dtype=np.int64)
    cold_items = cold["parent_asin"].map(item_of).to_numpy(dtype=np.int64)

    with h5py.File(FEATURES / "image_embeddings.h5", "r") as store:
        image_features, image_available = store["embeddings"][:], store["available"][:]
    with h5py.File(FEATURES / "text_embeddings.h5", "r") as store:
        text_features, text_available = store["embeddings"][:], store["available"][:]

    is_ghost = items["is_ghost"].to_numpy().copy()
    candidates = np.flatnonzero(is_ghost).astype(np.int64)
    position_of = {int(item): index for index, item in enumerate(candidates)}

    checkpoint = CHECKPOINTS / f"{arguments.system}_seed{arguments.seed}.pt"
    if not checkpoint.exists():
        raise SystemExit(f"no checkpoint at {checkpoint}")

    model = build_model(cfg, len(users), len(items), image_features, text_features,
                        image_available, text_available, arguments.system).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"])
    model.eval()

    print(f"system      : {arguments.system} (seed {arguments.seed})")
    print(f"candidates  : {len(candidates):,} withheld items")
    print(f"evaluating  : {len(cold_users):,} interactions\n")

    candidate_tensor = torch.as_tensor(candidates, device=device)
    absent = torch.zeros(len(candidates), dtype=torch.bool, device=device)

    ranks = np.empty(len(cold_users), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(cold_users), CHUNK):
            chunk_users = cold_users[start:start + CHUNK]
            chunk_items = cold_items[start:start + CHUNK]
            n_chunk = len(chunk_users)

            user_tensor = torch.as_tensor(chunk_users, device=device)
            scores = model(user_tensor.repeat_interleave(len(candidates)),
                           candidate_tensor.repeat(n_chunk),
                           collaborative_available=absent.repeat(n_chunk)
                           ).view(n_chunk, len(candidates))

            positions = torch.as_tensor(
                [position_of[int(i)] for i in chunk_items], device=device)
            target = scores.gather(1, positions.unsqueeze(1))
            greater = (scores > target).sum(dim=1).float()
            tied = (scores == target).sum(dim=1).float()
            ranks[start:start + n_chunk] = (
                greater + (tied + 1.0) / 2.0).cpu().numpy()

    clear_gpu_memory()

    ndcg = np.where(ranks <= K, 1.0 / np.log2(ranks + 1.0), 0.0)
    per_interaction = pd.DataFrame({"item_idx": cold_items, "rank": ranks, "ndcg": ndcg})
    per_item = (per_interaction.groupby("item_idx")
                .agg(mean_ndcg=("ndcg", "mean"),
                     median_rank=("rank", "median"),
                     hit_rate=("rank", lambda r: float((r <= K).mean())),
                     n_interactions=("ndcg", "size"))
                .reset_index())

    metadata = pd.read_parquet(
        PROCESSED / "items_domained.parquet",
        columns=["parent_asin", "title", "description", "features", "domain"])
    corpus = pd.read_parquet(PROCESSED / "corpus_items_final.parquet")

    per_item = (per_item
                .merge(items[["item_idx", "parent_asin", "domain"]], on="item_idx")
                .merge(metadata[["parent_asin", "title", "description", "features"]],
                       on="parent_asin", how="left")
                .merge(corpus[["parent_asin", "has_image", "n_interactions"]]
                       .rename(columns={"n_interactions": "corpus_interactions"}),
                       on="parent_asin", how="left"))

    per_item["title_length"] = per_item["title"].fillna("").str.len()
    per_item["description_length"] = per_item["description"].fillna("").str.len()
    per_item["features_length"] = per_item["features"].fillna("").str.len()
    per_item["text_length"] = (per_item["title_length"]
                               + per_item["description_length"]
                               + per_item["features_length"])
    per_item["has_description"] = per_item["description_length"] > 0

    output = TABLES / f"cold_start_per_item_{arguments.system}_seed{arguments.seed}.csv"
    per_item.drop(columns=["description", "features"]).to_csv(output, index=False)

    print(f"items scored        : {len(per_item):,}")
    print(f"mean NDCG@{K}        : {per_item['mean_ndcg'].mean():.4f}")
    print(f"items never retrieved: "
          f"{int((per_item['mean_ndcg'] == 0).sum()):,} "
          f"({(per_item['mean_ndcg'] == 0).mean():.1%})")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()