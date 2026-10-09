"""Validate and freeze the final dataset splits.

Check that all files are readable, identifiers are complete, splits do not
overlap, and no cold-start items leak into training. Record dataset statistics
such as popularity distribution, cold-start representation and domain density.

Save checksums for the final splits so that any later changes can be detected.
"""
from __future__ import annotations

# Allow this module to import the shared modules at the top of src/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
REPORTS = Path(cfg["paths"]["reports"])

SPLITS = {
    "train": PROCESSED / "split_train.parquet",
    "validation": PROCESSED / "split_validation.parquet",
    "test": PROCESSED / "split_test.parquet",
    "ghost_evaluation": PROCESSED / "split_ghost_evaluation.parquet",
}
ITEMS = PROCESSED / "corpus_items_final.parquet"
GHOST_ITEMS = PROCESSED / "ghost_items.parquet"
ITEM_META = PROCESSED / "items_domained.parquet"

OUT_USER_INDEX = PROCESSED / "user_index.parquet"
OUT_ITEM_INDEX = PROCESSED / "item_index.parquet"
OUT_STATS = REPORTS / "dataset_statistics.csv"
OUT_CHECKSUMS = REPORTS / "split_checksums.json"


def gini(values: np.ndarray) -> float:
    """Inequality of the popularity distribution; 0 is uniform, 1 is maximal."""
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    data = {name: pd.read_parquet(p) for name, p in SPLITS.items()}
    items = pd.read_parquet(ITEMS)
    ghost_ids = set(pd.read_parquet(GHOST_ITEMS)["parent_asin"])
    meta = pd.read_parquet(ITEM_META, columns=["parent_asin", "title"])

    print("ENGINEERING VALIDATION")

    for name, path in SPLITS.items():
        assert path.exists() and path.stat().st_size > 0, f"{name} missing or empty"
    print("  all split files present and readable        : yes")

    train = data["train"]
    leakage = int(train["parent_asin"].isin(ghost_ids).sum())
    print(f"  cold-start leakage into training            : {leakage}")
    assert leakage == 0, "cold-start items present in training"

    train_users = set(train["user_id"])
    for name in ("validation", "test", "ghost_evaluation"):
        assert set(data[name]["user_id"]) <= train_users, f"{name} user absent from training"
    print("  every evaluated user appears in training    : yes")

    key = ["user_id", "parent_asin"]
    names = list(SPLITS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = len(pd.merge(data[a][key], data[b][key], on=key, how="inner"))
            assert n == 0, f"{a} and {b} share {n} interactions"
    print("  all splits mutually disjoint                : yes")

    titled = meta[meta["title"].fillna("").str.len() > 0]["parent_asin"]
    missing_title = set(items["parent_asin"]) - set(titled)
    print(f"  corpus items without a title                : {len(missing_title)}")
    assert not missing_title, "an item has no text to embed"

    all_users = sorted({u for d in data.values() for u in d["user_id"].unique()})
    all_items = sorted({i for d in data.values() for i in d["parent_asin"].unique()}
                       | set(items["parent_asin"]))

    user_index = pd.DataFrame({"user_id": all_users,
                               "user_idx": range(len(all_users))})
    item_index = (pd.DataFrame({"parent_asin": all_items,
                                "item_idx": range(len(all_items))})
                  .merge(items[["parent_asin", "domain"]], on="parent_asin", how="left"))
    item_index["is_ghost"] = item_index["parent_asin"].isin(ghost_ids)

    for name, d in data.items():
        assert set(d["user_id"]) <= set(user_index["user_id"]), f"{name} user missing from index"
        assert set(d["parent_asin"]) <= set(item_index["parent_asin"]), f"{name} item missing from index"
    print("  index mappings cover every identifier       : yes")
    print(f"  users indexed {len(user_index):,}   items indexed {len(item_index):,}")

    print("\nRESEARCH VALIDATION")

    pop = items["n_interactions"].to_numpy()
    order = np.sort(pop)[::-1]
    print(f"  popularity Gini coefficient                 : {gini(pop):.3f}")
    for pct in (1, 5, 10):
        k = max(1, round(len(order) * pct / 100))
        print(f"  share of interactions held by top {pct:>2}% items : "
              f"{order[:k].sum() / order.sum():.1%}")

    ghost_pop = items.loc[items["parent_asin"].isin(ghost_ids), "n_interactions"]
    other_pop = items.loc[~items["parent_asin"].isin(ghost_ids), "n_interactions"]
    print(f"  mean interactions  cold-start {ghost_pop.mean():5.1f}"
          f"   warm {other_pop.mean():5.1f}")
    print(f"  median interactions cold-start {ghost_pop.median():4.0f}"
          f"   warm {other_pop.median():4.0f}")

    print(f"\n  {'domain':<22}{'items':>8}{'train int.':>12}{'density %':>12}")
    rows = []
    for d in cfg["data"]["domains"]:
        dom_items = set(items.loc[items["domain"] == d, "parent_asin"])
        sub = train[train["parent_asin"].isin(dom_items)]
        n_u, n_i = sub["user_id"].nunique(), sub["parent_asin"].nunique()
        dens = 100 * len(sub) / (n_u * n_i) if n_u and n_i else 0.0
        print(f"  {d:<22}{len(dom_items):>8,}{len(sub):>12,}{dens:>12.4f}")
        rows.append({"domain": d, "items": len(dom_items),
                     "train_interactions": len(sub), "density_pct": round(dens, 4)})

    print("\nFREEZING SPLIT")
    sums = {name: checksum(p) for name, p in SPLITS.items()}
    for name, digest in sums.items():
        print(f"  {name:<18}{digest[:16]}...")

    user_index.to_parquet(OUT_USER_INDEX, index=False)
    item_index.to_parquet(OUT_ITEM_INDEX, index=False)
    with open(OUT_CHECKSUMS, "w") as fh:
        json.dump({"seed": cfg["seeds"]["split_construction"], "sha256": sums}, fh, indent=2)

    n_u, n_i = train["user_id"].nunique(), train["parent_asin"].nunique()
    pd.DataFrame([{
        "corpus_items": len(items),
        "corpus_interactions": int(items["n_interactions"].sum()),
        "users": len(user_index),
        "train_interactions": len(train),
        "validation_interactions": len(data["validation"]),
        "test_interactions": len(data["test"]),
        "ghost_interactions": len(data["ghost_evaluation"]),
        "ghost_items": len(ghost_ids),
        "train_density_pct": round(100 * len(train) / (n_u * n_i), 4),
        "popularity_gini": round(gini(pop), 4),
        "image_success_rate": round(items["has_image"].mean(), 4),
        "split_seed": cfg["seeds"]["split_construction"],
    }]).to_csv(OUT_STATS, index=False)
    pd.DataFrame(rows).to_csv(REPORTS / "domain_density.csv", index=False)

    print(f"\nWrote {OUT_USER_INDEX}")
    print(f"Wrote {OUT_ITEM_INDEX}")
    print(f"Wrote {OUT_STATS}")
    print(f"Wrote {OUT_CHECKSUMS}")


if __name__ == "__main__":
    main()