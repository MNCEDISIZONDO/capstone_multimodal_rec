"""Construct the cold-start (Ghost) split (manual sections 1.10 and 1.11).

Ghost items are withheld entirely from training: the model never observes a
single interaction involving them. At evaluation it must recommend them from
image and text alone, simulating a newly listed product.

The order of operations is critical. The corpus is finalised first, Ghost items
are selected second, every interaction involving a Ghost item is removed third,
and only then is the remaining pool available for training. Selecting Ghost
items after splitting would allow the model to learn representations for them,
turning the cold-start evaluation into a warm-start evaluation that reports
better results than the truth.

Selection is stratified by domain and popularity band so the Ghost set mirrors
the catalogue rather than concentrating on unusually easy or unusually
difficult items.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
REPORTS = Path(cfg["paths"]["reports"])

ITEMS = PROCESSED / "corpus_items_final.parquet"
INTERACTIONS = PROCESSED / "corpus_interactions_final.parquet"

OUT_GHOST_ITEMS = PROCESSED / "ghost_items.parquet"
OUT_GHOST_EVAL = PROCESSED / "ghost_evaluation.parquet"
OUT_REMAINING = PROCESSED / "interactions_non_ghost.parquet"
OUT_SUMMARY = REPORTS / "ghost_split_summary.csv"

SEED = cfg["seeds"]["split_construction"]
GHOST_FRACTION = cfg["data"]["ghost_fraction"]
MIN_GHOST_INTERACTIONS = cfg["data"]["ghost_min_interactions"]
DOMAINS = cfg["data"]["domains"]
N_BANDS = 3


def select_ghost_items(items: pd.DataFrame, target: int, rng) -> set[str]:
    """Choose Ghost items stratified by domain and popularity band.

    Eligibility requires a verified image, usable text, and enough interactions
    to be worth evaluating. Popularity banding is applied within each domain so
    the Ghost set spans the full range rather than clustering at either extreme.
    """
    eligible = items[
        items["has_image"]
        & items["has_text"]
        & (items["n_interactions"] >= MIN_GHOST_INTERACTIONS)
    ]

    chosen: list[str] = []
    for domain in DOMAINS:
        pool = eligible[eligible["domain"] == domain].copy()
        if pool.empty:
            continue

        quota = round(target * len(items[items["domain"] == domain]) / len(items))
        quota = min(quota, len(pool))
        if quota == 0:
            continue

        pool["band"] = pd.qcut(pool["n_interactions"], q=N_BANDS,
                               labels=False, duplicates="drop")
        n_bands = pool["band"].nunique()
        picked: list[str] = []
        for band in sorted(pool["band"].unique()):
            band_items = pool.loc[pool["band"] == band, "parent_asin"].to_numpy()
            take = min(quota // n_bands, len(band_items))
            picked.extend(rng.choice(band_items, size=take, replace=False))

        if len(picked) < quota:
            remaining = pool.loc[~pool["parent_asin"].isin(picked), "parent_asin"].to_numpy()
            extra = min(quota - len(picked), len(remaining))
            if extra > 0:
                picked.extend(rng.choice(remaining, size=extra, replace=False))

        chosen.extend(picked)

    return set(chosen)


def main() -> None:
    rng = np.random.default_rng(SEED)

    items = pd.read_parquet(ITEMS)
    interactions = pd.read_parquet(INTERACTIONS)
    target = round(len(items) * GHOST_FRACTION)

    print(f"corpus items          : {len(items):,}")
    print(f"target ghost items    : {target:,}  ({GHOST_FRACTION:.0%})")

    ghost_ids = select_ghost_items(items, target, rng)
    print(f"selected ghost items  : {len(ghost_ids):,}\n")

    is_ghost = interactions["parent_asin"].isin(ghost_ids)
    ghost_eval = interactions[is_ghost].copy()
    remaining = interactions[~is_ghost].copy()

    print(f"ghost interactions    : {len(ghost_eval):,}")
    print(f"remaining for training: {len(remaining):,}")

    # Users absent from the remaining pool have no learnable representation, so
    # evaluating them would test user cold-start rather than item cold-start.
    training_users = set(remaining["user_id"].unique())
    orphans = set(ghost_eval["user_id"].unique()) - training_users
    ghost_eval = ghost_eval[~ghost_eval["user_id"].isin(orphans)]
    print(f"orphan users removed  : {len(orphans):,}")
    print(f"ghost eval after       : {len(ghost_eval):,} interactions")

    # Items left without evaluable interactions cannot be assessed.
    evaluable = set(ghost_eval["parent_asin"].unique())
    unevaluable = ghost_ids - evaluable
    if unevaluable:
        print(f"ghost items with no evaluable interaction: {len(unevaluable):,}")
        ghost_ids = evaluable

    ghost_items = items[items["parent_asin"].isin(ghost_ids)]

    print("\nVALIDATION")
    leakage = remaining["parent_asin"].isin(ghost_ids).sum()
    print(f"  leakage (must be 0)       : {leakage}")
    assert leakage == 0, "ghost items present in the training pool"

    covered = ghost_eval["parent_asin"].nunique()
    print(f"  ghost items covered       : {covered:,} / {len(ghost_ids):,}")
    assert covered == len(ghost_ids), "ghost items without evaluable interactions"

    remaining_users = set(remaining["user_id"].unique())
    assert set(ghost_eval["user_id"].unique()) <= remaining_users, "orphan users remain"
    print("  orphan users              : 0")

    key = ["user_id", "parent_asin"]
    overlap = pd.merge(remaining[key], ghost_eval[key], on=key, how="inner")
    print(f"  overlap between splits    : {len(overlap)}")
    assert len(overlap) == 0, "an interaction appears in more than one split"

    print("\nREPRESENTATIVENESS")
    ghost_mean = ghost_items["n_interactions"].mean()
    other_mean = items.loc[~items["parent_asin"].isin(ghost_ids), "n_interactions"].mean()
    print(f"  mean interactions  ghost {ghost_mean:6.1f}   non-ghost {other_mean:6.1f}")
    print(f"  {'domain':<22}{'ghost':>8}{'share':>9}{'corpus share':>15}")
    for d in DOMAINS:
        g = int((ghost_items["domain"] == d).sum())
        c = int((items["domain"] == d).sum())
        print(f"  {d:<22}{g:>8,}{g / len(ghost_items):>9.1%}{c / len(items):>15.1%}")

    ghost_items.to_parquet(OUT_GHOST_ITEMS, index=False)
    ghost_eval.to_parquet(OUT_GHOST_EVAL, index=False)
    remaining.to_parquet(OUT_REMAINING, index=False)
    pd.DataFrame([{
        "ghost_items": len(ghost_items),
        "ghost_interactions": len(ghost_eval),
        "training_pool_interactions": len(remaining),
        "orphan_users_removed": len(orphans),
        "leakage": int(leakage),
        "ghost_mean_interactions": round(ghost_mean, 2),
        "non_ghost_mean_interactions": round(other_mean, 2),
    }]).to_csv(OUT_SUMMARY, index=False)

    print(f"\nWrote {OUT_GHOST_ITEMS}")
    print(f"Wrote {OUT_GHOST_EVAL}")
    print(f"Wrote {OUT_REMAINING}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()