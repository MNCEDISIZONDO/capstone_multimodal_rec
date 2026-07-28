"""Partition the non-Ghost interactions into training, validation and test
(manual section 1.12).

The split is chronological and leave-one-out: each user's most recent
interaction becomes their test item, their second most recent their validation
item, and everything earlier is training. Time-based partitioning mirrors the
deployed task of predicting a user's next action from their history, whereas a
random split would allow the model to predict earlier behaviour from later
behaviour.

Ties in timestamp are broken deterministically by item identifier. The source
data batch-records reviews, so identical timestamps are common, and an
undefined tie-break would make the split non-reproducible across rebuilds.

Users with too few interactions are removed before splitting: one interaction
each goes to validation and test, so a minimum is required for the remainder to
support a learnable representation. Because this removal happens after the
cold-start split was constructed, the cold-start evaluation set is re-checked
afterwards for users who no longer appear in training.
"""
from __future__ import annotations

# Allow this module to import the shared modules at the top of src/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

import pandas as pd

from config import load_config

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
REPORTS = Path(cfg["paths"]["reports"])

IN_REMAINING = PROCESSED / "interactions_non_ghost.parquet"
IN_GHOST_EVAL = PROCESSED / "ghost_evaluation.parquet"
IN_GHOST_ITEMS = PROCESSED / "ghost_items.parquet"

OUT_TRAIN = PROCESSED / "split_train.parquet"
OUT_VAL = PROCESSED / "split_validation.parquet"
OUT_TEST = PROCESSED / "split_test.parquet"
OUT_GHOST_EVAL = PROCESSED / "split_ghost_evaluation.parquet"
OUT_SUMMARY = REPORTS / "split_summary.csv"

MIN_USER = cfg["data"]["min_interactions_per_user"]


def main() -> None:
    df = pd.read_parquet(IN_REMAINING)
    print(f"non-ghost interactions : {len(df):,}")
    print(f"users                  : {df['user_id'].nunique():,}")

    counts = df.groupby("user_id", observed=True)["parent_asin"].transform("size")
    df = df[counts >= MIN_USER]
    print(f"\nusers with >= {MIN_USER} interactions : {df['user_id'].nunique():,}")
    print(f"interactions retained  : {len(df):,}")

    df = df.sort_values(["user_id", "timestamp", "parent_asin"],
                        kind="mergesort").reset_index(drop=True)
    rank_desc = df.groupby("user_id", observed=True).cumcount(ascending=False)

    test = df[rank_desc == 0].copy()
    validation = df[rank_desc == 1].copy()
    train = df[rank_desc >= 2].copy()

    print("\nSPLIT")
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        print(f"  {name:<12}{len(part):>9,} interactions"
              f"{part['user_id'].nunique():>9,} users"
              f"{part['parent_asin'].nunique():>9,} items")

    train_users = set(train["user_id"].unique())
    train_items = set(train["parent_asin"].unique())

    print("\nVALIDATION")
    assert set(validation["user_id"]) <= train_users, "validation user absent from training"
    assert set(test["user_id"]) <= train_users, "test user absent from training"
    print("  every evaluated user appears in training : yes")

    key = ["user_id", "parent_asin"]
    for a_name, a, b_name, b in [("train", train, "validation", validation),
                                 ("train", train, "test", test),
                                 ("validation", validation, "test", test)]:
        overlap = len(pd.merge(a[key], b[key], on=key, how="inner"))
        print(f"  overlap {a_name} / {b_name:<11}: {overlap}")
        assert overlap == 0, f"{a_name} and {b_name} share an interaction"

    # Users dropped by the minimum-interaction filter are no longer represented
    # in training, so their cold-start interactions would test user cold-start
    # rather than item cold-start.
    ghost_eval = pd.read_parquet(IN_GHOST_EVAL)
    before = len(ghost_eval)
    ghost_eval = ghost_eval[ghost_eval["user_id"].isin(train_users)]
    print(f"\n  cold-start interactions removed (user not in training) : "
          f"{before - len(ghost_eval):,}")

    ghost_items = pd.read_parquet(IN_GHOST_ITEMS)
    still_evaluable = set(ghost_eval["parent_asin"].unique())
    lost = set(ghost_items["parent_asin"]) - still_evaluable
    print(f"  cold-start items left unevaluable                      : {len(lost):,}")

    assert not train["parent_asin"].isin(set(ghost_items["parent_asin"])).any(), \
        "cold-start item present in training"
    print("  cold-start leakage into training                        : 0")

    items_without_training = (set(validation["parent_asin"]) | set(test["parent_asin"])) - train_items
    print(f"  evaluated items with no training interaction           : "
          f"{len(items_without_training):,}")

    train.to_parquet(OUT_TRAIN, index=False)
    validation.to_parquet(OUT_VAL, index=False)
    test.to_parquet(OUT_TEST, index=False)
    ghost_eval.to_parquet(OUT_GHOST_EVAL, index=False)

    pd.DataFrame([{
        "train_interactions": len(train),
        "validation_interactions": len(validation),
        "test_interactions": len(test),
        "ghost_interactions": len(ghost_eval),
        "users": len(train_users),
        "train_items": len(train_items),
        "ghost_items_evaluable": len(still_evaluable),
        "min_interactions_per_user": MIN_USER,
    }]).to_csv(OUT_SUMMARY, index=False)

    print(f"\nWrote {OUT_TRAIN}")
    print(f"Wrote {OUT_VAL}")
    print(f"Wrote {OUT_TEST}")
    print(f"Wrote {OUT_GHOST_EVAL}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()