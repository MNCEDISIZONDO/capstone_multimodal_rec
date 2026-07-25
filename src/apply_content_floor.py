"""Enforce the content floor (manual section 1.9).

An item with neither a usable image nor usable text is invisible to the content
side of the model. As an ordinary item it contributes nothing beyond its
collaborative vector; as a cold-start item it would be unrecommendable by any
means, so its interactions would feed noise into the evaluation. Removing such
items here guarantees by construction that every candidate for the cold-start
split carries at least one content signal.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import load_config

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
REPORTS = Path(cfg["paths"]["reports"])

CORPUS = PROCESSED / "corpus_items.parquet"
INTERACTIONS = PROCESSED / "corpus_interactions.parquet"
ITEM_META = PROCESSED / "items_domained.parquet"
REGISTRY = REPORTS / "image_registry.csv"

OUT_ITEMS = PROCESSED / "corpus_items_final.parquet"
OUT_INTERACTIONS = PROCESSED / "corpus_interactions_final.parquet"
OUT_SUMMARY = REPORTS / "content_floor_summary.csv"

MIN_TEXT_CHARS = 10


def main() -> None:
    items = pd.read_parquet(CORPUS)
    meta = pd.read_parquet(
        ITEM_META, columns=["parent_asin", "title", "description", "features"])
    registry = pd.read_csv(REGISTRY)

    df = items.merge(meta, on="parent_asin", how="left")
    df = df.merge(registry[["parent_asin", "status"]], on="parent_asin", how="left")

    df["has_image"] = df["status"].isin(["success", "cached"])
    df["text"] = (df["title"].fillna("") + " "
                  + df["features"].fillna("") + " "
                  + df["description"].fillna("")).str.strip()
    df["has_text"] = df["text"].str.len() >= MIN_TEXT_CHARS

    print(f"corpus items        : {len(df):,}")
    print(f"  with image        : {df['has_image'].sum():,}  ({df['has_image'].mean():.1%})")
    print(f"  with text         : {df['has_text'].sum():,}  ({df['has_text'].mean():.1%})")
    print(f"  with both         : {(df['has_image'] & df['has_text']).sum():,}")
    print(f"  image only        : {(df['has_image'] & ~df['has_text']).sum():,}")
    print(f"  text only         : {(~df['has_image'] & df['has_text']).sum():,}")

    dropped = df[~df["has_image"] & ~df["has_text"]]
    kept = df[df["has_image"] | df["has_text"]].copy()

    print(f"\nremoved by content floor: {len(dropped):,}")
    print(f"final corpus items      : {len(kept):,}")

    interactions = pd.read_parquet(INTERACTIONS)
    before = len(interactions)
    interactions = interactions[interactions["parent_asin"].isin(set(kept["parent_asin"]))]
    print(f"interactions            : {before:,} -> {len(interactions):,}")

    print("\nitems per domain:")
    per_domain = kept["domain"].value_counts()
    for d in cfg["data"]["domains"]:
        print(f"  {d:<22}{int(per_domain.get(d, 0)):>6,}")

    kept[["parent_asin", "domain", "n_interactions",
          "has_image", "has_text"]].to_parquet(OUT_ITEMS, index=False)
    interactions.to_parquet(OUT_INTERACTIONS, index=False)
    pd.DataFrame([{
        "corpus_items_before": len(df),
        "removed_by_content_floor": len(dropped),
        "corpus_items_final": len(kept),
        "image_success_rate": round(df["has_image"].mean(), 4),
        "interactions_final": len(interactions),
    }]).to_csv(OUT_SUMMARY, index=False)

    print(f"\nWrote {OUT_ITEMS}")
    print(f"Wrote {OUT_INTERACTIONS}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()