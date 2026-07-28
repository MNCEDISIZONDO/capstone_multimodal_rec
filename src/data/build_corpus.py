"""Settle the final corpus size (manual section 1.8).

Applies the relaxed item threshold and runs iterative filtering to its natural
convergence point. The converged corpus is used in full rather than sampled
down to a smaller target: because the item and user filters are mutually
dependent, removing items disqualifies their purchasers from the power-user
constraint, which starves further items in a destructive cascade. Preserving
the full converged graph maintains corpus size, interaction density, and the
complete popularity spectrum simultaneously.
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
REPORTS.mkdir(parents=True, exist_ok=True)

IN = PROCESSED / "interactions_inscope.parquet"
OUT_CORPUS = PROCESSED / "corpus_items.parquet"
OUT_INTERACTIONS = PROCESSED / "corpus_interactions.parquet"
OUT_REPORT = REPORTS / "corpus_summary.csv"

MIN_ITEM = cfg["data"]["min_interactions_per_item"]
MIN_DOMAINS = cfg["data"]["min_domains_per_user"]
MIN_USER = cfg["data"]["min_interactions_per_user"]
DOMAINS = cfg["data"]["domains"]


def converge(df: pd.DataFrame, max_iter: int = 60) -> pd.DataFrame:
    """Alternate the item and user filters until a full pass removes nothing."""
    for _ in range(max_iter):
        before = len(df)
        item_counts = df.groupby("parent_asin", observed=True)["user_id"].transform("size")
        df = df[item_counts >= MIN_ITEM]
        g = df.groupby("user_id", observed=True)
        df = df[(g["domain"].transform("nunique") >= MIN_DOMAINS)
                & (g["user_id"].transform("size") >= MIN_USER)]
        if len(df) == before:
            return df
    raise RuntimeError(f"filtering did not converge within {max_iter} iterations")


def density(df: pd.DataFrame) -> float:
    """Percentage of possible user-item pairs that are observed."""
    users, items = df["user_id"].nunique(), df["parent_asin"].nunique()
    return 100 * len(df) / (users * items) if users and items else 0.0


def main() -> None:
    print(f"Converging at minimum {MIN_ITEM} interactions per item")
    corpus = converge(pd.read_parquet(IN))

    items = (corpus.groupby("parent_asin", observed=True)
                   .agg(n_interactions=("user_id", "size"),
                        domain=("domain", "first"))
                   .reset_index())

    print("\nFINAL CORPUS")
    print(f"  users        {corpus['user_id'].nunique():,}")
    print(f"  items        {corpus['parent_asin'].nunique():,}")
    print(f"  interactions {len(corpus):,}")
    print(f"  density      {density(corpus):.4f}%")

    print("\n  items per domain:")
    per_domain = items["domain"].value_counts()
    for d in DOMAINS:
        print(f"    {d:<22}{int(per_domain.get(d, 0)):>6,}")
    print(f"    {'ratio':<22}{per_domain.max() / per_domain.min():>6.1f}:1")

    items.to_parquet(OUT_CORPUS, index=False)
    corpus.to_parquet(OUT_INTERACTIONS, index=False)
    pd.DataFrame([{
        "users": corpus["user_id"].nunique(),
        "items": corpus["parent_asin"].nunique(),
        "interactions": len(corpus),
        "density_pct": round(density(corpus), 4),
        "min_interactions_per_item": MIN_ITEM,
    }]).to_csv(OUT_REPORT, index=False)

    print(f"\nWrote {OUT_CORPUS}")
    print(f"Wrote {OUT_INTERACTIONS}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()