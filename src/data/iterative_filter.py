"""Filter users and items repeatedly until the dataset stops changing.

Alternate between the cross-domain user filter and the minimum-interactions
item filter because each can invalidate the other. Stop when a complete pass
removes no additional users or items.
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
OUT = PROCESSED / "interactions_converged.parquet"
OUT_LOG = REPORTS / "iterative_filter_log.csv"

MIN_DOMAINS = cfg["data"]["min_domains_per_user"]
MIN_ITEM = cfg["data"]["min_interactions_per_item"]
MIN_USER = cfg["data"]["min_interactions_per_user"]
MAX_ITER = 50


def stats(df: pd.DataFrame) -> tuple[int, int, int, float]:
    n_u = df["user_id"].nunique()
    n_i = df["parent_asin"].nunique()
    n_x = len(df)
    dens = 100 * n_x / (n_u * n_i) if n_u and n_i else 0.0
    return n_u, n_i, n_x, dens


def main() -> None:
    print(f"Reading {IN}")
    df = pd.read_parquet(IN)
    print(f"thresholds: >={MIN_DOMAINS} domains/user, >={MIN_USER} interactions/user, "
          f">={MIN_ITEM} interactions/item\n")

    rows = []
    n_u, n_i, n_x, dens = stats(df)
    print(f"{'iter':>5}{'users':>12}{'items':>10}{'interactions':>15}{'density%':>11}")
    print(f"{0:>5}{n_u:>12,}{n_i:>10,}{n_x:>15,}{dens:>11.4f}")
    rows.append({"iteration": 0, "users": n_u, "items": n_i,
                 "interactions": n_x, "density_pct": round(dens, 5)})

    for it in range(1, MAX_ITER + 1):
        before = len(df)

        # Item filter
        ic = df.groupby("parent_asin", observed=True)["user_id"].transform("size")
        df = df[ic >= MIN_ITEM]

        # User filter: both the domain-breadth and the volume condition
        g = df.groupby("user_id", observed=True)
        u_dom = g["domain"].transform("nunique")
        u_cnt = g["user_id"].transform("size")
        df = df[(u_dom >= MIN_DOMAINS) & (u_cnt >= MIN_USER)]

        n_u, n_i, n_x, dens = stats(df)
        print(f"{it:>5}{n_u:>12,}{n_i:>10,}{n_x:>15,}{dens:>11.4f}")
        rows.append({"iteration": it, "users": n_u, "items": n_i,
                     "interactions": n_x, "density_pct": round(dens, 5)})

        if len(df) == before:
            print(f"\nConverged after {it} iteration(s) — a full pass removed nothing.")
            break
    else:
        print(f"\nWARNING: no convergence in {MAX_ITER} iterations.")

    # Verify both conditions actually hold
    ic = df.groupby("parent_asin", observed=True).size()
    g = df.groupby("user_id", observed=True)
    assert ic.min() >= MIN_ITEM, f"item floor violated: {ic.min()}"
    assert g["domain"].nunique().min() >= MIN_DOMAINS, "domain floor violated"
    assert g.size().min() >= MIN_USER, "user volume floor violated"
    print("Post-convergence assertions passed.")

    print("\nDOMAIN BALANCE AFTER CONVERGENCE")
    per_dom = df.groupby("domain", observed=True)["parent_asin"].nunique().sort_values(ascending=False)
    for d, n in per_dom.items():
        print(f"  {d:<22}{n:>8,} items")
    print(f"  {'ratio largest:smallest':<22}{per_dom.max() / per_dom.min():>8.1f}:1")

    print("\nINTERACTIONS PER ITEM")
    print(f"  median {ic.median():.0f}   mean {ic.mean():.1f}   max {ic.max():,}")
    print("INTERACTIONS PER USER")
    us = g.size()
    print(f"  median {us.median():.0f}   mean {us.mean():.1f}   max {us.max():,}")

    df.to_parquet(OUT, index=False)
    pd.DataFrame(rows).to_csv(OUT_LOG, index=False)
    print(f"\nWrote {OUT}")
    print(f"Wrote {OUT_LOG}")


if __name__ == "__main__":
    main()