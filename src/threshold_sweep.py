"""Sensitivity sweep: converge at several item thresholds and compare.

Section 1.8 says size serves density. This measures the trade directly so the
final threshold is chosen against evidence rather than assumption.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import load_config

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
REPORTS = Path(cfg["paths"]["reports"])

IN = PROCESSED / "interactions_inscope.parquet"
OUT = REPORTS / "threshold_sweep.csv"

MIN_DOMAINS = cfg["data"]["min_domains_per_user"]
MIN_USER = cfg["data"]["min_interactions_per_user"]
GRID = [3, 5, 7, 10]
DOMAINS = cfg["data"]["domains"]


def converge(df, min_item, max_iter=60):
    for _ in range(max_iter):
        before = len(df)
        ic = df.groupby("parent_asin", observed=True)["user_id"].transform("size")
        df = df[ic >= min_item]
        g = df.groupby("user_id", observed=True)
        df = df[(g["domain"].transform("nunique") >= MIN_DOMAINS)
                & (g["user_id"].transform("size") >= MIN_USER)]
        if len(df) == before:
            break
    return df


def main() -> None:
    base = pd.read_parquet(IN)
    rows = []
    for m in GRID:
        d = converge(base.copy(), m)
        n_u = d["user_id"].nunique()
        n_i = d["parent_asin"].nunique()
        n_x = len(d)
        dens = 100 * n_x / (n_u * n_i) if n_u and n_i else 0.0
        per = d.groupby("domain", observed=True)["parent_asin"].nunique()
        rec = {"min_item": m, "users": n_u, "items": n_i, "interactions": n_x,
               "density_pct": round(dens, 4),
               "ratio": round(per.max() / per.min(), 1) if len(per) > 1 else 0}
        for dom in DOMAINS:
            rec[dom] = int(per.get(dom, 0))
        rows.append(rec)
        print(f"min_item={m:>2}: {n_u:>7,} users  {n_i:>6,} items  "
              f"{n_x:>8,} interactions  density {dens:.4f}%  ratio {rec['ratio']}:1")
        print("   " + "  ".join(f"{dom.split('_')[0]}={rec[dom]:,}" for dom in DOMAINS))

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()