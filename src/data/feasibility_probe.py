"""Check whether the dataset is large enough before filtering.

This checks whether there are enough users active across different domains to
produce the required number of items and interactions. The results help explain
the threshold values used in config.yaml.

These counts are calculated before repeated filtering, so they are only upper
limits. The final dataset will be smaller. If the three-domain dataset is already
too small at this stage, later filtering will not make it larger.
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
OUT_USERS = REPORTS / "census_users_by_domain.csv"
OUT_ITEMS = REPORTS / "census_items_by_domain.csv"
OUT_WHATIF = REPORTS / "census_whatif.csv"

MIN_DOMAINS_GRID = [2, 3, 4]
MIN_ITEM_GRID = [3, 5, 10]


def main() -> None:
    print(f"Reading {IN}")
    df = pd.read_parquet(IN)
    df["user_id"] = df["user_id"].astype("category")
    df["parent_asin"] = df["parent_asin"].astype("category")
    df["domain"] = df["domain"].astype("category")
    print(f"  {len(df):,} interactions")
    print(f"  {df['user_id'].nunique():,} users")
    print(f"  {df['parent_asin'].nunique():,} items")
    print(f"  memory: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB\n")

    # ---------- 1. Users by domains touched ----------
    ud = df.groupby("user_id", observed=True)["domain"].nunique()
    un = df.groupby("user_id", observed=True).size()
    users = pd.DataFrame({"n_domains": ud, "n_interactions": un})

    print("USERS BY NUMBER OF DOMAINS TOUCHED")
    dist = users["n_domains"].value_counts().sort_index()
    total_u = len(users)
    for k, v in dist.items():
        print(f"  {k} domain(s): {v:>10,}  ({v / total_u:6.2%})")

    print("\n  cumulative (users touching AT LEAST k domains)")
    for k in range(1, 6):
        v = int((users["n_domains"] >= k).sum())
        ints = int(users.loc[users["n_domains"] >= k, "n_interactions"].sum())
        print(f"  >= {k}: {v:>10,} users   {ints:>12,} interactions")

    print("\n  median interactions per user, by domains touched")
    med = users.groupby("n_domains")["n_interactions"].median()
    for k, v in med.items():
        print(f"  {k} domain(s): {v:.1f}")

    users.groupby("n_domains").agg(
        n_users=("n_interactions", "size"),
        total_interactions=("n_interactions", "sum"),
        median_interactions=("n_interactions", "median"),
        mean_interactions=("n_interactions", "mean"),
    ).to_csv(OUT_USERS)

    # ---------- 2. Items by interaction count ----------
    ic = df.groupby("parent_asin", observed=True).size().rename("n_interactions")
    idom = df.groupby("parent_asin", observed=True)["domain"].first()
    items = pd.DataFrame({"n_interactions": ic, "domain": idom})

    print("\nITEMS SURVIVING EACH MINIMUM INTERACTION THRESHOLD")
    header = f"  {'domain':<22}" + "".join(f"{'>=' + str(t):>10}" for t in [1, 3, 5, 10, 20])
    print(header)
    for d in cfg["data"]["domains"]:
        sub = items[items["domain"] == d]
        row = f"  {d:<22}"
        for t in [1, 3, 5, 10, 20]:
            row += f"{int((sub['n_interactions'] >= t).sum()):>10,}"
        print(row)
    row = f"  {'ALL':<22}"
    for t in [1, 3, 5, 10, 20]:
        row += f"{int((items['n_interactions'] >= t).sum()):>10,}"
    print(row)

    items.groupby("domain", observed=True)["n_interactions"].describe().to_csv(OUT_ITEMS)

    # ---------- 3. What-if table ----------
    print("\nWHAT-IF TABLE (one filter pass; iteration will reduce further)")
    print(f"  {'min_dom':>8}{'min_item':>10}{'users':>12}{'items':>10}"
          f"{'interactions':>15}{'density%':>10}")

    rows = []
    for min_item in MIN_ITEM_GRID:
        keep_items = set(items.index[items["n_interactions"] >= min_item])
        sub = df[df["parent_asin"].isin(keep_items)]
        u_dom = sub.groupby("user_id", observed=True)["domain"].nunique()
        for min_dom in MIN_DOMAINS_GRID:
            keep_users = set(u_dom.index[u_dom >= min_dom])
            final = sub[sub["user_id"].isin(keep_users)]
            n_u = final["user_id"].nunique()
            n_i = final["parent_asin"].nunique()
            n_x = len(final)
            dens = 100 * n_x / (n_u * n_i) if n_u and n_i else 0.0
            print(f"  {min_dom:>8}{min_item:>10}{n_u:>12,}{n_i:>10,}{n_x:>15,}{dens:>10.4f}")
            rows.append({"min_domains_per_user": min_dom,
                         "min_interactions_per_item": min_item,
                         "users": n_u, "items": n_i,
                         "interactions": n_x, "density_pct": round(dens, 4)})

    pd.DataFrame(rows).to_csv(OUT_WHATIF, index=False)
    print(f"\nWrote {OUT_USERS}")
    print(f"Wrote {OUT_ITEMS}")
    print(f"Wrote {OUT_WHATIF}")


if __name__ == "__main__":
    main()