"""Assign items to the five domains (manual section 1.4).

Runs against the local Parquet file, never the raw source, so rules can be
revised and re-run in seconds. Also emits the title audit section 1.4 requires.

Three-stage filter, in order:
  1. LEAF veto     — the last token of the hierarchical category path is what
     the item IS. A laptop ends in 'traditional laptops'; a laptop bag ends in
     'bags, cases & sleeves'. Both share the ancestor 'computers &
     accessories', so vetoing on ancestors would reject the whole catalogue.
  2. TITLE veto    — Amazon files straps, cases and chargers under the device
     leaf they accompany. Only titles catch these.
  3. TITLE require — the title must contain a domain-relevant term. This is
     the robust stage: it kills outright metadata errors (a toothbrush filed
     under 'speakers') without needing them to be anticipated.
"""
from __future__ import annotations

# Allow this module to import the shared modules at the top of src/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import random
from pathlib import Path

import pandas as pd

from config import load_config

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
REPORTS = Path(cfg["paths"]["reports"])
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

IN_PARQUET = PROCESSED / "items_all.parquet"
OUT_PARQUET = PROCESSED / "items_domained.parquet"
OUT_AUDIT = REPORTS / "domain_title_audit.txt"
OUT_COUNTS = REPORTS / "domain_counts.csv"

rules = cfg["domain_rules"]
EXCLUDE = [s.lower() for s in rules["exclude_contains"]]
EXCLUDE_TITLE = [s.lower() for s in rules.get("exclude_title_contains", [])]
REQUIRE_TITLE = {d: [s.lower() for s in toks]
                 for d, toks in rules.get("require_title_contains", {}).items()}
INCLUDE = {d: {t.lower() for t in toks} for d, toks in rules["include_exact"].items()}
PRIORITY = cfg["domain_priority"]

SEED = cfg["seeds"]["split_construction"]
N_AUDIT = 25


def assign(cats_flat: str, title: str) -> tuple[str | None, int]:
    """Return (domain, n_domains_matched). Domain is None if out of scope."""
    if not cats_flat:
        return None, 0
    tokens = [t.strip() for t in cats_flat.split("|") if t.strip()]
    if not tokens:
        return None, 0

    # Stage 1: leaf veto
    leaf = tokens[-1]
    for bad in EXCLUDE:
        if bad in leaf:
            return None, 0

    # Stage 2: title veto
    t = (title or "").lower()
    for bad in EXCLUDE_TITLE:
        if bad in t:
            return None, 0

    token_set = set(tokens)
    matched = [d for d in PRIORITY if INCLUDE[d] & token_set]
    if not matched:
        return None, 0

    # Stage 3: title must affirm the domain
    domain = matched[0]
    required = REQUIRE_TITLE.get(domain)
    if required and not any(term in t for term in required):
        return None, 0

    return domain, len(matched)


def main() -> None:
    print(f"Reading {IN_PARQUET}")
    df = pd.read_parquet(
        IN_PARQUET,
        columns=["parent_asin", "title", "description", "features",
                 "categories_flat", "image_url", "n_images"],
    )
    print(f"  {len(df):,} items loaded\n")

    assigned = [assign(c, t) for c, t in zip(df["categories_flat"], df["title"])]
    df["domain"] = [a[0] for a in assigned]
    df["n_domains_matched"] = [a[1] for a in assigned]
    kept = df[df["domain"].notna()].copy()

    multi = int((kept["n_domains_matched"] > 1).sum())
    print(f"items matching >1 domain (resolved by priority): {multi:,}\n")

    print("DOMAIN DISTRIBUTION")
    counts = kept["domain"].value_counts()
    for d in cfg["data"]["domains"]:
        print(f"  {d:<22} {int(counts.get(d, 0)):>8,}")
    print(f"  {'TOTAL IN SCOPE':<22} {len(kept):>8,}")
    print(f"  {'dropped (out of scope)':<22} {len(df) - len(kept):>8,}")

    if len(counts) >= 2:
        print(f"\n  largest:smallest ratio = {counts.max() / counts.min():.1f}:1")

    target = cfg["data"]["corpus_size_max"]
    per_domain = target / len(cfg["data"]["domains"])
    print(f"  smallest domain vs per-domain need ({per_domain:,.0f}): "
          f"{counts.min() / per_domain:.1f}x headroom")

    print("\nCONTENT AVAILABILITY (in-scope items)")
    for col in ["title", "description", "features"]:
        print(f"  {col:<12} {(kept[col].str.len() > 0).mean():6.1%}")
    print(f"  {'image url':<12} {kept['image_url'].notna().mean():6.1%}")

    kept.to_parquet(OUT_PARQUET, index=False)
    print(f"\nWrote {OUT_PARQUET}  ({len(kept):,} rows)")

    counts.rename_axis("domain").reset_index(name="n_items").to_csv(OUT_COUNTS, index=False)
    print(f"Wrote {OUT_COUNTS}")

    # Section 1.4 title audit
    rng = random.Random(SEED)
    with open(OUT_AUDIT, "w", encoding="utf-8") as fh:
        for d in cfg["data"]["domains"]:
            sub = kept[kept["domain"] == d]
            fh.write(f"\n{'=' * 70}\n{d.upper()}  ({len(sub):,} items)\n{'=' * 70}\n")
            if sub.empty:
                fh.write("  NO ITEMS — keyword vocabulary too narrow.\n")
                continue
            idx = rng.sample(range(len(sub)), min(N_AUDIT, len(sub)))
            for j in idx:
                fh.write(f"  - {sub.iloc[j]['title'][:110]}\n")
    print(f"Wrote {OUT_AUDIT}")


if __name__ == "__main__":
    main()