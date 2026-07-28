"""One expensive pass over the raw metadata (manual section 1.3).

Streams meta_Electronics.jsonl.gz exactly once and writes every item to a
compact Parquet file, plus the full category-token vocabulary. Domain
assignment is deliberately NOT done here: it happens later against the
Parquet file, so keyword rules can be revised in seconds without ever
re-reading the 1.2 GB source.
"""
from __future__ import annotations

# Allow this module to import the shared modules at the top of src/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from config import load_config

cfg = load_config()
RAW = Path(cfg["paths"]["raw_data"]) / cfg["sources"]["metadata_file"]
PROCESSED = Path(cfg["paths"]["processed_data"])
REPORTS = Path(cfg["paths"]["reports"])
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

OUT_PARQUET = PROCESSED / "items_all.parquet"
OUT_VOCAB = REPORTS / "category_vocabulary.csv"
OUT_MAINCAT = REPORTS / "main_category_counts.csv"

BATCH = 50_000

SCHEMA = pa.schema([
    ("parent_asin", pa.string()),
    ("title", pa.string()),
    ("description", pa.string()),
    ("features", pa.string()),
    ("main_category", pa.string()),
    ("categories_flat", pa.string()),
    ("image_url", pa.string()),
    ("n_images", pa.int32()),
    ("average_rating", pa.float64()),
    ("rating_number", pa.int64()),
])


def flatten_strings(value) -> list[str]:
    """Flatten arbitrarily nested list/str structures into a flat str list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(flatten_strings(v))
        return out
    return [str(value)]


def pick_image(images) -> tuple[str | None, int]:
    """Return (best large-variant URL, number of usable images).

    Raw format observed: a list of dicts with thumb/large/hi_res/variant.
    Prefer the entry whose variant is MAIN; fall back to the first usable
    large URL. 'large' is preferred over 'hi_res' because CLIP consumes
    224x224 pixels, so hi-res is wasted bandwidth (and is often null).
    """
    urls: list[tuple[str, bool]] = []   # (url, is_main)

    if isinstance(images, dict):        # documented (processed) shape
        for key in ("large", "hi_res", "thumb"):
            for u in flatten_strings(images.get(key)):
                urls.append((u, False))
    elif isinstance(images, list):      # actual raw shape
        for im in images:
            if isinstance(im, dict):
                is_main = str(im.get("variant", "")).upper() == "MAIN"
                for key in ("large", "hi_res", "thumb"):
                    u = im.get(key)
                    if isinstance(u, str) and u:
                        urls.append((u, is_main))
                        break          # one URL per image entry
            elif isinstance(im, str) and im:
                urls.append((im, False))

    if not urls:
        return None, 0
    for u, is_main in urls:
        if is_main:
            return u, len(urls)
    return urls[0][0], len(urls)


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def main() -> None:
    cat_tokens: Counter[str] = Counter()
    main_cats: Counter[str] = Counter()

    cols = {name: [] for name in SCHEMA.names}
    n_total = n_bad = 0
    writer = None

    def flush():
        nonlocal writer, cols
        if not cols["parent_asin"]:
            return
        table = pa.Table.from_pydict(cols, schema=SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(OUT_PARQUET, SCHEMA, compression="snappy")
        writer.write_table(table)
        cols = {name: [] for name in SCHEMA.names}

    print(f"Reading  {RAW}")
    print(f"Writing  {OUT_PARQUET}\n")

    with gzip.open(RAW, "rt", encoding="utf-8") as f:
        for line in tqdm(f, unit=" items", unit_scale=True):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue

            pasin = rec.get("parent_asin")
            if not pasin:
                n_bad += 1
                continue

            cats = [c.strip().lower() for c in flatten_strings(rec.get("categories")) if c.strip()]
            cat_tokens.update(cats)

            mcat = rec.get("main_category")
            main_cats[mcat if mcat else "<none>"] += 1

            img_url, n_img = pick_image(rec.get("images"))

            cols["parent_asin"].append(pasin)
            cols["title"].append(rec.get("title") or "")
            cols["description"].append(" ".join(flatten_strings(rec.get("description"))))
            cols["features"].append(" ".join(flatten_strings(rec.get("features"))))
            cols["main_category"].append(mcat or "")
            cols["categories_flat"].append(" | ".join(cats))
            cols["image_url"].append(img_url)
            cols["n_images"].append(n_img)
            cols["average_rating"].append(as_float(rec.get("average_rating")))
            cols["rating_number"].append(as_int(rec.get("rating_number")))

            n_total += 1
            if len(cols["parent_asin"]) >= BATCH:
                flush()

    flush()
    if writer is not None:
        writer.close()

    with open(OUT_VOCAB, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["category_token", "n_items"])
        for tok, c in cat_tokens.most_common():
            w.writerow([tok, c])

    with open(OUT_MAINCAT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["main_category", "n_items"])
        for tok, c in main_cats.most_common():
            w.writerow([tok, c])

    size_mb = OUT_PARQUET.stat().st_size / 1024 ** 2
    print(f"\nitems written        : {n_total:,}")
    print(f"records skipped      : {n_bad:,}")
    print(f"distinct cat tokens  : {len(cat_tokens):,}")
    print(f"parquet size         : {size_mb:,.0f} MB")
    print(f"vocabulary           : {OUT_VOCAB}")
    print(f"main_category counts : {OUT_MAINCAT}")


if __name__ == "__main__":
    main()