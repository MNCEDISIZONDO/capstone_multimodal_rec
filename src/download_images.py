"""Download and validate product images (manual section 1.9).

Runs after the corpus is settled and before the Ghost split is built, because
Ghost selection requires every candidate to have a verified local image.

Amazon's image URLs are not stable and a meaningful fraction have expired, so
each download is retried with increasing waits and the result is verified to be
a genuine image rather than an error page or placeholder. Every outcome is
recorded in a registry consumed by feature extraction, by the content floor,
and by the data-quality figures reported in the thesis.

The job is resumable: items whose image is already present on disk are skipped.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from config import load_config

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
IMAGES = Path(cfg["paths"]["images"])
REPORTS = Path(cfg["paths"]["reports"])
IMAGES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

CORPUS = PROCESSED / "corpus_items.parquet"
ITEM_META = PROCESSED / "items_domained.parquet"
REGISTRY = REPORTS / "image_registry.csv"

IMG = cfg["images"]
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; image retrieval)"}


def fetch(parent_asin: str, url: str | None) -> dict:
    """Download one image, verify it is usable, and return its outcome."""
    dest = IMAGES / f"{parent_asin}.jpg"

    if dest.exists() and dest.stat().st_size >= IMG["min_bytes"]:
        return {"parent_asin": parent_asin, "status": "cached",
                "bytes": dest.stat().st_size}

    if not url:
        return {"parent_asin": parent_asin, "status": "no_url", "bytes": 0}

    for attempt in range(IMG["max_retries"]):
        try:
            r = requests.get(url, headers=HEADERS,
                             timeout=IMG["timeout_seconds"])
            if r.status_code != 200:
                time.sleep(2 ** attempt)
                continue

            data = r.content
            if len(data) < IMG["min_bytes"]:
                return {"parent_asin": parent_asin, "status": "too_small",
                        "bytes": len(data)}

            img = Image.open(BytesIO(data))
            img.verify()
            img = Image.open(BytesIO(data))
            if min(img.size) < IMG["min_pixels"]:
                return {"parent_asin": parent_asin, "status": "too_small",
                        "bytes": len(data), "width": img.size[0],
                        "height": img.size[1]}

            img.convert("RGB").save(dest, "JPEG", quality=90)
            return {"parent_asin": parent_asin, "status": "success",
                    "bytes": dest.stat().st_size,
                    "width": img.size[0], "height": img.size[1]}

        except requests.RequestException:
            time.sleep(2 ** attempt)
        except Exception:
            return {"parent_asin": parent_asin, "status": "invalid_image",
                    "bytes": 0}

    return {"parent_asin": parent_asin, "status": "failed", "bytes": 0}


def main() -> None:
    corpus = pd.read_parquet(CORPUS, columns=["parent_asin"])
    meta = pd.read_parquet(ITEM_META, columns=["parent_asin", "image_url"])
    work = corpus.merge(meta, on="parent_asin", how="left")

    print(f"items in corpus     : {len(work):,}")
    print(f"items with an image URL: {work['image_url'].notna().sum():,}")
    already = sum(1 for a in work["parent_asin"] if (IMAGES / f"{a}.jpg").exists())
    print(f"already downloaded  : {already:,}")
    print(f"destination         : {IMAGES}\n")

    results = []
    with ThreadPoolExecutor(max_workers=IMG["max_workers"]) as pool:
        futures = [pool.submit(fetch, row.parent_asin, row.image_url)
                   for row in work.itertuples()]
        for f in tqdm(as_completed(futures), total=len(futures), unit=" items"):
            results.append(f.result())

    reg = pd.DataFrame(results)
    reg.to_csv(REGISTRY, index=False)

    print("\nOUTCOMES")
    for status, n in reg["status"].value_counts().items():
        print(f"  {status:<15}{n:>7,}  ({n / len(reg):6.1%})")

    usable = reg["status"].isin(["success", "cached"]).sum()
    print(f"\nusable images       : {usable:,}  ({usable / len(reg):.1%})")
    print(f"Wrote {REGISTRY}")


if __name__ == "__main__":
    main()