"""Do additional product images carry information the main image lacks?

Phase 1 kept one image per product, the MAIN variant. This samples products with
several images, encodes each separately, and measures two things: how similar a
product's own images are to each other, and whether averaging them separates
different products better than the main image alone.

If a product's images are near-identical, the extra views are redundant and
single-image extraction was correct. If they differ substantially, information
was discarded.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel

from config import load_config
from device_utils import clear_gpu_memory, get_device

cfg = load_config()
RAW = Path(cfg["paths"]["raw_data"])
PROCESSED = Path(cfg["paths"]["processed_data"])
REPORTS = Path(cfg["paths"]["reports"])

N_PRODUCTS = 300
MAX_VIEWS = 4
ENCODE_BATCH = 64
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; image retrieval)"}


def collect_urls(corpus_asins: set[str]) -> dict[str, list[str]]:
    """Pull every large-variant URL for the sampled products from the raw metadata."""
    path = RAW / cfg["sources"]["metadata_file"]
    found: dict[str, list[str]] = {}

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if len(found) >= len(corpus_asins):
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = record.get("parent_asin")
            if asin not in corpus_asins or asin in found:
                continue

            urls = []
            images = record.get("images")
            if isinstance(images, list):
                for entry in images:
                    if isinstance(entry, dict):
                        url = entry.get("large") or entry.get("hi_res")
                        if isinstance(url, str) and url:
                            urls.append(url)
            elif isinstance(images, dict):
                for url in (images.get("large") or []):
                    if isinstance(url, str) and url:
                        urls.append(url)

            if len(urls) >= 2:
                found[asin] = urls[:MAX_VIEWS]

    return found


def fetch(url: str):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code != 200 or len(response.content) < 2000:
            return None
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return None


def unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-9, None)


def cross_product_similarity(matrix: np.ndarray) -> np.ndarray:
    similarity = matrix @ matrix.T
    return similarity[np.triu_indices(len(matrix), k=1)]


def main() -> None:
    device = get_device()
    rng = np.random.default_rng(cfg["seeds"]["split_construction"])

    items = pd.read_parquet(PROCESSED / "corpus_items_final.parquet")
    sample = rng.choice(items["parent_asin"].to_numpy(),
                        size=min(N_PRODUCTS * 2, len(items)), replace=False)

    print(f"scanning metadata for {len(sample):,} sampled products")
    started = time.time()
    urls = collect_urls(set(sample.tolist()))
    print(f"  {len(urls):,} have two or more images   ({time.time() - started:.0f}s)")

    counts = pd.Series([len(v) for v in urls.values()])
    print(f"  images available per product: median {counts.median():.0f}, "
          f"mean {counts.mean():.1f}, max {counts.max()}\n")

    selected = dict(list(urls.items())[:N_PRODUCTS])

    # download everything concurrently, then encode in batches
    jobs = [(asin, url) for asin, links in selected.items() for url in links]
    print(f"downloading {len(jobs):,} images for {len(selected):,} products")
    started = time.time()

    fetched: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=32) as pool:
        results = pool.map(fetch, [url for _, url in jobs])
        for (asin, _), image in zip(jobs, results):
            if image is not None:
                fetched.setdefault(asin, []).append(image)

    fetched = {a: v for a, v in fetched.items() if len(v) >= 2}
    print(f"  {len(fetched):,} products with two or more usable images "
          f"({time.time() - started:.0f}s)\n")

    if not fetched:
        raise SystemExit("no products with multiple usable images")

    print("encoding")
    processor = CLIPImageProcessor.from_pretrained(cfg["features"]["clip_model"])
    model = CLIPModel.from_pretrained(cfg["features"]["clip_model"]).to(device).eval()

    flat_images, owners = [], []
    for asin, images in fetched.items():
        flat_images.extend(images)
        owners.extend([asin] * len(images))

    encoded = []
    with torch.no_grad():
        for start in range(0, len(flat_images), ENCODE_BATCH):
            batch = flat_images[start:start + ENCODE_BATCH]
            pixels = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
            output = model.get_image_features(pixel_values=pixels)
            vectors = output if isinstance(output, torch.Tensor) else output.pooler_output
            encoded.append(vectors.float().cpu().numpy())

    encoded = np.concatenate(encoded)
    clear_gpu_memory()

    per_product: dict[str, np.ndarray] = {}
    for asin, vector in zip(owners, encoded):
        per_product.setdefault(asin, []).append(vector)
    per_product = {a: np.stack(v) for a, v in per_product.items()}
    print(f"  encoded {len(flat_images):,} images across "
          f"{len(per_product):,} products\n")

    within = []
    for vectors in per_product.values():
        normalised = unit(vectors)
        similarity = normalised @ normalised.T
        within.extend(similarity[np.triu_indices(len(normalised), k=1)].tolist())
    within = np.array(within)

    main_only = unit(np.stack([v[0] for v in per_product.values()]))
    averaged = unit(np.stack([v.mean(axis=0) for v in per_product.values()]))
    main_cross = cross_product_similarity(main_only)
    averaged_cross = cross_product_similarity(averaged)

    print("SIMILARITY BETWEEN A PRODUCT'S OWN IMAGES")
    print(f"  pairs compared : {len(within):,}")
    print(f"  mean           : {within.mean():.4f}")
    print(f"  median         : {np.median(within):.4f}")
    print(f"  s.d.           : {within.std():.4f}")
    print(f"  below 0.80     : {(within < 0.80).mean():.1%}\n")

    print("SIMILARITY BETWEEN DIFFERENT PRODUCTS")
    print(f"  main image only : mean {main_cross.mean():.4f}   "
          f"s.d. {main_cross.std():.4f}")
    print(f"  averaged views  : mean {averaged_cross.mean():.4f}   "
          f"s.d. {averaged_cross.std():.4f}\n")

    print("VERDICT")
    if within.mean() > 0.90:
        print("  A product's images are near-identical. The additional views are")
        print("  redundant and single-image extraction was the correct choice.")
    elif within.mean() > 0.80:
        print("  A product's images are similar but not identical. Averaging may")
        print("  reduce noise, but is unlikely to change the result substantially.")
    else:
        print("  A product's images differ substantially. Information was discarded")
        print("  by keeping only the main view, and re-extraction is justified.")

    separation = main_cross.mean() - averaged_cross.mean()
    print(f"\n  averaging changes between-product similarity by {separation:+.4f}")
    if separation > 0.02:
        print("  Averaging separates different products further, which is what a")
        print("  ranker needs.")
    elif separation < -0.02:
        print("  Averaging compresses products together, which would make ranking")
        print("  harder rather than easier.")
    else:
        print("  Averaging leaves product separation essentially unchanged.")

    output = REPORTS / "multi_image_test.csv"
    pd.DataFrame([{
        "products_encoded": len(per_product),
        "images_encoded": len(flat_images),
        "within_product_mean": round(float(within.mean()), 4),
        "within_product_median": round(float(np.median(within)), 4),
        "within_product_sd": round(float(within.std()), 4),
        "share_below_080": round(float((within < 0.80).mean()), 4),
        "cross_product_main_mean": round(float(main_cross.mean()), 4),
        "cross_product_main_sd": round(float(main_cross.std()), 4),
        "cross_product_averaged_mean": round(float(averaged_cross.mean()), 4),
        "cross_product_averaged_sd": round(float(averaged_cross.std()), 4),
    }]).to_csv(output, index=False)
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()