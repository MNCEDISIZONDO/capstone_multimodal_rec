"""Which way of using multiple product views carries the most preference signal?

Products have around four images each. Averaging them was measured to push
different products closer together, which is the opposite of what ranking needs.
This compares several strategies on the metric that predicted the earlier
results: whether items that are neighbours in embedding space are bought by the
same people.

No training required. The winner, if any, determines how features are extracted
in a rebuild.
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
from itertools import combinations
from pathlib import Path

import h5py
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
FEATURES = Path(cfg["paths"]["features"])
REPORTS = Path(cfg["paths"]["reports"])

N_ITEMS = 3000
MAX_VIEWS = 4
ENCODE_BATCH = 64
K_NEIGHBOURS = 20
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; image retrieval)"}


def collect_urls(wanted: set[str]) -> dict[str, list[str]]:
    path = RAW / cfg["sources"]["metadata_file"]
    found: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if len(found) >= len(wanted):
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = record.get("parent_asin")
            if asin not in wanted or asin in found:
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
            if urls:
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


def copurchase_lift(vectors: np.ndarray, item_ids: np.ndarray,
                    pairs: set[tuple[int, int]], device) -> tuple[float, float]:
    """Share of nearest neighbours that are co-purchased, and mean similarity."""
    normalised = torch.as_tensor(unit(vectors), dtype=torch.float32, device=device)
    hits = considered = 0
    similarity_sum = 0.0
    n_similarity = 0

    for start in range(0, len(item_ids), 512):
        block = normalised[start:start + 512]
        similarity = block @ normalised.T
        for offset in range(len(block)):
            similarity[offset, start + offset] = -np.inf
        top = similarity.topk(K_NEIGHBOURS, dim=1)
        neighbours = top.indices.cpu().numpy()
        similarity_sum += float(top.values.sum())
        n_similarity += top.values.numel()

        for offset, row in enumerate(neighbours):
            source = int(item_ids[start + offset])
            for neighbour in row:
                target = int(item_ids[neighbour])
                pair = (min(source, target), max(source, target))
                hits += pair in pairs
                considered += 1

    return hits / considered, similarity_sum / n_similarity


def main() -> None:
    device = get_device()
    rng = np.random.default_rng(cfg["seeds"]["split_construction"])

    items = pd.read_parquet(PROCESSED / "item_index.parquet").sort_values("item_idx")
    item_of = dict(zip(items["parent_asin"], items["item_idx"]))
    users = pd.read_parquet(PROCESSED / "user_index.parquet")
    user_of = dict(zip(users["user_id"], users["user_idx"]))

    train = pd.read_parquet(PROCESSED / "split_train.parquet")
    train_users = train["user_id"].map(user_of).to_numpy(dtype=np.int64)
    train_items = train["parent_asin"].map(item_of).to_numpy(dtype=np.int64)

    trained = np.array(sorted(set(train_items.tolist())))
    sample_idx = rng.choice(trained, size=min(N_ITEMS, len(trained)), replace=False)
    asin_of = dict(zip(items["item_idx"], items["parent_asin"]))
    wanted = {asin_of[i] for i in sample_idx}

    print(f"scanning metadata for {len(wanted):,} items")
    started = time.time()
    urls = collect_urls(wanted)
    print(f"  found {len(urls):,}   ({time.time() - started:.0f}s)")

    jobs = [(asin, url) for asin, links in urls.items() for url in links]
    print(f"\ndownloading {len(jobs):,} images")
    started = time.time()
    fetched: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=32) as pool:
        for (asin, _), image in zip(jobs, pool.map(fetch, [u for _, u in jobs])):
            if image is not None:
                fetched.setdefault(asin, []).append(image)

    fetched = {a: v for a, v in fetched.items() if len(v) >= MAX_VIEWS}
    print(f"  {len(fetched):,} items with {MAX_VIEWS} usable views "
          f"({time.time() - started:.0f}s)")

    if len(fetched) < 500:
        raise SystemExit("too few items with a full set of views")

    print("\nencoding")
    processor = CLIPImageProcessor.from_pretrained(cfg["features"]["clip_model"])
    model = CLIPModel.from_pretrained(cfg["features"]["clip_model"]).to(device).eval()

    asins = list(fetched)
    flat = [image for asin in asins for image in fetched[asin][:MAX_VIEWS]]
    encoded = []
    with torch.no_grad():
        for start in range(0, len(flat), ENCODE_BATCH):
            pixels = processor(images=flat[start:start + ENCODE_BATCH],
                               return_tensors="pt")["pixel_values"].to(device)
            output = model.get_image_features(pixel_values=pixels)
            vectors = output if isinstance(output, torch.Tensor) else output.pooler_output
            encoded.append(vectors.float().cpu().numpy())
    encoded = np.concatenate(encoded).reshape(len(asins), MAX_VIEWS, -1)
    clear_gpu_memory()
    print(f"  {len(flat):,} images, shape {encoded.shape}")

    item_ids = np.array([item_of[a] for a in asins])

    # co-purchase pairs restricted to the sampled items
    sampled = set(item_ids.tolist())
    pairs: set[tuple[int, int]] = set()
    frame = pd.DataFrame({"user": train_users, "item": train_items})
    for _, group in frame.groupby("user"):
        basket = [i for i in group["item"].unique() if int(i) in sampled]
        if len(basket) < 2:
            continue
        if len(basket) > 12:
            basket = rng.choice(basket, size=12, replace=False)
        for a, b in combinations(sorted(basket), 2):
            pairs.add((int(a), int(b)))
    print(f"  co-purchase pairs among sampled items: {len(pairs):,}")

    random_hits = 0
    trials = 200_000
    for _ in range(trials):
        a, b = rng.choice(item_ids, size=2, replace=False)
        pairs_key = (min(int(a), int(b)), max(int(a), int(b)))
        random_hits += pairs_key in pairs
    random_rate = random_hits / trials
    print(f"  random baseline: {random_rate:.4%}\n")

    strategies = {
        "main view only": encoded[:, 0, :],
        "second view": encoded[:, 1, :],
        "third view": encoded[:, 2, :],
        "fourth view": encoded[:, 3, :],
        "mean of four": encoded.mean(axis=1),
        "max-pool of four": encoded.max(axis=1),
        "mean of views 2-4": encoded[:, 1:, :].mean(axis=1),
        "concatenate four": encoded.reshape(len(asins), -1),
    }

    print("CO-PURCHASE LIFT BY STRATEGY")
    print(f"  {'strategy':<22}{'neighbours':>12}{'lift':>8}{'similarity':>13}")

    rows = []
    for name, vectors in strategies.items():
        rate, similarity = copurchase_lift(vectors, item_ids, pairs, device)
        lift = rate / max(random_rate, 1e-12)
        print(f"  {name:<22}{rate:>11.3%}{lift:>7.1f}x{similarity:>13.4f}")
        rows.append({"strategy": name, "neighbour_rate": round(rate, 5),
                     "lift": round(lift, 2),
                     "mean_neighbour_similarity": round(similarity, 4)})

    # text baseline on the same items, for reference
    with h5py.File(FEATURES / "text_embeddings.h5", "r") as store:
        text = store["embeddings"][:]
    rate, similarity = copurchase_lift(text[item_ids], item_ids, pairs, device)
    lift = rate / max(random_rate, 1e-12)
    print(f"  {'text (reference)':<22}{rate:>11.3%}{lift:>7.1f}x{similarity:>13.4f}")
    rows.append({"strategy": "text (reference)", "neighbour_rate": round(rate, 5),
                 "lift": round(lift, 2),
                 "mean_neighbour_similarity": round(similarity, 4)})

    frame = pd.DataFrame(rows).sort_values("lift", ascending=False)
    best_image = frame[frame["strategy"] != "text (reference)"].iloc[0]
    current = frame[frame["strategy"] == "main view only"].iloc[0]
    text_row = frame[frame["strategy"] == "text (reference)"].iloc[0]

    print("\nVERDICT")
    print(f"  current extraction : {current['lift']:.1f}x")
    print(f"  best strategy      : {best_image['strategy']} at {best_image['lift']:.1f}x")
    print(f"  text reference     : {text_row['lift']:.1f}x")

    gain = (best_image["lift"] - current["lift"]) / current["lift"] * 100
    if gain > 15:
        print(f"\n  A {gain:.0f}% improvement over current extraction. Rebuilding")
        print("  features with this strategy is justified.")
    elif gain > 5:
        print(f"\n  A {gain:.0f}% improvement, modest. Rebuilding may not change the")
        print("  ranking result materially.")
    else:
        print(f"\n  No strategy meaningfully improves on the main view ({gain:+.0f}%).")
        print("  Multiple views do not carry additional preference signal, and the")
        print("  original extraction stands.")

    output = REPORTS / "view_strategy_comparison.csv"
    frame.to_csv(output, index=False)
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()