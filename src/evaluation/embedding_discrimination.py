"""Do the content embeddings carry preference signal, or only category signal?

Feature validation in Phase 2 measured whether embeddings separate product
domains. Ranking requires something stricter: separating items within a domain,
and doing so in a way that aligns with what users actually buy together.

Two measurements. The first is the spread of pairwise similarity within a
domain: a narrow distribution means the encoder assigns near-identical
representations to different products, leaving nothing to rank on. The second
compares each item's nearest neighbours in embedding space against the items it
is genuinely co-purchased with, which is the property a content-based
recommender depends on.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from config import load_config
from device_utils import get_device

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
FEATURES = Path(cfg["paths"]["features"])
REPORTS = Path(cfg["paths"]["reports"])

K_NEIGHBOURS = 20
MAX_PAIRS_PER_USER = 50


def normalise(matrix: np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(matrix, dtype=torch.float32)
    return tensor / tensor.norm(dim=1, keepdim=True).clamp_min(1e-9)


def main() -> None:
    device = get_device()

    items = pd.read_parquet(PROCESSED / "item_index.parquet").sort_values("item_idx")
    domains = items["domain"].to_numpy()

    with h5py.File(FEATURES / "image_embeddings.h5", "r") as store:
        image = store["embeddings"][:]
        image_available = store["available"][:]
    with h5py.File(FEATURES / "text_embeddings.h5", "r") as store:
        text = store["embeddings"][:]

    train = pd.read_parquet(PROCESSED / "split_train.parquet")
    item_of = dict(zip(items["parent_asin"], items["item_idx"]))
    user_of = dict(zip(
        pd.read_parquet(PROCESSED / "user_index.parquet")["user_id"],
        pd.read_parquet(PROCESSED / "user_index.parquet")["user_idx"]))
    train_users = train["user_id"].map(user_of).to_numpy(dtype=np.int64)
    train_items = train["parent_asin"].map(item_of).to_numpy(dtype=np.int64)

    print("SIMILARITY SPREAD WITHIN EACH DOMAIN")
    print("  A narrow spread means different products receive near-identical")
    print("  representations, leaving nothing for a ranker to separate.\n")
    print(f"  {'domain':<22}{'image mean':>12}{'image s.d.':>12}"
          f"{'text mean':>12}{'text s.d.':>12}")

    rng = np.random.default_rng(cfg["seeds"]["split_construction"])
    spread_rows = []

    for domain in cfg["data"]["domains"]:
        inside = np.flatnonzero((domains == domain) & image_available)
        if len(inside) < 100:
            continue
        sample = rng.choice(inside, size=min(1500, len(inside)), replace=False)

        row = {"domain": domain}
        for name, matrix in (("image", image), ("text", text)):
            vectors = normalise(matrix[sample]).to(device)
            similarity = (vectors @ vectors.T).cpu().numpy()
            upper = similarity[np.triu_indices(len(sample), k=1)]
            row[f"{name}_mean"] = float(upper.mean())
            row[f"{name}_std"] = float(upper.std())

        print(f"  {domain:<22}{row['image_mean']:>12.4f}{row['image_std']:>12.4f}"
              f"{row['text_mean']:>12.4f}{row['text_std']:>12.4f}")
        spread_rows.append(row)

    print("\nDO NEAREST NEIGHBOURS SHARE BUYERS?")
    print("  For each item, the K most similar items in embedding space are")
    print("  compared against the items it is genuinely co-purchased with.")
    print("  A content-based recommender depends on these agreeing.\n")

    # Co-purchase pairs from training interactions.
    co_purchased: set[tuple[int, int]] = set()
    frame = pd.DataFrame({"user": train_users, "item": train_items})
    for _, group in frame.groupby("user"):
        basket = group["item"].unique()
        if len(basket) < 2:
            continue
        if len(basket) > 12:
            basket = rng.choice(basket, size=12, replace=False)
        for a, b in combinations(sorted(basket), 2):
            co_purchased.add((int(a), int(b)))

    print(f"  co-purchase pairs observed : {len(co_purchased):,}")

    trained_items = np.array(sorted(set(train_items.tolist())))
    evaluated = trained_items[np.isin(trained_items, np.flatnonzero(image_available))]
    print(f"  items evaluated            : {len(evaluated):,}\n")

    results = {}
    for name, matrix in (("image", image), ("text", text)):
        vectors = normalise(matrix[evaluated]).to(device)
        hits = 0
        considered = 0

        for start in range(0, len(evaluated), 512):
            block = vectors[start:start + 512]
            similarity = block @ vectors.T
            # Exclude each item from its own neighbour list.
            for offset in range(len(block)):
                similarity[offset, start + offset] = -np.inf
            neighbours = similarity.topk(K_NEIGHBOURS, dim=1).indices.cpu().numpy()

            for offset, row in enumerate(neighbours):
                source = int(evaluated[start + offset])
                for neighbour in row:
                    target = int(evaluated[neighbour])
                    pair = (min(source, target), max(source, target))
                    hits += pair in co_purchased
                    considered += 1

        results[name] = hits / considered
        print(f"  {name:<8}{hits / considered:.4%} of nearest neighbours "
              f"are co-purchased")

    # Random pairs give the rate expected with no signal at all.
    random_hits = 0
    trials = 200_000
    for _ in range(trials):
        a, b = rng.choice(evaluated, size=2, replace=False)
        pair = (min(int(a), int(b)), max(int(a), int(b)))
        random_hits += pair in co_purchased
    results["random"] = random_hits / trials
    print(f"  {'random':<8}{random_hits / trials:.4%} baseline\n")

    print("  lift over random")
    for name in ("image", "text"):
        print(f"    {name:<8}{results[name] / max(results['random'], 1e-12):>6.1f}x")

    pd.DataFrame(spread_rows).to_csv(REPORTS / "embedding_spread.csv", index=False)
    pd.DataFrame([results]).to_csv(REPORTS / "neighbour_copurchase.csv", index=False)
    print(f"\nWrote {REPORTS / 'embedding_spread.csv'}")
    print(f"Wrote {REPORTS / 'neighbour_copurchase.csv'}")


if __name__ == "__main__":
    main()