"""Encode product text into fixed vectors with a frozen SBERT encoder (§2.3).

Text is assembled from the title, feature bullets, and the opening of the
description, in that order of prominence. The title leads because it is the
most consistently present and most informative field, and because the encoder
truncates beyond a fixed token limit — so the ordering determines what survives
truncation rather than what is merely included.

As with the image encoder, embeddings are written in the row order defined by
item_index.parquet, so row i always corresponds to item index i.

The embedding dimension is fixed by the specific model loaded and must match the
configured value, because the fusion layer's input projection is sized around
it. A mismatch surfaces as a shape error at fusion time at best, and as silently
incorrect embeddings at worst.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from config import load_config
from device_utils import clear_gpu_memory, get_device, vram_report

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
FEATURES = Path(cfg["paths"]["features"])
FEATURES.mkdir(parents=True, exist_ok=True)

ITEM_INDEX = PROCESSED / "item_index.parquet"
ITEM_META = PROCESSED / "items_domained.parquet"
OUT = FEATURES / "text_embeddings.h5"

MODEL_NAME = cfg["features"]["sbert_model"]
EXPECTED_DIM = cfg["features"]["sbert_dim"]
BATCH_SIZE = cfg["features"]["extraction_batch_size"]

MAX_FEATURE_CHARS = 400
MAX_DESCRIPTION_CHARS = 600
MIN_USABLE_CHARS = 10


def build_text(row) -> str:
    """Assemble one product's text, most informative field first."""
    parts = []
    title = (row.title or "").strip()
    if title:
        parts.append(title)

    features = (row.features or "").strip()
    if features:
        parts.append(features[:MAX_FEATURE_CHARS])

    description = (row.description or "").strip()
    if description:
        parts.append(description[:MAX_DESCRIPTION_CHARS])

    return " ".join(parts).strip()


def main() -> None:
    items = pd.read_parquet(ITEM_INDEX).sort_values("item_idx").reset_index(drop=True)
    assert list(items["item_idx"]) == list(range(len(items))), \
        "item_idx must be a contiguous range starting at zero"

    meta = pd.read_parquet(
        ITEM_META, columns=["parent_asin", "title", "features", "description"])
    items = items.merge(meta, on="parent_asin", how="left")
    assert len(items) == len(meta.drop_duplicates("parent_asin").merge(
        items[["parent_asin"]], on="parent_asin")), "metadata join changed row count"

    texts = [build_text(row) for row in items.itertuples()]
    available = np.array([len(t) >= MIN_USABLE_CHARS for t in texts], dtype=bool)

    lengths = np.array([len(t) for t in texts])
    print(f"items to encode : {len(texts):,}")
    print(f"with usable text: {int(available.sum()):,}")
    print(f"text length     : median {np.median(lengths):.0f}  "
          f"mean {lengths.mean():.0f}  max {lengths.max():,} characters")

    device = get_device()
    print(f"device          : {device}")
    print(f"model           : {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME, device=str(device))
    model.eval()
    print(f"max sequence    : {model.max_seq_length} tokens")
    print(f"{vram_report()}\n")

    with torch.no_grad():
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)

    # Items without usable text carry a placeholder zero vector. As with images,
    # absence is handled by the availability mask, never by consuming the zeros.
    embeddings[~available] = 0.0

    clear_gpu_memory()

    print(f"\nencoded             : {int(available.sum()):,}")
    print(f"placeholder rows    : {int((~available).sum()):,}")
    print(f"embedding dimension : {embeddings.shape[1]}")
    assert embeddings.shape[1] == EXPECTED_DIM, (
        f"model produced {embeddings.shape[1]} dimensions but configuration "
        f"specifies {EXPECTED_DIM}; the wrong encoder may have been loaded")
    assert np.isfinite(embeddings).all(), "embeddings contain invalid numbers"

    norms = np.linalg.norm(embeddings[available], axis=1)
    print(f"vector norm         : mean {norms.mean():.3f}  "
          f"min {norms.min():.3f}  max {norms.max():.3f}")

    with h5py.File(OUT, "w") as store:
        store.create_dataset("embeddings", data=embeddings, compression="gzip")
        store.create_dataset("available", data=available)
        store.create_dataset("parent_asin",
                             data=np.array(items["parent_asin"].tolist(),
                                           dtype=h5py.string_dtype()))
        store.attrs["model"] = MODEL_NAME
        store.attrs["dimension"] = EXPECTED_DIM
        store.attrs["row_order"] = "item_idx from item_index.parquet"

    print(f"\nWrote {OUT}  ({OUT.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()