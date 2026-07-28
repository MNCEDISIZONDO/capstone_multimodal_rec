"""Encode product images into fixed vectors with a frozen CLIP encoder (§2.2).

Extraction runs once, ahead of training. The recommender therefore never loads
CLIP, which is what makes the project feasible within a 6 GB memory budget:
only the small saved vectors are needed at training time.

Embeddings are written in the row order defined by item_index.parquet, so that
row i of the matrix always corresponds to item index i. This removes an entire
class of silent failure in which features are fetched for the wrong item and
training proceeds without error.

Items whose image could not be retrieved receive a placeholder zero vector and
are flagged in the stored availability mask. The placeholder exists only to keep
the matrix rectangular; the model must consult the mask and never consume the
zeros as though they were a real image.
"""
from __future__ import annotations

# Allow this module to import the shared modules at the top of src/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel

from config import load_config
from device_utils import clear_gpu_memory, get_device, vram_report

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
IMAGES = Path(cfg["paths"]["images"])
FEATURES = Path(cfg["paths"]["features"])
FEATURES.mkdir(parents=True, exist_ok=True)

ITEM_INDEX = PROCESSED / "item_index.parquet"
OUT = FEATURES / "image_embeddings.h5"

MODEL_NAME = cfg["features"]["clip_model"]
EXPECTED_DIM = cfg["features"]["clip_dim"]
BATCH_SIZE = cfg["features"]["extraction_batch_size"]
USE_FP16 = cfg["features"]["use_fp16"]
NUM_WORKERS = 4


class ProductImages(Dataset):
    """Yields preprocessed image tensors and an availability flag per item."""

    def __init__(self, parent_asins: list[str], processor: CLIPImageProcessor):
        self.parent_asins = parent_asins
        self.processor = processor
        size = cfg["features"]["image_size"]
        self.blank = torch.zeros(3, size, size)

    def __len__(self) -> int:
        return len(self.parent_asins)

    def __getitem__(self, i: int):
        path = IMAGES / f"{self.parent_asins[i]}.jpg"
        if not path.exists():
            return self.blank, False
        try:
            with Image.open(path) as img:
                pixels = self.processor(images=img.convert("RGB"),
                                        return_tensors="pt")["pixel_values"][0]
            return pixels, True
        except Exception:
            # A file that cannot be decoded is treated as a missing image
            # rather than allowed to halt a long batch job.
            return self.blank, False


def main() -> None:
    items = pd.read_parquet(ITEM_INDEX).sort_values("item_idx").reset_index(drop=True)
    assert list(items["item_idx"]) == list(range(len(items))), \
        "item_idx must be a contiguous range starting at zero"
    parent_asins = items["parent_asin"].tolist()
    print(f"items to encode : {len(parent_asins):,}")

    device = get_device()
    print(f"device          : {device}")
    print(f"model           : {MODEL_NAME}")

    processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    if USE_FP16 and device.type == "cuda":
        model = model.half()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(f"{vram_report()}\n")

    loader = DataLoader(
        ProductImages(parent_asins, processor),
        batch_size=BATCH_SIZE,
        shuffle=False,               # order must match item_idx
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    embeddings = np.zeros((len(parent_asins), EXPECTED_DIM), dtype=np.float32)
    available = np.zeros(len(parent_asins), dtype=bool)
    position = 0

    with torch.no_grad():
        for pixels, present in tqdm(loader, unit=" batch"):
            pixels = pixels.to(device, non_blocking=True)
            if USE_FP16 and device.type == "cuda":
                pixels = pixels.half()

            
            output = model.get_image_features(pixel_values=pixels)
            features = output if isinstance(output, torch.Tensor) else output.pooler_output
            batch = features.float().cpu().numpy()

            n = len(batch)
            embeddings[position:position + n] = batch
            available[position:position + n] = present.numpy()
            position += n

    # Zero the placeholder rows explicitly: the encoder produces a non-zero
    # vector even for a blank input, and that vector carries no information
    # about the product.
    embeddings[~available] = 0.0

    clear_gpu_memory()

    print(f"\nencoded             : {int(available.sum()):,}")
    print(f"placeholder rows    : {int((~available).sum()):,}")
    print(f"embedding dimension : {embeddings.shape[1]}")
    assert embeddings.shape[1] == EXPECTED_DIM, \
        f"expected {EXPECTED_DIM} dimensions, produced {embeddings.shape[1]}"
    assert np.isfinite(embeddings).all(), "embeddings contain invalid numbers"

    norms = np.linalg.norm(embeddings[available], axis=1)
    print(f"vector norm         : mean {norms.mean():.3f}  "
          f"min {norms.min():.3f}  max {norms.max():.3f}")

    with h5py.File(OUT, "w") as store:
        store.create_dataset("embeddings", data=embeddings, compression="gzip")
        store.create_dataset("available", data=available)
        store.create_dataset("parent_asin",
                             data=np.array(parent_asins, dtype=h5py.string_dtype()))
        store.attrs["model"] = MODEL_NAME
        store.attrs["dimension"] = EXPECTED_DIM
        store.attrs["row_order"] = "item_idx from item_index.parquet"

    print(f"\nWrote {OUT}  ({OUT.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()