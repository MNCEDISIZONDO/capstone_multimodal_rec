

from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from config import load_config

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
IMAGES = Path(cfg["paths"]["images"])
REPORTS = Path(cfg["paths"]["reports"])

EXPECTED = {
    "split_train.parquet": 93_701,
    "split_validation.parquet": 15_794,
    "split_test.parquet": 15_794,
    "split_ghost_evaluation.parquet": 25_656,
    "corpus_items_final.parquet": 9_503,
    "corpus_interactions_final.parquet": 191_763,
    "ghost_items.parquet": 1_901,
    "user_index.parquet": 15_794,
    "item_index.parquet": 9_503,
}

failures: list[str] = []


def check(label: str, passed: bool, detail: str = "") -> None:
    print(f"  {'PASS' if passed else 'FAIL'}  {label:<44}{detail}")
    if not passed:
        failures.append(label)


print("ENVIRONMENT")
check("CUDA available", torch.cuda.is_available(), torch.__version__)
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"        VRAM free {free / 1024**3:.2f} GB of {total / 1024**3:.2f} GB")
    print(f"        device {torch.cuda.get_device_name(0)}")

print("\nARTIFACT PRESENCE AND ROW COUNTS")
for name, expected_rows in EXPECTED.items():
    path = PROCESSED / name
    if not path.exists():
        check(name, False, "missing")
        continue
    n = len(pd.read_parquet(path))
    check(name, n == expected_rows, f"{n:,} rows (expected {expected_rows:,})")

print("\nFROZEN SPLIT INTEGRITY")
checksum_file = REPORTS / "split_checksums.json"
if not checksum_file.exists():
    check("split_checksums.json", False, "missing")
else:
    record = json.loads(checksum_file.read_text())
    check("recorded construction seed",
          record.get("seed") == cfg["seeds"]["split_construction"],
          f"seed {record.get('seed')}")
    for name, expected_hash in record["sha256"].items():
        path = PROCESSED / f"split_{name}.parquet"
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
        check(f"checksum {name}", actual == expected_hash, actual[:16] or "missing")

print("\nIMAGE CORPUS")
n_images = len(list(IMAGES.glob("*.jpg")))
check("image files on disk", n_images == 9_479, f"{n_images:,}")

registry_path = REPORTS / "image_registry.csv"
if registry_path.exists():
    registry = pd.read_csv(registry_path)
    usable = registry["status"].isin(["success", "cached"]).sum()
    check("registry usable images", usable == 9_479, f"{usable:,}")
else:
    check("image_registry.csv", False, "missing")

print("\nCOLD-START INTEGRITY")
train = pd.read_parquet(PROCESSED / "split_train.parquet")
ghost_ids = set(pd.read_parquet(PROCESSED / "ghost_items.parquet")["parent_asin"])
leakage = int(train["parent_asin"].isin(ghost_ids).sum())
check("cold-start leakage into training", leakage == 0, f"{leakage} rows")

ghost_eval = pd.read_parquet(PROCESSED / "split_ghost_evaluation.parquet")
covered = ghost_eval["parent_asin"].nunique()
check("cold-start items evaluable", covered == len(ghost_ids),
      f"{covered:,} of {len(ghost_ids):,}")

train_users = set(train["user_id"])
orphans = len(set(ghost_eval["user_id"]) - train_users)
check("cold-start users present in training", orphans == 0, f"{orphans} orphans")

print("\nCONTENT FLOOR (required before feature extraction)")
items = pd.read_parquet(PROCESSED / "corpus_items_final.parquet")
no_image = set(items.loc[~items["has_image"], "parent_asin"])
no_text = set(items.loc[~items["has_text"], "parent_asin"])
check("items lacking an image", True, f"{len(no_image)} (masked at extraction)")
check("items lacking text", len(no_text) == 0, f"{len(no_text)}")
check("items lacking BOTH modalities", len(no_image & no_text) == 0,
      f"{len(no_image & no_text)}")

ghost_double_blank = ghost_ids & no_image & no_text
check("cold-start items lacking both", len(ghost_double_blank) == 0,
      f"{len(ghost_double_blank)}")

print("\n" + "=" * 62)
if failures:
    print(f"VERIFICATION FAILED — {len(failures)} check(s) did not pass:")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)
print("VERIFICATION PASSED — the Phase 1 dataset is intact and unmodified.")
print("=" * 62)