
from __future__ import annotations

# Allow this module to import the shared modules at the top of src/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from config import load_config

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
FEATURES = Path(cfg["paths"]["features"])
RESULTS = Path(cfg["paths"]["results"])
PER_USER = RESULTS / "per_user"
TABLES = RESULTS / "tables"

SYSTEMS = ["ncf", "image_only", "text_only", "content_only",
           "concat_fusion", "attention_fusion"]


def main() -> None:
    items = pd.read_parquet(PROCESSED / "item_index.parquet").sort_values("item_idx")
    users = pd.read_parquet(PROCESSED / "user_index.parquet")
    item_of = dict(zip(items["parent_asin"], items["item_idx"]))
    user_of = dict(zip(users["user_id"], users["user_idx"]))

    def to_indices(name: str) -> tuple[np.ndarray, np.ndarray]:
        frame = pd.read_parquet(PROCESSED / name)
        return (frame["user_id"].map(user_of).to_numpy(dtype=np.int64),
                frame["parent_asin"].map(item_of).to_numpy(dtype=np.int64))

    train_users, train_items = to_indices("split_train.parquet")
    test_users, test_items = to_indices("split_test.parquet")

    with h5py.File(FEATURES / "image_embeddings.h5", "r") as store:
        image_available = store["available"][:]

    no_image = set(np.flatnonzero(~image_available).tolist())
    trained = set(train_items.tolist())
    untrained = {int(i) for i in np.unique(test_items) if int(i) not in trained}

    # A user is affected when their held-out test item belongs to the group.
    affected_no_image = np.array([int(i) in no_image for i in test_items])
    affected_untrained = np.array([int(i) in untrained for i in test_items])

    print("THE TWO ITEM GROUPS")
    print(f"  items without a retrievable image      : {len(no_image):,} "
          f"({len(no_image) / len(items):.2%} of catalogue)")
    print(f"  items in test with no training history : {len(untrained):,} "
          f"({len(untrained) / len(items):.2%} of catalogue)")
    print(f"  overlap between the two groups         : "
          f"{len(no_image & untrained):,}")
    print()
    print("USERS AFFECTED IN THE TEST SPLIT")
    print(f"  held-out item has no image             : {int(affected_no_image.sum()):,} "
          f"of {len(test_items):,} ({affected_no_image.mean():.3%})")
    print(f"  held-out item has no training history  : {int(affected_untrained.sum()):,} "
          f"of {len(test_items):,} ({affected_untrained.mean():.3%})")
    print()

    rows = []
    print("EFFECT ON REPORTED METRICS")
    print(f"  {'system':<18}{'condition':<10}{'reported':>10}{'excluded':>10}"
          f"{'change':>10}")

    for system in SYSTEMS:
        for condition in ("control", "blind", "silent"):
            path = PER_USER / f"{system}_{condition}_full_ndcg.npy"
            users_path = PER_USER / f"{system}_{condition}_full_users.npy"
            if not path.exists():
                continue

            scores = np.load(path)
            if len(scores) != len(test_items):
                print(f"  {system:<18}{condition:<10}length mismatch, skipped")
                continue

            # Image-less items are excluded from Control and Silent, where their
            # empty image feature is a data artifact. Blind suppresses the image
            # for every item, so the group is not distinguishable there.
            mask = np.zeros(len(scores), dtype=bool)
            if condition in ("control", "silent"):
                mask |= affected_no_image
            mask |= affected_untrained

            reported = float(scores.mean())
            excluded = float(scores[~mask].mean())
            change = (excluded - reported) / reported * 100 if reported else 0.0

            print(f"  {system:<18}{condition:<10}{reported:>10.5f}{excluded:>10.5f}"
                  f"{change:>9.2f}%")
            rows.append({
                "system": system, "condition": condition,
                "reported_ndcg": round(reported, 6),
                "ndcg_excluding_affected": round(excluded, 6),
                "change_pct": round(change, 3),
                "users_removed": int(mask.sum()),
            })

    table = pd.DataFrame(rows)
    output = TABLES / "exclusion_impact.csv"
    table.to_csv(output, index=False)

    largest = table["change_pct"].abs().max() if len(table) else 0.0
    print(f"\n  largest absolute change across all systems and conditions: "
          f"{largest:.2f}%")
    if largest < 1.0:
        print("  The affected items do not materially alter any reported figure.")
        print("  Reported results retain them; this measurement is the justification.")
    else:
        print("  The effect exceeds one percent and the affected items should be")
        print("  excluded from the reported figures.")

    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()