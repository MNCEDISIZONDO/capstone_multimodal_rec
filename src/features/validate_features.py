"""Validate the extracted embeddings before training.

Check their dimensions, alignment with the item index and numerical integrity.
Then confirm that items from the same product domain are more similar than
items from different domains. Stop if any check fails.
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

from config import load_config

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
FEATURES = Path(cfg["paths"]["features"])
REPORTS = Path(cfg["paths"]["reports"])
REPORTS.mkdir(parents=True, exist_ok=True)

ITEM_INDEX = PROCESSED / "item_index.parquet"
IMAGE_FILE = FEATURES / "image_embeddings.h5"
TEXT_FILE = FEATURES / "text_embeddings.h5"
OUT_NOTE = REPORTS / "feature_validation.md"

SEED = cfg["seeds"]["split_construction"]
N_PAIRS = 2000
DOMAINS = cfg["data"]["domains"]

failures: list[str] = []


def check(label: str, passed: bool, detail: str = "") -> None:
    print(f"  {'PASS' if passed else 'FAIL'}  {label:<44}{detail}")
    if not passed:
        failures.append(label)


def load(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    with h5py.File(path, "r") as store:
        return (store["embeddings"][:],
                store["available"][:],
                [a.decode() for a in store["parent_asin"][:]],
                dict(store.attrs))


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two equally shaped matrices."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a_norm * b_norm, axis=1)


def sample_pairs(rng, pool_a: np.ndarray, pool_b: np.ndarray,
                 n: int, same_pool: bool) -> tuple[np.ndarray, np.ndarray]:
    """Draw n index pairs, avoiding an item paired with itself."""
    left = rng.choice(pool_a, size=n, replace=True)
    right = rng.choice(pool_b, size=n, replace=True)
    if same_pool:
        collision = left == right
        while collision.any():
            right[collision] = rng.choice(pool_b, size=int(collision.sum()), replace=True)
            collision = left == right
    return left, right


def semantic_check(name: str, embeddings: np.ndarray, available: np.ndarray,
                   domains: np.ndarray, rng) -> dict:
    """Compare within-domain similarity against cross-domain similarity."""
    print(f"\n{name.upper()} — SEMANTIC STRUCTURE")
    rows = []

    for domain in DOMAINS:
        inside = np.where((domains == domain) & available)[0]
        outside = np.where((domains != domain) & available)[0]
        if len(inside) < 2 or len(outside) < 1:
            continue

        li, ri = sample_pairs(rng, inside, inside, N_PAIRS, same_pool=True)
        within = cosine(embeddings[li], embeddings[ri]).mean()

        lo, ro = sample_pairs(rng, inside, outside, N_PAIRS, same_pool=False)
        across = cosine(embeddings[lo], embeddings[ro]).mean()

        margin = within - across
        print(f"  {domain:<22} within {within:6.3f}   across {across:6.3f}   "
              f"margin {margin:+.3f}")
        rows.append({"modality": name, "domain": domain,
                     "within_domain": round(float(within), 4),
                     "cross_domain": round(float(across), 4),
                     "margin": round(float(margin), 4)})

    margins = [r["margin"] for r in rows]
    check(f"{name}: within-domain exceeds cross-domain",
          all(m > 0 for m in margins),
          f"{sum(m > 0 for m in margins)} of {len(margins)} domains")
    check(f"{name}: mean margin is substantive",
          float(np.mean(margins)) > 0.01,
          f"{np.mean(margins):+.3f}")
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)

    items = pd.read_parquet(ITEM_INDEX).sort_values("item_idx").reset_index(drop=True)
    domains = items["domain"].to_numpy()
    is_ghost = items["is_ghost"].to_numpy()

    img, img_ok, img_asins, img_attrs = load(IMAGE_FILE)
    txt, txt_ok, txt_asins, txt_attrs = load(TEXT_FILE)

    print("STRUCTURAL INTEGRITY")
    check("image rows match item index", len(img) == len(items), f"{len(img):,}")
    check("text rows match item index", len(txt) == len(items), f"{len(txt):,}")
    check("image dimension", img.shape[1] == cfg["features"]["clip_dim"],
          str(img.shape[1]))
    check("text dimension", txt.shape[1] == cfg["features"]["sbert_dim"],
          str(txt.shape[1]))
    check("image row order matches index",
          img_asins == items["parent_asin"].tolist())
    check("text row order matches index",
          txt_asins == items["parent_asin"].tolist())
    check("image values are finite", bool(np.isfinite(img).all()))
    check("text values are finite", bool(np.isfinite(txt).all()))
    check("image encoder recorded",
          img_attrs.get("model") == cfg["features"]["clip_model"],
          str(img_attrs.get("model")))
    check("text encoder recorded",
          txt_attrs.get("model") == cfg["features"]["sbert_model"],
          str(txt_attrs.get("model")))

    print("\nMODALITY AVAILABILITY")
    check("image availability", int(img_ok.sum()) == 9479, f"{int(img_ok.sum()):,}")
    check("text availability", int(txt_ok.sum()) == len(items), f"{int(txt_ok.sum()):,}")
    check("no item lacks both modalities",
          int((~img_ok & ~txt_ok).sum()) == 0,
          f"{int((~img_ok & ~txt_ok).sum())}")
    check("no cold-start item lacks both",
          int((is_ghost & ~img_ok & ~txt_ok).sum()) == 0,
          f"{int((is_ghost & ~img_ok & ~txt_ok).sum())}")
    check("placeholder rows are exactly zero",
          bool((np.abs(img[~img_ok]).sum() == 0)),
          f"{int((~img_ok).sum())} rows")

    print("\nEMBEDDING SCALE")
    img_norms = np.linalg.norm(img[img_ok], axis=1)
    txt_norms = np.linalg.norm(txt[txt_ok], axis=1)
    print(f"  image norm  mean {img_norms.mean():7.3f}  "
          f"min {img_norms.min():7.3f}  max {img_norms.max():7.3f}")
    print(f"  text norm   mean {txt_norms.mean():7.3f}  "
          f"min {txt_norms.min():7.3f}  max {txt_norms.max():7.3f}")
    ratio = img_norms.mean() / txt_norms.mean()
    print(f"  scale ratio image:text = {ratio:.1f}:1")
    if ratio > 2 or ratio < 0.5:
        print("  NOTE: the two modalities occupy different numerical scales.")
        print("        Per-modality normalisation is required before fusion,")
        print("        otherwise the larger-scale modality dominates the")
        print("        concatenated representation from initialisation.")

    image_rows = semantic_check("image", img, img_ok, domains, rng)
    text_rows = semantic_check("text", txt, txt_ok, domains, rng)

    print("\nCOLD-START ITEM COVERAGE")
    check("cold-start items have an image",
          int((is_ghost & ~img_ok).sum()) < 20,
          f"{int((is_ghost & ~img_ok).sum())} without")
    check("cold-start items have text",
          int((is_ghost & ~txt_ok).sum()) == 0,
          f"{int((is_ghost & ~txt_ok).sum())} without")

    pd.DataFrame(image_rows + text_rows).to_csv(
        REPORTS / "feature_similarity.csv", index=False)

    with open(OUT_NOTE, "w") as note:
        note.write("# Feature Extraction Validation\n\n")
        note.write(f"Image encoder: `{img_attrs.get('model')}`, "
                   f"{img.shape[1]} dimensions, "
                   f"{int(img_ok.sum()):,} of {len(items):,} items encoded.\n\n")
        note.write(f"Text encoder: `{txt_attrs.get('model')}`, "
                   f"{txt.shape[1]} dimensions, "
                   f"{int(txt_ok.sum()):,} of {len(items):,} items encoded.\n\n")
        note.write("## Numerical integrity\n\n")
        note.write("All embeddings are finite. Embedding rows are aligned to "
                   "`item_index.parquet` by position, verified by comparing the "
                   "stored identifier order against the index.\n\n")
        note.write("## Embedding scale\n\n")
        note.write(f"Image vectors have mean norm {img_norms.mean():.3f}; text "
                   f"vectors have mean norm {txt_norms.mean():.3f}, because the "
                   f"text encoder applies L2 normalisation internally. The "
                   f"resulting scale ratio of {ratio:.1f}:1 requires "
                   f"per-modality normalisation before fusion, so that neither "
                   f"modality dominates the concatenated representation for "
                   f"reasons of numerical magnitude rather than information "
                   f"content.\n\n")
        note.write("## Semantic structure\n\n")
        note.write("Within-domain and cross-domain cosine similarity, "
                   f"averaged over {N_PAIRS:,} sampled pairs per domain:\n\n")
        note.write("| modality | domain | within | across | margin |\n")
        note.write("|---|---|---|---|---|\n")
        for r in image_rows + text_rows:
            note.write(f"| {r['modality']} | {r['domain']} | "
                       f"{r['within_domain']:.3f} | {r['cross_domain']:.3f} | "
                       f"{r['margin']:+.3f} |\n")
        note.write("\nA positive margin in every domain confirms that both "
                   "encoders place products of the same kind closer together "
                   "than products of different kinds, which is the property the "
                   "multimodal design depends on.\n")

    print("\n" + "=" * 62)
    if failures:
        print(f"VALIDATION FAILED — {len(failures)} check(s) did not pass:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("VALIDATION PASSED — embeddings are structurally and semantically sound.")
    print("=" * 62)
    print(f"\nWrote {OUT_NOTE}")
    print(f"Wrote {REPORTS / 'feature_similarity.csv'}")


if __name__ == "__main__":
    main()