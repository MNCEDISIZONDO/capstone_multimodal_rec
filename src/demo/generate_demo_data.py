"""Precompute storefront data: catalogue, per-customer scores from both models.

"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import shutil
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image

from config import load_config
from device_utils import get_device
from models import build_model

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
FEATURES = Path(cfg["paths"]["features"])
CHECKPOINTS = Path(cfg["paths"]["checkpoints"])
IMAGES = Path(cfg["paths"]["images"])
DEMO = Path(cfg["paths"]["demo"])
STATIC = Path(cfg["project"]["root"]) / "static" / "img"
DEMO.mkdir(parents=True, exist_ok=True)
STATIC.mkdir(parents=True, exist_ok=True)

SEED = cfg["seeds"]["statistical"][0]
MULTIMODAL = "concat_fusion"
COLLABORATIVE = "ncf"
N_USERS = 6
N_ESTABLISHED = 1400
N_NEW = 600
THUMB = 300


def load_model(system, n_users, n_items, features, device):
    path = CHECKPOINTS / f"{system}_seed{SEED}.pt"
    if not path.exists():
        raise SystemExit(f"missing checkpoint: {path}")
    model = build_model(cfg, n_users, n_items, features["image"], features["text"],
                        features["image_available"], features["text_available"],
                        system).to(device)
    model.load_state_dict(torch.load(path, map_location=device)["state_dict"])
    model.eval()
    return model


def score_all(model, device, user_idx, n_items, is_ghost, chunk=2048):
    """Score every product; the collaborative signal is masked for new listings."""
    available = torch.as_tensor(~is_ghost, device=device)
    out = np.empty(n_items, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_items, chunk):
            idx = torch.arange(start, min(start + chunk, n_items), device=device)
            users = torch.full((len(idx),), user_idx, dtype=torch.long, device=device)
            out[start:start + len(idx)] = model(
                users, idx, collaborative_available=available[idx]).cpu().numpy()
    return out


def copy_thumb(asin: str) -> str | None:
    source = IMAGES / f"{asin}.jpg"
    if not source.exists():
        return None
    target = STATIC / f"{asin}.jpg"
    if not target.exists():
        try:
            with Image.open(source) as image:
                image = image.convert("RGB")
                image.thumbnail((THUMB, THUMB))
                image.save(target, "JPEG", quality=82)
        except Exception:
            return None
    return f"app/static/img/{asin}.jpg"


def main() -> None:
    device = get_device()
    rng = np.random.default_rng(SEED)

    items = pd.read_parquet(PROCESSED / "item_index.parquet").sort_values("item_idx")
    users = pd.read_parquet(PROCESSED / "user_index.parquet")
    full = pd.read_parquet(PROCESSED / "items_all.parquet",
                           columns=["parent_asin", "title", "description",
                                    "features", "average_rating", "rating_number"])
    corpus = pd.read_parquet(PROCESSED / "corpus_items_final.parquet",
                             columns=["parent_asin", "n_interactions"])
    items = items.merge(full, on="parent_asin", how="left") \
                 .merge(corpus, on="parent_asin", how="left")

    train = pd.read_parquet(PROCESSED / "split_train.parquet")
    user_of = dict(zip(users["user_id"], users["user_idx"]))
    item_of = dict(zip(items["parent_asin"], items["item_idx"]))

    with h5py.File(FEATURES / "image_embeddings.h5", "r") as store:
        image_features, image_available = store["embeddings"][:], store["available"][:]
    with h5py.File(FEATURES / "text_embeddings.h5", "r") as store:
        text_features, text_available = store["embeddings"][:], store["available"][:]
    feats = {"image": image_features, "text": text_features,
             "image_available": image_available, "text_available": text_available}

    is_ghost = items.sort_values("item_idx")["is_ghost"].to_numpy().copy()
    n_items = len(items)

    multimodal = load_model(MULTIMODAL, len(users), n_items, feats, device)
    collaborative = load_model(COLLABORATIVE, len(users), n_items, feats, device)

    domain_of = dict(zip(items["parent_asin"], items["domain"]))
    labelled = train.assign(domain=train["parent_asin"].map(domain_of))
    profiles = []
    for user_id, group in labelled.groupby("user_id"):
        if not 6 <= len(group) <= 14:
            continue
        counts = group["domain"].value_counts()
        profiles.append({"user_id": user_id, "n": len(group),
                         "dominant": counts.index[0],
                         "share": counts.iloc[0] / len(group)})
    frame = pd.DataFrame(profiles).sort_values(["share", "n"], ascending=False)

    selected = []
    for domain in cfg["data"]["domains"]:
        pool = frame[(frame["dominant"] == domain) & (frame["share"] >= 0.5)]
        if len(pool):
            selected.append(pool.iloc[0].to_dict())
    picked = {c["user_id"] for c in selected}
    for row in frame.itertuples():
        if len(selected) >= N_USERS:
            break
        if row.user_id not in picked:
            selected.append({"user_id": row.user_id, "n": row.n,
                             "dominant": row.dominant, "share": row.share})
    selected = selected[:N_USERS]

    # demonstration catalogue: a balanced sample plus every customer's history
    established = np.flatnonzero(~is_ghost)
    new_items = np.flatnonzero(is_ghost)
    chosen = set(rng.choice(established, N_ESTABLISHED, replace=False).tolist())
    chosen.update(rng.choice(new_items, N_NEW, replace=False).tolist())
    for profile in selected:
        chosen.update(item_of[a] for a in
                      train.loc[train["user_id"] == profile["user_id"], "parent_asin"]
                      if a in item_of)
    listing = sorted(chosen)
    print(f"demonstration catalogue: {len(listing):,} of {n_items:,} products")

    customers = []
    for profile in selected:
        user_id = profile["user_id"]
        user_idx = user_of[user_id]
        history = [item_of[a] for a in
                   train.loc[train["user_id"] == user_id, "parent_asin"] if a in item_of]

        mm = score_all(multimodal, device, user_idx, n_items, is_ghost)
        cf = score_all(collaborative, device, user_idx, n_items, is_ghost)

        # where does each model place its best new listing across the whole store
        # rank position of every product under each model, so the interface can
        # compare models on a common scale rather than on raw scores
        mm_positions = np.argsort(np.argsort(-mm)) + 1
        cf_positions = np.argsort(np.argsort(-cf)) + 1
        mm_rank = int(mm_positions[new_items].min())
        cf_rank = int(cf_positions[new_items].min())

        customers.append({
            "user_id": user_id,
            "label": profile["dominant"].replace("_", " ").title() + " Shopper",
            "dominant_domain": profile["dominant"].replace("_", " "),
            "domain_share": round(float(profile["share"]), 2),
            "history": [int(i) for i in history],
            "scores_multimodal": {str(i): round(float(mm[i]), 4) for i in listing},
            "scores_collaborative": {str(i): round(float(cf[i]), 4) for i in listing},
            "rank_multimodal": {str(i): int(mm_positions[i]) for i in listing},
            "rank_collaborative": {str(i): int(cf_positions[i]) for i in listing},
            "best_new_rank_multimodal": mm_rank,
            "best_new_rank_collaborative": cf_rank,
            "spread_multimodal": round(float(mm[new_items].std()), 4),
            "spread_collaborative": round(float(cf[new_items].std()), 6),
            "distinct_collaborative": int(len(np.unique(np.round(cf[new_items], 4)))),
        })
        print(f"  {profile['dominant']:<22}best new listing at rank "
              f"{mm_rank:>5,} (multimodal) vs {cf_rank:>5,} (collaborative)   "
              f"distinct cf scores {len(np.unique(np.round(cf[new_items], 4)))}")

    # products similar to each catalogue item, by combined image and text content
    print("\ncomputing similar products")
    content = np.concatenate([image_features, text_features], axis=1)
    content = content / np.clip(np.linalg.norm(content, axis=1, keepdims=True),
                                1e-9, None)
    subset = content[listing]
    position_of = {idx: n for n, idx in enumerate(listing)}
    similar = {}
    for n, idx in enumerate(listing):
        scores = subset @ subset[n]
        scores[n] = -np.inf
        similar[idx] = [int(listing[j]) for j in np.argsort(-scores)[:8]]

    print("copying images")
    detail = items.set_index("item_idx")



    
    catalogue = {}
    for idx in listing:
        row = detail.loc[idx]
        new = bool(is_ghost[idx])
        text = " ".join(str(row["features"] or "").split()) or \
               " ".join(str(row["description"] or "").split())
        catalogue[str(idx)] = {
            "title": (row["title"] or "(untitled)")[:150],
            "domain": row["domain"].replace("_", " "),
            "image": copy_thumb(row["parent_asin"]),
            "description": text[:500],
            "is_new": new,
            "rating": None if new or pd.isna(row["average_rating"])
                      else round(float(row["average_rating"]), 1),
            "reviews": 0 if new or pd.isna(row["rating_number"])
                       else int(row["rating_number"]),
            "purchases": 0 if new or pd.isna(row["n_interactions"])
                         else int(row["n_interactions"]),
            "similar": similar[idx],
        }

    payload = {"generated": datetime.now().isoformat(timespec="seconds"),
               "multimodal_model": MULTIMODAL,
               "collaborative_model": COLLABORATIVE,
               "seed": SEED,
               "n_total": int(n_items),
               "n_established": int(len(established)),
               "n_new": int(len(new_items)),
               "catalogue": catalogue,
               "customers": customers}

    output = DEMO / "demo_data.json"
    output.write_text(json.dumps(payload))
    print(f"wrote {output}  ({output.stat().st_size / 1024**2:.1f} MB)")
    print(f"images in {STATIC}")


if __name__ == "__main__":
    main()