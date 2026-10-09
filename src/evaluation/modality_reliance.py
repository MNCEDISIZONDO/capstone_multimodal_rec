"""Measure which signals a trained fusion model actually relies on
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

from config import load_config
from device_utils import get_device
from models import build_model

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
FEATURES = Path(cfg["paths"]["features"])
CHECKPOINTS = Path(cfg["paths"]["checkpoints"])
REPORTS = Path(cfg["paths"]["reports"])

SEED = cfg["seeds"]["development"]
N_PAIRS = 4000


def main() -> None:
    device = get_device()
    rng = np.random.default_rng(SEED)

    items = pd.read_parquet(PROCESSED / "item_index.parquet").sort_values("item_idx")
    users = pd.read_parquet(PROCESSED / "user_index.parquet")

    with h5py.File(FEATURES / "image_embeddings.h5", "r") as store:
        image_features, image_available = store["embeddings"][:], store["available"][:]
    with h5py.File(FEATURES / "text_embeddings.h5", "r") as store:
        text_features, text_available = store["embeddings"][:], store["available"][:]

    is_ghost = items["is_ghost"].to_numpy().copy()
    warm_items = np.flatnonzero(~is_ghost)
    cold_items = np.flatnonzero(is_ghost)

    checkpoint = CHECKPOINTS / f"attention_fusion_seed{SEED}.pt"
    if not checkpoint.exists():
        raise SystemExit(f"no checkpoint at {checkpoint}")

    model = build_model(cfg, len(users), len(items), image_features, text_features,
                        image_available, text_available, "attention_fusion").to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    names = model.signal_names()
    print(f"checkpoint  : epoch {state['epoch']}, "
          f"validation NDCG {state['val_ndcg']:.4f}")
    print(f"signals     : {names}\n")

    def weights_for(item_pool: np.ndarray, collaborative: bool) -> np.ndarray:
        user_idx = torch.as_tensor(rng.choice(len(users), N_PAIRS), device=device)
        item_idx = torch.as_tensor(rng.choice(item_pool, N_PAIRS), device=device)
        available = None if collaborative else torch.zeros(N_PAIRS, dtype=torch.bool,
                                                           device=device)
        with torch.no_grad():
            return model.attention_weights(user_idx, item_idx,
                                           collaborative_available=available).cpu().numpy()

    warm = weights_for(warm_items, collaborative=True)
    cold = weights_for(cold_items, collaborative=False)

    print("ATTENTION WEIGHT PER SIGNAL")
    print(f"  {'condition':<18}" + "".join(f"{n:>18}" for n in names))
    print(f"  {'warm items':<18}" +
          "".join(f"{warm[:, i].mean():>12.4f} ±{warm[:, i].std():.3f}"
                  for i in range(len(names))))
    print(f"  {'cold-start items':<18}" +
          "".join(f"{cold[:, i].mean():>12.4f} ±{cold[:, i].std():.3f}"
                  for i in range(len(names))))

    print("\nINTERPRETATION")
    collaborative_share = warm[:, names.index("collaborative")].mean()
    image_share = warm[:, names.index("image")].mean()
    text_share = warm[:, names.index("text")].mean()
    dominant = max(zip([collaborative_share, image_share, text_share], names))
    print(f"  dominant signal on warm items : {dominant[1]} ({dominant[0]:.1%})")
    if dominant[0] > 0.6:
        print("  One pathway dominates. A signal that reduces training loss quickly")
        print("  can suppress the development of signals that generalise better,")
        print("  which would explain a fusion model performing below one of its")
        print("  own components.")
    else:
        print("  No single pathway dominates; reliance is comparatively balanced.")

    print("\nEMBEDDING TABLE UTILISATION")
    item_embedding = model.item_embedding.weight.detach().cpu().numpy()
    warm_norms = np.linalg.norm(item_embedding[warm_items], axis=1)
    cold_norms = np.linalg.norm(item_embedding[cold_items], axis=1)
    print(f"  trained item vectors   mean norm {warm_norms.mean():.4f}")
    print(f"  untrained (cold-start) mean norm {cold_norms.mean():.4f}")
    print(f"  ratio {warm_norms.mean() / max(cold_norms.mean(), 1e-9):.1f}x")
    print("  A ratio near one indicates the collaborative table barely moved from")
    print("  its initialisation, which is expected when interactions per item are")
    print("  too few to identify an embedding of this width.")

    rows = []
    for index, name in enumerate(names):
        rows.append({"condition": "warm", "signal": name,
                     "mean_weight": round(float(warm[:, index].mean()), 5),
                     "std_weight": round(float(warm[:, index].std()), 5)})
        rows.append({"condition": "cold_start", "signal": name,
                     "mean_weight": round(float(cold[:, index].mean()), 5),
                     "std_weight": round(float(cold[:, index].std()), 5)})
    output = REPORTS / "modality_reliance.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()