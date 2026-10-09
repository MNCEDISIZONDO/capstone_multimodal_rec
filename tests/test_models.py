"""Verify the model architectures before training 
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import h5py
import numpy as np
import pandas as pd
import torch

from config import load_config
from device_utils import get_device, vram_report
from models import SYSTEMS, PopularityBaseline, RandomBaseline, build_model

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
FEATURES = Path(cfg["paths"]["features"])

failures: list[str] = []


def check(label: str, passed: bool, detail: str = "") -> None:
    print(f"  {'PASS' if passed else 'FAIL'}  {label:<46}{detail}")
    if not passed:
        failures.append(label)


def load_features(name: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(FEATURES / name, "r") as store:
        return store["embeddings"][:], store["available"][:]


def main() -> None:
    items = pd.read_parquet(PROCESSED / "item_index.parquet").sort_values("item_idx")
    users = pd.read_parquet(PROCESSED / "user_index.parquet")
    n_items, n_users = len(items), len(users)
    #
    is_ghost = torch.as_tensor(items["is_ghost"].to_numpy().copy())

    image_features, image_available = load_features("image_embeddings.h5")
    text_features, text_available = load_features("text_embeddings.h5")

    print("INPUTS")
    print(f"        users {n_users:,}   items {n_items:,}   "
          f"cold-start {int(is_ghost.sum()):,}")
    print(f"        image {image_features.shape}   text {text_features.shape}")

    device = get_device()
    print(f"        device {device}\n")

    print("SYSTEM CONSTRUCTION AND OUTPUT SHAPE")
    models = {}
    for system in SYSTEMS:
        model = build_model(cfg, n_users, n_items, image_features, text_features,
                            image_available, text_available, system).to(device)
        model.eval()
        models[system] = model

        user_idx = torch.arange(16, device=device)
        warm = torch.as_tensor(items.loc[~items["is_ghost"], "item_idx"].to_numpy()[:16],
                               device=device)
        with torch.no_grad():
            scores = model(user_idx, warm)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        check(f"{system}", tuple(scores.shape) == (16,) and torch.isfinite(scores).all(),
              f"{trainable:>9,} parameters   signals {model.signal_names()}")

    print("\nCOLD-START BRANCHING")
    cold = torch.as_tensor(items.loc[items["is_ghost"], "item_idx"].to_numpy()[:32],
                           device=device)
    user_idx = torch.arange(32, device=device)

    for system in ("concat_fusion", "attention_fusion"):
        model = models[system]
        absent = torch.zeros(32, dtype=torch.bool, device=device)
        present = torch.ones(32, dtype=torch.bool, device=device)

        with torch.no_grad():
            content_only = model(user_idx, cold, collaborative_available=absent)
            with_collaborative = model(user_idx, cold, collaborative_available=present)

            saved = model.item_embedding.weight.data.clone()
            model.item_embedding.weight.data[cold] = 0.0
            zero_substituted = model(user_idx, cold, collaborative_available=present)
            model.item_embedding.weight.data = saved

        differs_from_warm_path = (content_only - with_collaborative).abs().max().item()
        differs_from_zeros = (content_only - zero_substituted).abs().max().item()

        check(f"{system}: cold-start path is distinct",
              differs_from_warm_path > 1e-6, f"max difference {differs_from_warm_path:.6f}")
        check(f"{system}: not substituting zeros",
              differs_from_zeros > 1e-6, f"max difference {differs_from_zeros:.6f}")

    print("\nATTENTION BEHAVIOUR")
    model = models["attention_fusion"]
    with torch.no_grad():
        warm_weights = model.attention_weights(user_idx, cold)
        cold_weights = model.attention_weights(
            user_idx, cold, collaborative_available=torch.zeros(32, dtype=torch.bool,
                                                               device=device))
    names = model.signal_names()
    print(f"        signals {names}")
    print(f"        with collaborative    {np.round(warm_weights[0].cpu().numpy(), 4)}")
    print(f"        without collaborative {np.round(cold_weights[0].cpu().numpy(), 4)}")
    check("collaborative receives no attention when absent",
          bool((cold_weights[:, 0] == 0).all()))
    check("attention weights sum to one",
          bool(torch.allclose(cold_weights.sum(dim=1),
                              torch.ones(32, device=device), atol=1e-4)))

    print("\nPLACEHOLDER ITEMS ARE MASKED, NOT CONSUMED")
    no_image = np.where(~image_available)[0]
    if len(no_image):
        sample = torch.as_tensor(no_image[:min(16, len(no_image))], device=device)
        with torch.no_grad():
            weights = model.attention_weights(torch.arange(len(sample), device=device), sample)
        check("items without an image receive no image attention",
              bool((weights[:, names.index("image")] == 0).all()),
              f"{len(no_image)} such items")

    print("\nMODALITY SUPPRESSION FOR EVALUATION CONDITIONS")
    warm = torch.as_tensor(items.loc[~items["is_ghost"], "item_idx"].to_numpy()[:32],
                           device=device)
    with torch.no_grad():
        control = model(user_idx, warm)
        blind = model(user_idx, warm, image_enabled=False)
        silent = model(user_idx, warm, text_enabled=False)
    check("suppressing the image changes scores",
          (control - blind).abs().max().item() > 1e-6)
    check("suppressing the text changes scores",
          (control - silent).abs().max().item() > 1e-6)

    print("\nGRADIENT FLOW")
    model = build_model(cfg, n_users, n_items, image_features, text_features,
                        image_available, text_available, "attention_fusion").to(device)
    model.train()
    user_idx = torch.randint(0, n_users, (256,), device=device)
    positive = torch.as_tensor(np.random.choice(
        items.loc[~items["is_ghost"], "item_idx"].to_numpy(), 256), device=device)
    negative = torch.as_tensor(np.random.choice(
        items.loc[~items["is_ghost"], "item_idx"].to_numpy(), 256), device=device)
    loss = -torch.nn.functional.logsigmoid(
        model(user_idx, positive) - model(user_idx, negative)).mean()
    loss.backward()

    # The final bias is excluded: under a pairwise ranking objective a constant
    # added to both scores cancels in their difference, so the parameter is
    # unidentifiable and receives exactly zero gradient by construction.
    without_gradient = [name for name, p in model.named_parameters()
                        if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)
                        and not name.endswith("head.9.bias")]
    check("every identifiable parameter receives gradient",
          not without_gradient, f"loss {loss.item():.4f}")
    if without_gradient:
        for name in without_gradient:
            print(f"        no gradient: {name}")
    check("precomputed features are not trainable",
          not model.image_features.requires_grad and not model.text_features.requires_grad)

    print("\nBASELINE ANCHORS")
    train = pd.read_parquet(PROCESSED / "split_train.parquet")
    mapping = dict(zip(items["parent_asin"], items["item_idx"]))
    train_indices = train["parent_asin"].map(mapping).to_numpy()
    popularity = PopularityBaseline(n_items, train_indices)
    cold_indices = items.loc[items["is_ghost"], "item_idx"].to_numpy()
    check("popularity cannot rank a withheld item",
          bool((popularity.score(cold_indices) == 0).all()),
          f"{len(cold_indices):,} cold-start items scored zero")
    random_baseline = RandomBaseline(cfg["seeds"]["split_construction"])
    check("random baseline produces scores",
          len(random_baseline.score(np.arange(10))) == 10)

    print("\nMEMORY")
    if device.type == "cuda":
        print(f"        {vram_report()}")

    print("\n" + "=" * 66)
    if failures:
        print(f"MODEL VERIFICATION FAILED — {len(failures)} check(s) did not pass:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("MODEL VERIFICATION PASSED — architectures are correct and cold-start")
    print("branching is confirmed to be genuine.")
    print("=" * 66)


if __name__ == "__main__":
    main()