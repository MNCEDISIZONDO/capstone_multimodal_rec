"""Pilot: does decorrelating the image embeddings improve their transfer?

Image embeddings occupy a narrow similarity band within each domain, with a
standard deviation around 0.08 against 0.09 to 0.16 for text. A projection
trained on that space learns to amplify small differences, and the measured
consequence is that models relying on images retain the least performance when
moved to unseen items.

Whitening rescales the embedding space so that its dimensions are uncorrelated
and of equal variance. The transform is fitted on training items only and
applied as a fixed operation, so unseen items are transformed by the same rule
rather than by anything learned from them.

Retention is the metric of interest: content-only performance on unseen items as
a share of content-only performance on seen items. It isolates transfer from
pathway quality, since both are measured through the same pathway.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import load_config
from device_utils import clear_gpu_memory, get_device
from models import build_model
from train import NegativeSampler, load_everything

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
REPORTS = Path(cfg["paths"]["reports"])

SEED = cfg["seeds"]["development"]
EPOCHS = 20
K = cfg["evaluation"]["k"]
CHUNK = 64


def whiten(features: np.ndarray, fit_rows: np.ndarray,
           epsilon: float = 1e-5) -> np.ndarray:
    """Decorrelate and equalise variance, fitted on the given rows only.

    Fitting on training items alone keeps the transform independent of the
    withheld items, so applying it to them is a projection rather than a leak.
    """
    fitted = features[fit_rows]
    mean = fitted.mean(axis=0, keepdims=True)
    centred = fitted - mean

    covariance = np.cov(centred, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scaling = eigenvectors / np.sqrt(eigenvalues + epsilon)

    transformed = (features - mean) @ scaling
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    return (transformed / np.clip(norms, 1e-9, None)).astype(np.float32)


def content_only_ndcg(model, device, eval_users, eval_items,
                      candidates, seen) -> float:
    model.eval()
    candidate_tensor = torch.as_tensor(candidates, device=device)
    position_of = {int(item): index for index, item in enumerate(candidates)}
    n_candidates = len(candidates)
    absent = torch.zeros(n_candidates, dtype=torch.bool, device=device)

    total, counted = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(eval_users), CHUNK):
            chunk_users = eval_users[start:start + CHUNK]
            chunk_items = eval_items[start:start + CHUNK]
            n_chunk = len(chunk_users)

            positions = np.array([position_of.get(int(i), -1) for i in chunk_items])
            valid = positions >= 0
            if not valid.any():
                continue

            user_tensor = torch.as_tensor(chunk_users, device=device)
            scores = model(user_tensor.repeat_interleave(n_candidates),
                           candidate_tensor.repeat(n_chunk),
                           collaborative_available=absent.repeat(n_chunk)
                           ).view(n_chunk, n_candidates)

            for row, user in enumerate(chunk_users):
                already = seen.get(int(user))
                if already is not None and len(already):
                    scores[row, torch.as_tensor(already, device=device)] = float("-inf")

            target_positions = torch.as_tensor(np.where(valid, positions, 0),
                                               device=device)
            target = scores.gather(1, target_positions.unsqueeze(1))
            greater = (scores > target).sum(dim=1).float()
            tied = (scores == target).sum(dim=1).float()
            ranks = (greater + (tied + 1.0) / 2.0).cpu().numpy()[valid]
            total += float(np.where(ranks <= K, 1.0 / np.log2(ranks + 1.0), 0.0).sum())
            counted += int(valid.sum())

    return total / counted if counted else 0.0


def run(system: str, image_features: np.ndarray, label: str) -> dict:
    data = load_everything()
    device = get_device()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    model = build_model(cfg, data["n_users"], data["n_items"],
                        image_features, data["text_features"],
                        data["image_available"], data["text_available"],
                        system).to(device)

    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=cfg["training"]["learning_rate"],
                                 weight_decay=cfg["training"]["weight_decay"])

    sampler = NegativeSampler(data["n_items"], data["is_ghost"],
                              data["train_users"], data["train_items"], rng)

    seen_candidates = np.flatnonzero(~data["is_ghost"]).astype(np.int64)
    unseen_candidates = np.flatnonzero(data["is_ghost"]).astype(np.int64)
    position_of = {int(i): p for p, i in enumerate(seen_candidates)}

    seen_map: dict[int, np.ndarray] = {}
    frame = pd.DataFrame({"user": data["train_users"], "item": data["train_items"]})
    for user, group in frame.groupby("user"):
        positions = [position_of[int(i)] for i in group["item"] if int(i) in position_of]
        if positions:
            seen_map[int(user)] = np.array(positions, dtype=np.int64)

    cold = pd.read_parquet(PROCESSED / "split_ghost_evaluation.parquet")
    items = pd.read_parquet(PROCESSED / "item_index.parquet").sort_values("item_idx")
    users = pd.read_parquet(PROCESSED / "user_index.parquet")
    cold_users = cold["user_id"].map(dict(zip(users["user_id"], users["user_idx"])))\
                                .to_numpy(dtype=np.int64)
    cold_items = cold["parent_asin"].map(dict(zip(items["parent_asin"], items["item_idx"])))\
                                    .to_numpy(dtype=np.int64)

    sample = rng.choice(len(data["val_users"]), size=800, replace=False)
    val_users, val_items = data["val_users"][sample], data["val_items"][sample]
    cold_sample = rng.choice(len(cold_users), size=4000, replace=False)
    cold_users, cold_items = cold_users[cold_sample], cold_items[cold_sample]

    batch_size = cfg["training"]["batch_size"]
    n_negatives = cfg["training"]["num_negatives"]
    n_train = len(data["train_users"])

    best = {"seen": 0.0, "unseen": 0.0, "retention": 0.0, "epoch": 0}

    for epoch in range(1, EPOCHS + 1):
        started = time.time()
        model.train()
        order = rng.permutation(n_train)

        for start in range(0, n_train, batch_size):
            index = order[start:start + batch_size]
            users_t = torch.as_tensor(data["train_users"][index], device=device)
            positives = torch.as_tensor(data["train_items"][index], device=device)
            negatives = torch.as_tensor(
                sampler.sample(data["train_users"][index], n_negatives), device=device)

            positive = model(users_t, positives)
            negative = model(users_t.repeat_interleave(n_negatives), negatives)
            loss = -F.logsigmoid(
                positive.repeat_interleave(n_negatives) - negative).mean()

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

        seen_score = content_only_ndcg(model, device, val_users, val_items,
                                       seen_candidates, seen_map)
        unseen_score = content_only_ndcg(model, device, cold_users, cold_items,
                                         unseen_candidates, {})
        retention = unseen_score / seen_score if seen_score else 0.0

        if unseen_score > best["unseen"]:
            best = {"seen": seen_score, "unseen": unseen_score,
                    "retention": retention, "epoch": epoch}

        print(f"    epoch {epoch:>3}  seen {seen_score:.4f}  "
              f"unseen {unseen_score:.4f}  retention {retention:>6.1%}  "
              f"{time.time() - started:.0f}s", flush=True)

    clear_gpu_memory()
    return {"system": system, "variant": label, **best}


def main() -> None:
    with h5py.File(Path(cfg["paths"]["features"]) / "image_embeddings.h5", "r") as store:
        original = store["embeddings"][:]

    items = pd.read_parquet(PROCESSED / "item_index.parquet").sort_values("item_idx")
    is_ghost = items["is_ghost"].to_numpy().copy()
    train_rows = np.flatnonzero(~is_ghost)

    whitened = whiten(original, train_rows)

    print("EMBEDDING SPREAD BEFORE AND AFTER WHITENING")
    rng = np.random.default_rng(SEED)
    sample = rng.choice(train_rows, size=1500, replace=False)
    for label, matrix in (("original", original), ("whitened", whitened)):
        vectors = torch.as_tensor(matrix[sample])
        vectors = vectors / vectors.norm(dim=1, keepdim=True).clamp_min(1e-9)
        similarity = (vectors @ vectors.T).numpy()
        upper = similarity[np.triu_indices(len(sample), k=1)]
        print(f"  {label:<10}mean {upper.mean():.4f}   s.d. {upper.std():.4f}")

    results = []
    for system in ("image_only", "concat_fusion"):
        for label, matrix in (("original", original), ("whitened", whitened)):
            print(f"\n{system} — {label} image features")
            results.append(run(system, matrix, label))

    frame = pd.DataFrame(results)
    print("\nBEST RETENTION BY VARIANT")
    print(f"  {'system':<18}{'variant':<12}{'seen':>9}{'unseen':>9}{'retention':>12}")
    for row in frame.itertuples():
        print(f"  {row.system:<18}{row.variant:<12}{row.seen:>9.4f}"
              f"{row.unseen:>9.4f}{row.retention:>11.1%}")

    output = REPORTS / "pilot_whitening.csv"
    frame.to_csv(output, index=False)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()