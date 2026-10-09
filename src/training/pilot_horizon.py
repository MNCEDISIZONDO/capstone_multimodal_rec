
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import load_config
from device_utils import clear_gpu_memory, get_device
from models import build_model
from train import NegativeSampler, evaluate, load_everything

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
REPORTS = Path(cfg["paths"]["reports"])

SEED = cfg["seeds"]["development"]
EPOCHS = 60
K = cfg["evaluation"]["k"]
CHUNK = 64


def content_only_ndcg(model, device, users, items, candidates, seen) -> float:
    model.eval()
    candidate_tensor = torch.as_tensor(candidates, device=device)
    position_of = {int(item): index for index, item in enumerate(candidates)}
    n_candidates = len(candidates)
    absent = torch.zeros(n_candidates, dtype=torch.bool, device=device)

    total, counted = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(users), CHUNK):
            chunk_users = users[start:start + CHUNK]
            chunk_items = items[start:start + CHUNK]
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


def run(system: str) -> pd.DataFrame:
    data = load_everything()
    device = get_device()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    model = build_model(cfg, data["n_users"], data["n_items"],
                        data["image_features"], data["text_features"],
                        data["image_available"], data["text_available"],
                        system).to(device)
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=cfg["training"]["learning_rate"],
                                 weight_decay=cfg["training"]["weight_decay"])
    sampler = NegativeSampler(data["n_items"], data["is_ghost"],
                              data["train_users"], data["train_items"], rng)

    warm_candidates = np.flatnonzero(~data["is_ghost"]).astype(np.int64)
    ghost_candidates = np.flatnonzero(data["is_ghost"]).astype(np.int64)
    position_of = {int(i): p for p, i in enumerate(warm_candidates)}

    seen: dict[int, np.ndarray] = {}
    frame = pd.DataFrame({"user": data["train_users"], "item": data["train_items"]})
    for user, group in frame.groupby("user"):
        positions = [position_of[int(i)] for i in group["item"] if int(i) in position_of]
        if positions:
            seen[int(user)] = np.array(positions, dtype=np.int64)

    items = pd.read_parquet(PROCESSED / "item_index.parquet").sort_values("item_idx")
    users = pd.read_parquet(PROCESSED / "user_index.parquet")
    ghost = pd.read_parquet(PROCESSED / "split_ghost_evaluation.parquet")
    ghost_users = ghost["user_id"].map(dict(zip(users["user_id"], users["user_idx"]))) \
                                  .to_numpy(dtype=np.int64)
    ghost_items = ghost["parent_asin"].map(dict(zip(items["parent_asin"],
                                                    items["item_idx"]))) \
                                      .to_numpy(dtype=np.int64)

    sample = rng.choice(len(data["val_users"]), size=800, replace=False)
    val_users, val_items = data["val_users"][sample], data["val_items"][sample]
    ghost_sample = rng.choice(len(ghost_users), size=4000, replace=False)
    ghost_users, ghost_items = ghost_users[ghost_sample], ghost_items[ghost_sample]

    batch_size = cfg["training"]["batch_size"]
    n_negatives = cfg["training"]["num_negatives"]
    n_train = len(data["train_users"])

    history = []
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
            loss = -F.logsigmoid(positive.repeat_interleave(n_negatives) - negative).mean()

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

        warm, _, _ = evaluate(model, device, val_users, val_items,
                              warm_candidates, seen, CHUNK)
        proxy = content_only_ndcg(model, device, val_users, val_items,
                                  warm_candidates, seen)
        ghost_score = content_only_ndcg(model, device, ghost_users, ghost_items,
                                        ghost_candidates, {})

        history.append({"system": system, "epoch": epoch, "warm": warm,
                        "cold_proxy": proxy, "ghost": ghost_score})
        print(f"    {epoch:>3}  warm {warm:.4f}  proxy {proxy:.4f}  "
              f"ghost {ghost_score:.4f}  {time.time() - started:.0f}s", flush=True)

    clear_gpu_memory()
    return pd.DataFrame(history)


def main() -> None:
    print(f"{EPOCHS} epochs, no early stopping, seed {SEED}\n")

    frames = []
    for system in ("concat_fusion", "text_only"):
        print(f"{system}")
        frames.append(run(system))
        print()

    frame = pd.concat(frames, ignore_index=True)

    print("PEAK EPOCH BY METRIC")
    print(f"  {'system':<16}{'metric':<12}{'peak':>9}{'epoch':>7}")
    for system, group in frame.groupby("system"):
        for metric in ("warm", "cold_proxy", "ghost"):
            best = group.loc[group[metric].idxmax()]
            print(f"  {system:<16}{metric:<12}{best[metric]:>9.4f}{int(best['epoch']):>7}")

    print("\nGHOST SCORE AT EACH SELECTION RULE")
    print(f"  {'system':<16}{'selected on warm':>18}{'selected on proxy':>19}"
          f"{'best possible':>15}")
    for system, group in frame.groupby("system"):
        by_warm = group.loc[group["warm"].idxmax(), "ghost"]
        by_proxy = group.loc[group["cold_proxy"].idxmax(), "ghost"]
        print(f"  {system:<16}{by_warm:>18.4f}{by_proxy:>19.4f}"
              f"{group['ghost'].max():>15.4f}")

    output = REPORTS / "pilot_horizon.csv"
    frame.to_csv(output, index=False)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()