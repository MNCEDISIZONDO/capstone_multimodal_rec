
from __future__ import annotations

# Allow this module to import the shared modules at the top of src/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from config import load_config
from device_utils import clear_gpu_memory, get_device
from models import SYSTEMS, PopularityBaseline, RandomBaseline, build_model

cfg = load_config()
PROCESSED = Path(cfg["paths"]["processed_data"])
FEATURES = Path(cfg["paths"]["features"])
CHECKPOINTS = Path(cfg["paths"]["checkpoints"])
RESULTS = Path(cfg["paths"]["results"])
TABLES = RESULTS / "tables"
PER_USER = RESULTS / "per_user"
TABLES.mkdir(parents=True, exist_ok=True)
PER_USER.mkdir(parents=True, exist_ok=True)

K = cfg["evaluation"]["k"]
SEED = cfg["seeds"]["development"]
CHUNK = cfg["training"]["validation_chunk_users"]


def load_inputs() -> dict:
    items = pd.read_parquet(PROCESSED / "item_index.parquet").sort_values("item_idx")
    users = pd.read_parquet(PROCESSED / "user_index.parquet")
    item_of = dict(zip(items["parent_asin"], items["item_idx"]))
    user_of = dict(zip(users["user_id"], users["user_idx"]))

    def to_indices(name: str) -> tuple[np.ndarray, np.ndarray]:
        frame = pd.read_parquet(PROCESSED / name)
        return (frame["user_id"].map(user_of).to_numpy(dtype=np.int64),
                frame["parent_asin"].map(item_of).to_numpy(dtype=np.int64))

    with h5py.File(FEATURES / "image_embeddings.h5", "r") as store:
        image_features, image_available = store["embeddings"][:], store["available"][:]
    with h5py.File(FEATURES / "text_embeddings.h5", "r") as store:
        text_features, text_available = store["embeddings"][:], store["available"][:]

    train_users, train_items = to_indices("split_train.parquet")
    test_users, test_items = to_indices("split_test.parquet")
    cold_users, cold_items = to_indices("split_ghost_evaluation.parquet")

    trained = set(train_items.tolist())
    no_image = set(np.flatnonzero(~image_available).tolist())
    untrained = {int(i) for i in np.unique(test_items) if int(i) not in trained}

    return dict(
        items=items, users=users,
        n_users=len(users), n_items=len(items),
        train_users=train_users, train_items=train_items,
        test_users=test_users, test_items=test_items,
        cold_users=cold_users, cold_items=cold_items,
        image_features=image_features, image_available=image_available,
        text_features=text_features, text_available=text_available,
        is_ghost=items["is_ghost"].to_numpy().copy(),
        no_image=no_image, untrained=untrained,
    )


def build_seen(train_users: np.ndarray, train_items: np.ndarray,
               position_of: dict[int, int]) -> dict[int, np.ndarray]:
    """Map each user to the candidate positions they interacted with in training."""
    seen: dict[int, np.ndarray] = {}
    frame = pd.DataFrame({"user": train_users, "item": train_items})
    for user, group in frame.groupby("user"):
        positions = [position_of[int(i)] for i in group["item"] if int(i) in position_of]
        if positions:
            seen[int(user)] = np.array(positions, dtype=np.int64)
    return seen


def metrics_from_ranks(ranks: np.ndarray) -> dict[str, float]:
    """Ranking metrics for a leave-one-out protocol with one relevant item."""
    if len(ranks) == 0:
        return {"ndcg": 0.0, "hit_rate": 0.0, "mrr": 0.0}
    within = ranks <= K
    return {
        "ndcg": float(np.where(within, 1.0 / np.log2(ranks + 1.0), 0.0).mean()),
        "hit_rate": float(within.mean()),
        "mrr": float(np.where(within, 1.0 / ranks, 0.0).mean()),
    }


def evaluate_model(model, device, eval_users: np.ndarray, eval_items: np.ndarray,
                   candidates: np.ndarray, seen: dict[int, np.ndarray],
                   collaborative_available: np.ndarray | None,
                   image_enabled: bool = True, text_enabled: bool = True,
                   restrict_to: np.ndarray | None = None,
                   exclude_targets: set[int] | None = None) -> dict:
    """Rank every held-out item and return per-user ranks plus coverage.

    restrict_to, when given, names a subset of the candidate list against which
    ranks are additionally computed, so that one scoring pass yields both the
    full-catalogue and the restricted-pool measurement.

    exclude_targets names held-out items whose evaluation is not meaningful for
    this condition; the corresponding pairs are omitted from the reported metric.
    """
    model.eval()
    exclude_targets = exclude_targets or set()
    candidate_tensor = torch.as_tensor(candidates, device=device)
    position_of = {int(item): index for index, item in enumerate(candidates)}
    available_tensor = (None if collaborative_available is None
                        else torch.as_tensor(collaborative_available, device=device))

    restrict_positions = None
    if restrict_to is not None:
        restrict_positions = torch.as_tensor(
            np.array([position_of[int(i)] for i in restrict_to]), device=device)

    all_ranks, restricted_ranks, evaluated_users = [], [], []
    recommended: set[int] = set()

    with torch.no_grad():
        for start in range(0, len(eval_users), CHUNK):
            chunk_users = eval_users[start:start + CHUNK]
            chunk_items = eval_items[start:start + CHUNK]
            n_chunk = len(chunk_users)

            positions = np.array([position_of.get(int(i), -1) for i in chunk_items])
            keep = (positions >= 0) & np.array(
                [int(i) not in exclude_targets for i in chunk_items])
            if not keep.any():
                continue

            user_tensor = torch.as_tensor(chunk_users, device=device)
            n_candidates = len(candidates)
            expanded_available = (None if available_tensor is None
                                  else available_tensor.repeat(n_chunk))
            scores = model(user_tensor.repeat_interleave(n_candidates),
                           candidate_tensor.repeat(n_chunk),
                           collaborative_available=expanded_available,
                           image_enabled=image_enabled,
                           text_enabled=text_enabled).view(n_chunk, n_candidates)

            # Exclusion indices are assembled on the host and applied in a
            # single scattered write, avoiding a device round trip per user.
            rows, columns = [], []
            for row, user in enumerate(chunk_users):
                already_seen = seen.get(int(user))
                if already_seen is not None and len(already_seen):
                    rows.append(np.full(len(already_seen), row, dtype=np.int64))
                    columns.append(already_seen)
            if rows:
                scores[torch.as_tensor(np.concatenate(rows), device=device),
                       torch.as_tensor(np.concatenate(columns), device=device)] = float("-inf")

            recommended.update(
                scores.topk(K, dim=1).indices.cpu().numpy().reshape(-1).tolist())

            target_positions = torch.as_tensor(np.where(keep, positions, 0), device=device)
            target_scores = scores.gather(1, target_positions.unsqueeze(1))

            # Items scoring identically to the target occupy a block of
            # consecutive rank positions; the target's expected position within
            # that block is its midpoint.
            def midrank(matrix: torch.Tensor) -> np.ndarray:
                greater = (matrix > target_scores).sum(dim=1).float()
                tied = (matrix == target_scores).sum(dim=1).float()
                return (greater + (tied + 1.0) / 2.0).cpu().numpy()

            all_ranks.append(midrank(scores)[keep])
            evaluated_users.append(chunk_users[keep])

            if restrict_positions is not None:
                subset = scores.index_select(1, restrict_positions)
                restricted_ranks.append(midrank(subset)[keep])

    result = {
        "ranks": np.concatenate(all_ranks) if all_ranks else np.array([]),
        "users": np.concatenate(evaluated_users) if evaluated_users else np.array([]),
        "coverage": len(recommended),
        "n_candidates": len(candidates),
    }
    if restrict_positions is not None:
        result["restricted_ranks"] = (np.concatenate(restricted_ranks)
                                      if restricted_ranks else np.array([]))
        result["n_restricted"] = len(restrict_to)
    return result


def evaluate_scorer(scorer, eval_users: np.ndarray, eval_items: np.ndarray,
                    candidates: np.ndarray, seen: dict[int, np.ndarray],
                    restrict_to: np.ndarray | None = None,
                    exclude_targets: set[int] | None = None) -> dict:
    """Evaluate a non-learned baseline, which scores items independently of the user."""
    exclude_targets = exclude_targets or set()
    position_of = {int(item): index for index, item in enumerate(candidates)}
    base = scorer.score(candidates).astype(np.float64)
    restrict_positions = (None if restrict_to is None
                          else np.array([position_of[int(i)] for i in restrict_to]))

    ranks, restricted, users_kept = [], [], []
    recommended: set[int] = set()

    for user, target in zip(eval_users, eval_items):
        target_position = position_of.get(int(target))
        if target_position is None or int(target) in exclude_targets:
            continue
        row = base.copy()
        already_seen = seen.get(int(user))
        if already_seen is not None and len(already_seen):
            row[already_seen] = -np.inf
        recommended.update(np.argpartition(-row, K)[:K].tolist())

        target_score = row[target_position]
        ranks.append((row > target_score).sum() + ((row == target_score).sum() + 1) / 2)
        users_kept.append(user)
        if restrict_positions is not None:
            subset = row[restrict_positions]
            restricted.append((subset > target_score).sum()
                              + ((subset == target_score).sum() + 1) / 2)

    result = {"ranks": np.array(ranks), "users": np.array(users_kept),
              "coverage": len(recommended), "n_candidates": len(candidates)}
    if restrict_positions is not None:
        result["restricted_ranks"] = np.array(restricted)
        result["n_restricted"] = len(restrict_to)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all systems.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="")
    arguments = parser.parse_args()

    device = get_device()
    data = load_inputs()

    warm_candidates = np.flatnonzero(~data["is_ghost"]).astype(np.int64)
    all_candidates = np.arange(data["n_items"], dtype=np.int64)
    cold_candidates = np.flatnonzero(data["is_ghost"]).astype(np.int64)

    warm_position = {int(i): p for p, i in enumerate(warm_candidates)}
    all_position = {int(i): p for p, i in enumerate(all_candidates)}
    seen_warm = build_seen(data["train_users"], data["train_items"], warm_position)
    seen_all = build_seen(data["train_users"], data["train_items"], all_position)

    # Under the cold-start condition the candidate list contains both withheld
    # and established items. Each is scored through the path appropriate to it,
    # which is the situation a deployed catalogue presents.
    collaborative_available = ~data["is_ghost"]

    # Exclusions differ by condition. Items lacking an image are not
    # distinguishable under Blind, where the image is suppressed for every item.
    exclude_control = data["no_image"] | data["untrained"]
    exclude_blind = data["untrained"]
    exclude_silent = data["no_image"] | data["untrained"]

    print(f"users              : {data['n_users']:,}")
    print(f"items              : {data['n_items']:,}")
    print(f"warm candidates    : {len(warm_candidates):,}")
    print(f"cold-start items   : {len(cold_candidates):,}")
    print(f"test interactions  : {len(data['test_users']):,}")
    print(f"cold interactions  : {len(data['cold_users']):,}")
    print(f"excluded from warm : {len(data['no_image'])} without an image, "
          f"{len(data['untrained'])} without training history\n")

    suffix = f"_{arguments.tag}" if arguments.tag else ""

    rows = []

    def record(system: str, condition: str, result: dict, pool: str) -> None:
        ranks = result["ranks"] if pool == "full" else result["restricted_ranks"]
        n_candidates = (result["n_candidates"] if pool == "full"
                        else result["n_restricted"])
        summary = metrics_from_ranks(ranks)
        summary.update({
            "system": system, "condition": condition, "candidate_pool": pool,
            "n_candidates": n_candidates, "n_evaluated": len(ranks),
            "coverage_items": result["coverage"] if pool == "full" else None,
            "coverage_share": (round(result["coverage"] / result["n_candidates"], 5)
                               if pool == "full" else None),
        })
        rows.append(summary)
        print(f"  {system:<18}{condition:<12}{pool:<11}"
              f"NDCG {summary['ndcg']:.4f}  HR {summary['hit_rate']:.4f}  "
              f"MRR {summary['mrr']:.4f}  n={len(ranks):,}")
        stem = f"{system}_{condition}_{pool}{suffix}"
        np.save(PER_USER / f"{stem}_ndcg.npy",
                np.where(ranks <= K, 1.0 / np.log2(ranks + 1.0), 0.0))
        np.save(PER_USER / f"{stem}_users.npy", result["users"])

    print("NON-LEARNED ANCHORS")
    for name, scorer in [("popularity", PopularityBaseline(data["n_items"],
                                                           data["train_items"])),
                         ("random", RandomBaseline(arguments.seed))]:
        record(name, "control", evaluate_scorer(
            scorer, data["test_users"], data["test_items"],
            warm_candidates, seen_warm, exclude_targets=exclude_control), "full")
        cold = evaluate_scorer(scorer, data["cold_users"], data["cold_items"],
                               all_candidates, seen_all, restrict_to=cold_candidates)
        record(name, "cold_start", cold, "full")
        record(name, "cold_start", cold, "cold_only")

    print("\nLEARNED SYSTEMS")
    for system in SYSTEMS:
        name = f"{system}{('_' + arguments.tag) if arguments.tag else ''}_seed{arguments.seed}"
        path = CHECKPOINTS / f"{name}.pt"
        if not path.exists():
            print(f"  {system:<18}no checkpoint, skipped")
            continue

        model = build_model(cfg, data["n_users"], data["n_items"],
                            data["image_features"], data["text_features"],
                            data["image_available"], data["text_available"],
                            system).to(device)
        model.load_state_dict(torch.load(path, map_location=device)["state_dict"])

        record(system, "control", evaluate_model(
            model, device, data["test_users"], data["test_items"],
            warm_candidates, seen_warm, None,
            exclude_targets=exclude_control), "full")

        if model.use_image:
            record(system, "blind", evaluate_model(
                model, device, data["test_users"], data["test_items"],
                warm_candidates, seen_warm, None, image_enabled=False,
                exclude_targets=exclude_blind), "full")
        if model.use_text:
            record(system, "silent", evaluate_model(
                model, device, data["test_users"], data["test_items"],
                warm_candidates, seen_warm, None, text_enabled=False,
                exclude_targets=exclude_silent), "full")

        cold = evaluate_model(
            model, device, data["cold_users"], data["cold_items"],
            all_candidates, seen_all, collaborative_available,
            restrict_to=cold_candidates)
        record(system, "cold_start", cold, "full")
        record(system, "cold_start", cold, "cold_only")

        clear_gpu_memory()

    table = pd.DataFrame(rows)[
        ["system", "condition", "candidate_pool", "ndcg", "hit_rate", "mrr",
         "coverage_items", "coverage_share", "n_candidates", "n_evaluated"]]
    
    output = TABLES / f"evaluation_summary{suffix}.csv"
    table.to_csv(output, index=False)

    (TABLES / "evaluation_context.json").write_text(json.dumps({
        "seed": arguments.seed, "k": K,
        "users": int(data["n_users"]), "items": int(data["n_items"]),
        "warm_candidates": int(len(warm_candidates)),
        "cold_start_items": int(len(cold_candidates)),
        "excluded_no_image": sorted(int(i) for i in data["no_image"]),
        "excluded_untrained": sorted(int(i) for i in data["untrained"]),
    }, indent=2))

    print(f"\nWrote {output}")
    print(f"Wrote per-user scores to {PER_USER}")


if __name__ == "__main__":
    main()