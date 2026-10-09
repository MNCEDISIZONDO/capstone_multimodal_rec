"""Evaluate every trained seed and summarise the results.

Evaluate each seed independently using the same protocol. Report the mean and
standard deviation across seeds, while retaining the individual seed results
to show how much performance varies between training runs.
"""

from __future__ import annotations

# Allow this module to import the shared modules at the top of src/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from config import load_config

cfg = load_config()
RESULTS = Path(cfg["paths"]["results"])
TABLES = RESULTS / "tables"
SEEDS = cfg["seeds"]["statistical"]

EVALUATE = Path(__file__).resolve().parent / "evaluate.py"

KEY = ["system", "condition", "candidate_pool"]
METRICS = ["ndcg", "hit_rate", "mrr"]


def per_seed_path(seed: int) -> Path:
    return TABLES / f"evaluation_summary_seed{seed}.csv"


def main() -> None:
    outstanding = [s for s in SEEDS if not per_seed_path(s).exists()]
    print(f"seeds          : {SEEDS}")
    print(f"already done   : {len(SEEDS) - len(outstanding)}")
    print(f"to evaluate    : {len(outstanding)}\n")

    started = time.time()
    for index, seed in enumerate(outstanding, start=1):
        elapsed = time.time() - started
        estimate = (elapsed / (index - 1) * len(outstanding)) if index > 1 else 0
        print(f"[{index}/{len(outstanding)}] seed {seed}"
              f"   elapsed {elapsed / 60:.0f}m"
              f"{f'   estimated total {estimate / 60:.0f}m' if estimate else ''}",
              flush=True)

        # The evaluation is invoked as a separate process so that each seed
        # starts from a clean allocator state, which matters on a device where
        # memory pressure degrades speed rather than raising an error.
        #
        # No tag is passed, because the tag also forms part of the checkpoint
        # filename the evaluation derives, and the training runs wrote plain
        # per-seed names. The output file is therefore renamed afterwards.
        completed = subprocess.run(
            [sys.executable, str(EVALUATE), "--seed", str(seed)],
            capture_output=True, text=True)

        if completed.returncode != 0:
            print(completed.stdout[-2000:])
            print(completed.stderr[-2000:])
            raise SystemExit(f"evaluation failed for seed {seed}")

        default_output = TABLES / "evaluation_summary.csv"
        if not default_output.exists():
            print(completed.stdout[-2000:])
            raise SystemExit("evaluation produced no summary file")

        # A run in which every learned system was skipped would leave only the
        # two non-learned anchors, which indicates the checkpoints were not
        # found rather than that the evaluation succeeded.
        produced = pd.read_csv(default_output)
        learned = set(produced["system"]) - {"popularity", "random"}
        if not learned:
            print(completed.stdout[-2000:])
            raise SystemExit(
                f"seed {seed}: no learned system was evaluated; "
                "checkpoints were not found")

        default_output.rename(per_seed_path(seed))
        print(f"            {len(learned)} systems, "
              f"wrote {per_seed_path(seed).name}\n", flush=True)

    frames = []
    for seed in SEEDS:
        path = per_seed_path(seed)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frames.append(frame)

    if not frames:
        raise SystemExit("no per-seed results found")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(TABLES / "evaluation_per_seed.csv", index=False)

    aggregate = (combined.groupby(KEY)[METRICS]
                 .agg(["mean", "std", "count"])
                 .round(6))
    aggregate.columns = [f"{metric}_{statistic}" for metric, statistic in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.to_csv(TABLES / "evaluation_aggregate.csv", index=False)

    print("AGGREGATE NDCG@10 ACROSS SEEDS")
    for (condition, pool), group in aggregate.groupby(["condition", "candidate_pool"]):
        print(f"\n  {condition} / {pool}")
        for row in group.sort_values("ndcg_mean", ascending=False).itertuples():
            print(f"    {row.system:<20}{row.ndcg_mean:.4f} ± {row.ndcg_std:.4f}"
                  f"   ({int(row.ndcg_count)} seeds)")

    print(f"\ntotal time {(time.time() - started) / 60:.0f} minutes")
    print(f"Wrote {TABLES / 'evaluation_per_seed.csv'}")
    print(f"Wrote {TABLES / 'evaluation_aggregate.csv'}")


if __name__ == "__main__":
    main()