"""Train every configuration across the statistical seeds (manual section 4.3).

Tuning is complete and the configuration is locked, so replication is now
meaningful: each run differs from the others only in the random seed governing
initialisation and sampling order. Reporting a mean with its standard deviation
across those runs distinguishes a stable result from a fluctuation, which a
single run cannot do.

The set of configurations covers both the headline comparison and the structural
ablation. The ablation requires every non-empty combination of the three signals,
so that the contribution of each can be attributed: seven combinations, of which
the two mixing the interaction signal with a single content signal are not part
of the headline comparison but are required to complete the matrix.

The batch is resumable. A configuration whose summary file already exists is
skipped, so an interruption costs only the run in progress.
"""
from __future__ import annotations

# Allow this module to import the shared modules at the top of src/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import time
from pathlib import Path

import pandas as pd

from config import load_config
from train import train

cfg = load_config()
LOGS = Path(cfg["paths"]["logs"])
REPORTS = Path(cfg["paths"]["reports"])
REPORTS.mkdir(parents=True, exist_ok=True)

SEEDS = cfg["seeds"]["statistical"]

# The six systems of the headline comparison, plus the two signal combinations
# needed to complete the seven-way structural ablation.
CONFIGURATIONS = [
    "ncf",
    "image_only",
    "text_only",
    "content_only",
    "interaction_image",
    "interaction_text",
    "concat_fusion",
    "attention_fusion",
]


def already_done(system: str, seed: int) -> bool:
    return (LOGS / f"{system}_seed{seed}.json").exists()


def main() -> None:
    planned = [(system, seed) for system in CONFIGURATIONS for seed in SEEDS]
    remaining = [(s, d) for s, d in planned if not already_done(s, d)]

    print(f"configurations : {len(CONFIGURATIONS)}")
    print(f"seeds          : {SEEDS}")
    print(f"runs planned   : {len(planned)}")
    print(f"already done   : {len(planned) - len(remaining)}")
    print(f"to run         : {len(remaining)}\n")

    started = time.time()
    for index, (system, seed) in enumerate(remaining, start=1):
        elapsed = time.time() - started
        estimate = (elapsed / (index - 1) * len(remaining)) if index > 1 else 0
        print(f"[{index}/{len(remaining)}] {system} seed {seed}"
              f"   elapsed {elapsed / 60:.0f}m"
              f"{f'   estimated total {estimate / 60:.0f}m' if estimate else ''}",
              flush=True)
        summary = train(system, seed, quiet=True)
        print(f"            NDCG {summary['best_val_ndcg']:.4f}   "
              f"coverage {summary['coverage_at_best']:>5,}   "
              f"epoch {summary['best_epoch']}\n", flush=True)

    # Collect every completed run, including any from earlier invocations.
    records = []
    for system in CONFIGURATIONS:
        for seed in SEEDS:
            path = LOGS / f"{system}_seed{seed}.json"
            if path.exists():
                records.append(json.loads(path.read_text()))

    frame = pd.DataFrame(records)
    frame.to_csv(REPORTS / "seed_runs.csv", index=False)

    summary = (frame.groupby("system")
               .agg(runs=("best_val_ndcg", "size"),
                    ndcg_mean=("best_val_ndcg", "mean"),
                    ndcg_std=("best_val_ndcg", "std"),
                    coverage_mean=("coverage_at_best", "mean"),
                    coverage_std=("coverage_at_best", "std"),
                    epoch_mean=("best_epoch", "mean"))
               .round(5)
               .sort_values("ndcg_mean", ascending=False))
    summary.to_csv(REPORTS / "seed_summary.csv")

    print("VALIDATION NDCG@10 ACROSS SEEDS")
    print(f"  {'system':<20}{'runs':>6}{'mean':>10}{'std':>9}{'coverage':>11}")
    for system, row in summary.iterrows():
        print(f"  {system:<20}{int(row['runs']):>6}{row['ndcg_mean']:>10.4f}"
              f"{row['ndcg_std']:>9.4f}{row['coverage_mean']:>11.0f}")

    print(f"\ntotal time {(time.time() - started) / 60:.0f} minutes")
    print(f"Wrote {REPORTS / 'seed_runs.csv'}")
    print(f"Wrote {REPORTS / 'seed_summary.csv'}")


if __name__ == "__main__":
    main()