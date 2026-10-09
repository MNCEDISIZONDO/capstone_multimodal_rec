"""Tuning sweep over the parameters governing modality competition in the attention fusion model.
"""
from __future__ import annotations

# Allow this module to import the shared modules at the top of src/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import itertools
from pathlib import Path

import pandas as pd

from config import load_config
from train import train

cfg = load_config()
REPORTS = Path(cfg["paths"]["reports"])
REPORTS.mkdir(parents=True, exist_ok=True)

SYSTEM = "attention_fusion"
SEED = cfg["seeds"]["development"]

COLLABORATIVE_DROPOUT = [0.2, 0.5, 0.8, 1.0]
EMBEDDING_DIM = [64, 16]


def main() -> None:
    combinations = list(itertools.product(COLLABORATIVE_DROPOUT, EMBEDDING_DIM))
    print(f"system         : {SYSTEM}")
    print(f"configurations : {len(combinations)}")
    print(f"baseline       : dropout 0.2, embedding 64 "
          f"(NDCG 0.0281, coverage 236)\n")

    results = []
    for index, (dropout, dimension) in enumerate(combinations, start=1):
        tag = f"cd{int(dropout * 100)}_ed{dimension}"
        print(f"[{index}/{len(combinations)}] dropout {dropout}, embedding {dimension}",
              flush=True)
        summary = train(
            SYSTEM, SEED, tag=tag, quiet=True,
            overrides={"model.collaborative_dropout": dropout,
                       "model.embedding_dim": dimension},
        )
        results.append(summary)
        print(f"            NDCG {summary['best_val_ndcg']:.4f}   "
              f"coverage {summary['coverage_at_best']:>5,}   "
              f"epoch {summary['best_epoch']}   "
              f"({summary['epochs_run']} run)\n", flush=True)

        # Written after every configuration, so an interrupted sweep still
        # leaves a usable record of the configurations that completed.
        pd.DataFrame(results).to_csv(REPORTS / "tuning_sweep.csv", index=False)

    table = pd.DataFrame(results).sort_values("best_val_ndcg", ascending=False)
    print("RESULTS, best first")
    print(f"  {'dropout':>8}{'embed':>7}{'NDCG@10':>10}{'coverage':>10}{'epoch':>7}")
    for row in table.itertuples():
        print(f"  {row.collaborative_dropout:>8}{row.embedding_dim:>7}"
              f"{row.best_val_ndcg:>10.4f}{row.coverage_at_best:>10,}"
              f"{row.best_epoch:>7}")

    print(f"\nWrote {REPORTS / 'tuning_sweep.csv'}")


if __name__ == "__main__":
    main()