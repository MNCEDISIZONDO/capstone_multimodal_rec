
from __future__ import annotations

# Allow this module to import the shared modules at the top of src/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import load_config

cfg = load_config()
RESULTS = Path(cfg["paths"]["results"])
PER_USER = RESULTS / "per_user"
TABLES = RESULTS / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05

# Each comparison states a question the results chapter must answer.
COMPARISONS = [
    ("text_only", "attention_fusion", "cold_start", "cold_only",
     "does the best single content signal beat attention fusion under cold-start"),
    ("text_only", "concat_fusion", "cold_start", "cold_only",
     "does the best single content signal beat concatenation under cold-start"),
    ("text_only", "image_only", "cold_start", "cold_only",
     "does text carry more cold-start signal than images"),
    ("concat_fusion", "attention_fusion", "cold_start", "cold_only",
     "which fusion mechanism ranks better on withheld items"),
    ("concat_fusion", "ncf", "control", "full",
     "does fusion beat the collaborative baseline on established items"),
    ("concat_fusion", "popularity", "control", "full",
     "does the best learned system beat the non-learned popularity anchor"),
    ("attention_fusion", "content_only", "silent", "full",
     "how differently do the fusion designs behave when text is suppressed"),
    ("text_only", "text_only", "cold_start", "cold_only",
     "self-comparison, expected to show no difference"),
]


def load_scores(system: str, condition: str, pool: str) -> tuple[np.ndarray, np.ndarray] | None:
    scores = PER_USER / f"{system}_{condition}_{pool}_ndcg.npy"
    users = PER_USER / f"{system}_{condition}_{pool}_users.npy"
    if not scores.exists() or not users.exists():
        return None
    return np.load(scores), np.load(users)


def align(a_scores: np.ndarray, a_users: np.ndarray,
          b_scores: np.ndarray, b_users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair observations by user, so that each pair concerns the same evaluation.

    Both systems are evaluated on the same split in the same order, so the
    identifier arrays should be identical; they are matched explicitly rather
    than assumed, because an unnoticed misalignment would silently compare
    unrelated observations.
    """
    if len(a_users) == len(b_users) and np.array_equal(a_users, b_users):
        return a_scores, b_scores
    order_a = np.argsort(a_users, kind="stable")
    order_b = np.argsort(b_users, kind="stable")
    shared = np.intersect1d(a_users, b_users)
    index_a = order_a[np.searchsorted(a_users[order_a], shared)]
    index_b = order_b[np.searchsorted(b_users[order_b], shared)]
    return a_scores[index_a], b_scores[index_b]


def compare(a: str, b: str, condition: str, pool: str, question: str) -> dict | None:
    left = load_scores(a, condition, pool)
    right = load_scores(b, condition, pool)
    if left is None or right is None:
        print(f"  missing scores for {a} or {b} ({condition}, {pool})")
        return None

    a_scores, b_scores = align(left[0], left[1], right[0], right[1])
    difference = a_scores - b_scores
    n_differing = int((difference != 0).sum())

    mean_a, mean_b = float(a_scores.mean()), float(b_scores.mean())
    relative = (mean_a - mean_b) / mean_b * 100 if mean_b else float("nan")

    if n_differing == 0:
        p_value = 1.0
        statistic = float("nan")
    else:
        statistic, p_value = stats.wilcoxon(a_scores, b_scores,
                                            zero_method="wilcox",
                                            alternative="two-sided")

    # Rank-biserial correlation: the share of discordant pairs favouring one
    # system minus the share favouring the other. Independent of sample size,
    # unlike the p-value.
    wins = int((difference > 0).sum())
    losses = int((difference < 0).sum())
    effect = (wins - losses) / n_differing if n_differing else 0.0

    return {
        "system_a": a, "system_b": b, "condition": condition, "pool": pool,
        "question": question,
        "n_pairs": len(a_scores), "n_differing": n_differing,
        "mean_a": round(mean_a, 6), "mean_b": round(mean_b, 6),
        "relative_gain_pct": round(relative, 2),
        "a_better": wins, "b_better": losses,
        "effect_size": round(effect, 4),
        "p_value": p_value,
        "significant": bool(p_value < ALPHA),
    }


def main() -> None:
    print(f"significance level : {ALPHA}")
    print(f"test               : paired Wilcoxon signed-rank over per-user NDCG@10")
    print(f"source             : {PER_USER}\n")

    rows = []
    for a, b, condition, pool, question in COMPARISONS:
        result = compare(a, b, condition, pool, question)
        if result is None:
            continue
        rows.append(result)

        verdict = "SIGNIFICANT" if result["significant"] else "not significant"
        print(f"{a} vs {b}  [{condition}, {pool}]")
        print(f"  {question}")
        print(f"  means {result['mean_a']:.5f} vs {result['mean_b']:.5f}   "
              f"relative {result['relative_gain_pct']:+.1f}%")
        print(f"  pairs {result['n_pairs']:,}   differing {result['n_differing']:,}   "
              f"({result['a_better']:,} favour {a}, {result['b_better']:,} favour {b})")
        print(f"  p = {result['p_value']:.3g}   effect size {result['effect_size']:+.4f}"
              f"   {verdict}\n")

    table = pd.DataFrame(rows)
    output = TABLES / "significance_tests.csv"
    table.to_csv(output, index=False)

    print("SUMMARY")
    for row in table.itertuples():
        marker = "*" if row.significant else " "
        print(f" {marker} {row.system_a:<17} vs {row.system_b:<15} "
              f"{row.condition:<11}{row.relative_gain_pct:>+7.1f}%   "
              f"p={row.p_value:.3g}")

    print("\n  * significant at the 0.05 level")
    print("\nA p-value states whether a difference is distinguishable from chance;")
    print("the effect size states whether it is large enough to matter. With")
    print("thousands of paired observations, a small difference can be")
    print("significant, so both are reported for every comparison.")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()