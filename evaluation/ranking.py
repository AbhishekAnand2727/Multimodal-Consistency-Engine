"""
Pairwise ranking analysis.

Measures whether the scoring system correctly ranks dubs — i.e.,
does a 'good' dub consistently score higher than a 'bad' one?

This is a stricter test than class separation:
  - Separation asks: "are means different?"
  - Ranking asks: "does the system get the order right, sample by sample?"

Method: Exhaustive pairwise comparison across all samples.
For every pair (i, j) where label_i is higher quality than label_j,
check whether score_i > score_j.

Ranking accuracy = correct_pairs / total_comparable_pairs

This is equivalent to a 1-sided Mann-Whitney U statistic normalized to [0, 1].
A value of 0.5 = random; 1.0 = perfect ranking.

Improvements over binary accuracy:
  - Margin analysis: measures *how confidently* rankings are correct
  - Margin-weighted accuracy: penalizes large errors more than small ones
  - Adaptive tie margin: scales with the dataset's score range
  - Dynamic label ordering: derived from data, not hardcoded
  - Imbalance detection: flags when one class pair dominates

v2 additions (statistical robustness):
  - Normalized margins: scale-invariant (divided by score range)
  - Clipped margin weighting: reduces outlier influence (clip at 0.3)
  - Confidence-weighted accuracy: pairs weighted by min(conf_i, conf_j)
  - Bootstrap stability analysis: mean/std of accuracy and margin over resamples
"""

import numpy as np
from dataclasses import dataclass, field
from loguru import logger

from evaluation.dataset_loader import EvalResult


# -----------------------------------------------------------------------
# RankingAnalysis dataclass — extended with margin + imbalance fields
# -----------------------------------------------------------------------

@dataclass
class RankingAnalysis:
    """Results of pairwise ranking evaluation."""
    # --- Binary accuracy ---
    accuracy: float                         # correct_pairs / total_pairs
    correct_pairs: int
    total_pairs: int
    tied_pairs: int
    per_class_accuracy: dict[str, float]
    per_class_counts: dict[str, int]

    # --- Margin analysis (raw — score_high - score_low) ---
    # margin = score_high - score_low for each non-tied pair
    # Positive = correct direction, negative = incorrect direction
    mean_margin: float                      # mean across all non-tied pairs
    median_margin: float
    margin_std: float
    per_class_margin: dict[str, float]      # mean margin per class pair

    # --- Margin-weighted accuracy ---
    # weighted_accuracy = sum(margin if correct else 0) / sum(|margin|)
    # Rewards large correct margins, penalizes large incorrect margins
    weighted_accuracy: float

    # --- Imbalance ---
    pair_distribution: dict[str, int]       # total pairs per class pair (incl. ties)
    imbalance_warning: bool                 # True if one pair > 60% of total

    # --- Interpretation ---
    interpretation: str

    # --- Normalized margins (scale-invariant: divided by score range) ---
    # These are the preferred metrics for cross-dataset comparison.
    normalized_mean_margin: float           # mean normalized margin
    normalized_per_class_margin: dict[str, float]
    normalized_weighted_accuracy: float     # weighted accuracy using normalized margins

    # --- Clipped margin weighting (outlier-robust) ---
    # clipped_margin = sign(m) * min(|m|, 0.3) applied to raw margins
    # Prevents extreme outlier pairs from dominating the weighted score.
    clipped_weighted_accuracy: float

    # --- Confidence-weighted accuracy ---
    # Pairs weighted by min(conf_i, conf_j) — high-confidence pairs matter more.
    # Uses normalized margins for scale invariance.
    confidence_weighted_accuracy: float

    # --- Bootstrap stability ---
    # stability = {mean_accuracy, std_accuracy, mean_margin, std_margin, high_variance}
    # high_variance = True if std_accuracy > 0.10 (results unreliable)
    stability: dict = field(default_factory=dict)


# -----------------------------------------------------------------------
# Core analysis function
# -----------------------------------------------------------------------

def pairwise_ranking_analysis(results: list[EvalResult]) -> RankingAnalysis:
    """
    Compute pairwise ranking accuracy with margin analysis across all
    valid result pairs.

    Improvements vs original:
      1. Adaptive tie margin: scales with observed score range (1% of range)
      2. Dynamic label ordering: derived from validate_label_ordering(),
         not from a hardcoded LABEL_RANK dict
      3. Margin tracking: mean/median/std of score_high - score_low
      4. Margin-weighted accuracy: margin-magnitude-weighted ranking score
      5. Imbalance detection: warns if one class pair dominates

    v2 additions:
      6. Normalized margins: divided by score_range for scale invariance
      7. Clipped margin weighting: clip_threshold=0.3 to reduce outlier influence
      8. Confidence-weighted accuracy: weighted by min(conf_i, conf_j)
      9. Bootstrap stability: mean/std over 100 resamples

    Args:
        results: List of evaluation results (valid and invalid).

    Returns:
        RankingAnalysis with all metrics.
    """
    valid = [r for r in results if r.is_valid]
    if len(valid) < 2:
        logger.warning("Need at least 2 valid samples for ranking analysis.")
        return _empty_ranking()

    # --- 1. Dynamic label ordering from observed medians ---
    from evaluation.analysis import validate_label_ordering
    ordered_labels = validate_label_ordering(results)

    if len(ordered_labels) < 2:
        logger.warning("Need at least 2 label classes for ranking analysis.")
        return _empty_ranking()

    # Assign numeric ranks: highest quality label gets highest rank
    label_rank = {
        label: len(ordered_labels) - i
        for i, label in enumerate(ordered_labels)
    }
    logger.info(f"Dynamic label ranks: {label_rank}")

    # Build comparable pairs from ordered labels (higher first)
    comparable_pairs = [
        (ordered_labels[i], ordered_labels[j])
        for i in range(len(ordered_labels))
        for j in range(i + 1, len(ordered_labels))
    ]

    # --- 2. Adaptive tie margin + normalization epsilon ---
    all_scores = np.array([r.overall_score for r in valid])
    score_range = float(all_scores.max() - all_scores.min())
    tie_margin = 0.01 * score_range if score_range > 1e-6 else 1e-4
    # Normalization epsilon: avoids divide-by-zero when score range is degenerate
    norm_epsilon = score_range if score_range > 1e-6 else 1e-6
    logger.info(
        f"Score range: {score_range:.4f}, adaptive tie margin: {tie_margin:.4f}, "
        f"norm_epsilon: {norm_epsilon:.6f}"
    )

    # --- 3. Group by label ---
    by_label: dict[str, list[EvalResult]] = {}
    for r in valid:
        if r.label in label_rank:
            by_label.setdefault(r.label, []).append(r)

    # --- 4. Pairwise comparison ---
    CLIP_THRESHOLD = 0.3   # for clipped margin weighting

    total_correct = 0
    total_pairs = 0          # excludes ties
    total_tied = 0
    per_class_correct: dict[str, int] = {}
    per_class_total: dict[str, int] = {}   # excludes ties
    pair_distribution: dict[str, int] = {} # all pairs including ties

    # Collect margins for all flavours of statistics
    all_margins: list[float] = []                      # signed raw
    all_normalized_margins: list[float] = []           # signed normalized
    all_clipped_margins: list[float] = []              # sign(m) * min(|m|, clip)
    per_class_margins: dict[str, list[float]] = {}
    per_class_normalized_margins: dict[str, list[float]] = {}

    # Confidence weighting: pair_weight = min(conf_i, conf_j)
    conf_weighted_correct_sum = 0.0   # sum(w * |norm_margin|) for correct pairs
    conf_weighted_total_sum = 0.0     # sum(w * |norm_margin|) for all pairs

    for label_high, label_low in comparable_pairs:
        highs = by_label.get(label_high, [])
        lows = by_label.get(label_low, [])
        if not highs or not lows:
            continue

        key = f"{label_high} vs {label_low}"
        per_class_correct[key] = 0
        per_class_total[key] = 0
        pair_distribution[key] = 0
        per_class_margins[key] = []
        per_class_normalized_margins[key] = []

        for r_high in highs:
            for r_low in lows:
                # Raw margin: positive = correct direction
                margin = r_high.overall_score - r_low.overall_score
                norm_margin = margin / norm_epsilon
                pair_distribution[key] += 1

                if abs(margin) <= tie_margin:
                    total_tied += 1
                    # Ties don't contribute to margin or accuracy stats
                    continue

                # Non-tied pair
                total_pairs += 1
                per_class_total[key] += 1

                all_margins.append(margin)
                all_normalized_margins.append(norm_margin)

                # Clipped margin: cap magnitude to reduce outlier influence
                clipped = float(np.sign(margin)) * min(abs(margin), CLIP_THRESHOLD)
                all_clipped_margins.append(clipped)

                # Confidence weighting (scale-invariant: uses |norm_margin|)
                pair_weight = min(r_high.confidence, r_low.confidence)
                conf_weighted_total_sum += pair_weight * abs(norm_margin)

                per_class_margins[key].append(margin)
                per_class_normalized_margins[key].append(norm_margin)

                if margin > 0:
                    # Correct ranking
                    total_correct += 1
                    per_class_correct[key] += 1
                    conf_weighted_correct_sum += pair_weight * abs(norm_margin)

    # --- 5. Accuracy ---
    overall_accuracy = total_correct / total_pairs if total_pairs > 0 else 0.0

    per_class_accuracy = {
        key: per_class_correct[key] / per_class_total[key]
        if per_class_total[key] > 0 else 0.0
        for key in per_class_total
    }

    # --- 6. Raw margin statistics ---
    if all_margins:
        margins_arr = np.array(all_margins)
        mean_margin = float(np.mean(margins_arr))
        median_margin = float(np.median(margins_arr))
        margin_std = float(np.std(margins_arr))
    else:
        mean_margin = median_margin = margin_std = 0.0

    per_class_margin = {
        key: float(np.mean(per_class_margins[key]))
        if per_class_margins[key] else 0.0
        for key in per_class_margins
    }

    # --- 7. Raw margin-weighted accuracy ---
    # weighted_accuracy = sum(margin if correct else 0) / sum(|margin|)
    if all_margins:
        abs_sum = sum(abs(m) for m in all_margins)
        correct_sum = sum(m for m in all_margins if m > 0)
        weighted_accuracy = correct_sum / abs_sum if abs_sum > 0 else 0.0
    else:
        weighted_accuracy = 0.0

    # --- 8. Normalized margin statistics ---
    if all_normalized_margins:
        norm_arr = np.array(all_normalized_margins)
        normalized_mean_margin = float(np.mean(norm_arr))
    else:
        normalized_mean_margin = 0.0

    normalized_per_class_margin = {
        key: float(np.mean(per_class_normalized_margins[key]))
        if per_class_normalized_margins[key] else 0.0
        for key in per_class_normalized_margins
    }

    if all_normalized_margins:
        norm_abs_sum = sum(abs(m) for m in all_normalized_margins)
        norm_correct_sum = sum(m for m in all_normalized_margins if m > 0)
        normalized_weighted_accuracy = (
            norm_correct_sum / norm_abs_sum if norm_abs_sum > 0 else 0.0
        )
    else:
        normalized_weighted_accuracy = 0.0

    # --- 9. Clipped margin-weighted accuracy ---
    # Uses raw margins clipped at CLIP_THRESHOLD to reduce outlier influence.
    # A pair with margin=0.9 has the same weight as margin=0.3 — preventing
    # a handful of extreme pairs from determining the final score.
    if all_clipped_margins:
        clip_abs_sum = sum(abs(m) for m in all_clipped_margins)
        clip_correct_sum = sum(m for m in all_clipped_margins if m > 0)
        clipped_weighted_accuracy = (
            clip_correct_sum / clip_abs_sum if clip_abs_sum > 0 else 0.0
        )
    else:
        clipped_weighted_accuracy = 0.0

    logger.info(
        f"Clipped margin weighting (threshold={CLIP_THRESHOLD}): "
        f"clipped_weighted_accuracy={clipped_weighted_accuracy:.1%}"
    )

    # --- 10. Confidence-weighted accuracy ---
    # Pairs with higher minimum confidence contribute more to the score.
    # This rewards the system for being correct when it is most certain,
    # and penalizes less for errors on low-confidence pairs.
    confidence_weighted_accuracy = (
        conf_weighted_correct_sum / conf_weighted_total_sum
        if conf_weighted_total_sum > 1e-9 else 0.0
    )
    logger.info(
        f"Confidence-weighted accuracy: {confidence_weighted_accuracy:.1%} "
        f"(conf_correct_sum={conf_weighted_correct_sum:.3f}, "
        f"conf_total_sum={conf_weighted_total_sum:.3f})"
    )

    # --- 11. Imbalance detection ---
    # Warn if one class pair contributes > 60% of all pairs (including ties)
    total_all_pairs = sum(pair_distribution.values())
    imbalance_warning = False
    if total_all_pairs > 0:
        for key, count in pair_distribution.items():
            if count / total_all_pairs > 0.60:
                imbalance_warning = True
                logger.warning(
                    f"Dataset imbalance: '{key}' contributes "
                    f"{count/total_all_pairs:.1%} of all pairs ({count}/{total_all_pairs}). "
                    f"Overall accuracy may be dominated by this pair."
                )
                break

    # --- 12. Bootstrap stability analysis ---
    stability = bootstrap_ranking_analysis(
        results,
        n_samples=100,
        ordered_labels=ordered_labels,
        label_rank=label_rank,
        comparable_pairs=comparable_pairs,
        score_range=score_range,
    )

    # --- 13. Interpretation ---
    interpretation = _build_interpretation(
        accuracy=overall_accuracy,
        mean_margin=normalized_mean_margin,
        imbalance_warning=imbalance_warning,
        confidence_weighted_accuracy=confidence_weighted_accuracy,
        stability=stability,
    )

    # --- Log summary ---
    logger.info("--- Pairwise Ranking Analysis ---")
    logger.info(
        f"  Accuracy:              {overall_accuracy:.1%}  "
        f"({total_correct}/{total_pairs} pairs, {total_tied} ties)"
    )
    logger.info(
        f"  Weighted accuracy:     {weighted_accuracy:.1%}  (raw margins)"
    )
    logger.info(
        f"  Norm. weighted acc.:   {normalized_weighted_accuracy:.1%}  (normalized)"
    )
    logger.info(
        f"  Clipped weighted acc.: {clipped_weighted_accuracy:.1%}  (clip={CLIP_THRESHOLD})"
    )
    logger.info(
        f"  Conf-weighted acc.:    {confidence_weighted_accuracy:.1%}"
    )
    logger.info(
        f"  Mean margin (raw):     {mean_margin:+.3f}  std={margin_std:.3f}"
    )
    logger.info(
        f"  Mean margin (norm.):   {normalized_mean_margin:+.3f}"
    )
    logger.info(
        f"  Bootstrap stability:   acc={stability.get('mean_accuracy', 0):.1%} "
        f"± {stability.get('std_accuracy', 0):.3f}"
    )
    logger.info(f"  Interpretation: {interpretation}")
    for key in sorted(per_class_accuracy):
        acc = per_class_accuracy[key]
        mg = normalized_per_class_margin.get(key, 0.0)
        n = per_class_total.get(key, 0)
        logger.info(f"  {key:>25}: acc={acc:.1%}  norm_margin={mg:+.3f}  ({n} pairs)")
    if imbalance_warning:
        logger.warning("  ⚠ Imbalance warning: results may be skewed by dominant class pair")

    return RankingAnalysis(
        accuracy=round(overall_accuracy, 4),
        correct_pairs=total_correct,
        total_pairs=total_pairs,
        tied_pairs=total_tied,
        per_class_accuracy={k: round(v, 4) for k, v in per_class_accuracy.items()},
        per_class_counts=per_class_total,
        mean_margin=round(mean_margin, 4),
        median_margin=round(median_margin, 4),
        margin_std=round(margin_std, 4),
        per_class_margin={k: round(v, 4) for k, v in per_class_margin.items()},
        weighted_accuracy=round(weighted_accuracy, 4),
        pair_distribution=pair_distribution,
        imbalance_warning=imbalance_warning,
        interpretation=interpretation,
        normalized_mean_margin=round(normalized_mean_margin, 4),
        normalized_per_class_margin={
            k: round(v, 4) for k, v in normalized_per_class_margin.items()
        },
        normalized_weighted_accuracy=round(normalized_weighted_accuracy, 4),
        clipped_weighted_accuracy=round(clipped_weighted_accuracy, 4),
        confidence_weighted_accuracy=round(confidence_weighted_accuracy, 4),
        stability=stability,
    )


# -----------------------------------------------------------------------
# Bootstrap stability analysis
# -----------------------------------------------------------------------

def bootstrap_ranking_analysis(
    results: list[EvalResult],
    n_samples: int = 100,
    ordered_labels: list[str] | None = None,
    label_rank: dict[str, int] | None = None,
    comparable_pairs: list[tuple[str, str]] | None = None,
    score_range: float | None = None,
) -> dict:
    """
    Estimate ranking stability via bootstrap resampling.

    Samples the valid result pool with replacement n_samples times, computing
    binary ranking accuracy and mean normalized margin each time. The resulting
    mean and std tell us how stable the ranking metrics are across dataset
    variations — a high std_accuracy suggests the dataset is too small or
    the class distributions too overlapping for reliable ranking conclusions.

    Args:
        results:          Full result list (valid + invalid).
        n_samples:        Number of bootstrap iterations (default 100).
        ordered_labels:   Pre-computed label ordering (avoids recomputing).
        label_rank:       Pre-computed label→rank dict (avoids recomputing).
        comparable_pairs: Pre-computed ordered pairs (avoids recomputing).
        score_range:      Pre-computed score range for normalization.

    Returns:
        dict with keys:
          mean_accuracy  — mean binary ranking accuracy across resamples
          std_accuracy   — standard deviation of accuracy (stability signal)
          mean_margin    — mean normalized margin across resamples
          std_margin     — std of normalized margin
          high_variance  — True if std_accuracy > 0.10 (interpret with caution)
    """
    from evaluation.analysis import validate_label_ordering

    valid = [r for r in results if r.is_valid]
    if len(valid) < 2:
        logger.warning("Bootstrap: fewer than 2 valid samples — skipping.")
        return {"mean_accuracy": 0.0, "std_accuracy": 0.0,
                "mean_margin": 0.0, "std_margin": 0.0, "high_variance": False}

    # Build ranking infrastructure if not pre-computed
    if ordered_labels is None:
        ordered_labels = validate_label_ordering(results)
    if len(ordered_labels) < 2:
        return {"mean_accuracy": 0.0, "std_accuracy": 0.0,
                "mean_margin": 0.0, "std_margin": 0.0, "high_variance": False}

    if label_rank is None:
        label_rank = {
            label: len(ordered_labels) - i
            for i, label in enumerate(ordered_labels)
        }
    if comparable_pairs is None:
        comparable_pairs = [
            (ordered_labels[i], ordered_labels[j])
            for i in range(len(ordered_labels))
            for j in range(i + 1, len(ordered_labels))
        ]

    if score_range is None:
        all_scores = np.array([r.overall_score for r in valid])
        score_range = float(all_scores.max() - all_scores.min())

    norm_epsilon = score_range if score_range > 1e-6 else 1e-6
    tie_margin = 0.01 * score_range if score_range > 1e-6 else 1e-4

    rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility
    valid_arr = np.array(valid, dtype=object)
    n_valid = len(valid_arr)

    boot_accuracies: list[float] = []
    boot_margins: list[float] = []

    for _ in range(n_samples):
        indices = rng.integers(0, n_valid, size=n_valid)
        sample = valid_arr[indices].tolist()
        acc, mean_mg = _compute_pair_metrics_simple(
            sample, label_rank, comparable_pairs, norm_epsilon, tie_margin
        )
        boot_accuracies.append(acc)
        boot_margins.append(mean_mg)

    mean_accuracy = float(np.mean(boot_accuracies))
    std_accuracy = float(np.std(boot_accuracies))
    mean_margin = float(np.mean(boot_margins))
    std_margin = float(np.std(boot_margins))
    high_variance = std_accuracy > 0.10

    if high_variance:
        logger.warning(
            f"Bootstrap: high variance in accuracy (std={std_accuracy:.3f} > 0.10). "
            f"Results may be unstable — collect more samples."
        )
    else:
        logger.info(
            f"Bootstrap stability (n={n_samples}): "
            f"acc={mean_accuracy:.1%} ± {std_accuracy:.3f}, "
            f"norm_margin={mean_margin:+.3f} ± {std_margin:.3f}"
        )

    return {
        "mean_accuracy": round(mean_accuracy, 4),
        "std_accuracy": round(std_accuracy, 4),
        "mean_margin": round(mean_margin, 4),
        "std_margin": round(std_margin, 4),
        "high_variance": high_variance,
    }


def _compute_pair_metrics_simple(
    valid: list[EvalResult],
    label_rank: dict[str, int],
    comparable_pairs: list[tuple[str, str]],
    norm_epsilon: float,
    tie_margin: float,
) -> tuple[float, float]:
    """
    Lightweight pair metric computation for bootstrap resampling.

    Returns:
        (accuracy, mean_normalized_margin) — the two most informative
        summary statistics for stability analysis.
    """
    by_label: dict[str, list[EvalResult]] = {}
    for r in valid:
        if r.label in label_rank:
            by_label.setdefault(r.label, []).append(r)

    total_correct = 0
    total_pairs = 0
    norm_margins: list[float] = []

    for label_high, label_low in comparable_pairs:
        highs = by_label.get(label_high, [])
        lows = by_label.get(label_low, [])
        for r_high in highs:
            for r_low in lows:
                margin = r_high.overall_score - r_low.overall_score
                if abs(margin) <= tie_margin:
                    continue
                norm_margin = margin / norm_epsilon
                total_pairs += 1
                norm_margins.append(norm_margin)
                if margin > 0:
                    total_correct += 1

    accuracy = total_correct / total_pairs if total_pairs > 0 else 0.0
    mean_margin = float(np.mean(norm_margins)) if norm_margins else 0.0
    return accuracy, mean_margin


# -----------------------------------------------------------------------
# Interpretation builder
# -----------------------------------------------------------------------

def _build_interpretation(
    accuracy: float,
    mean_margin: float,
    imbalance_warning: bool,
    confidence_weighted_accuracy: float = 0.0,
    stability: dict | None = None,
) -> str:
    """
    Build an interpretation string combining accuracy level, normalized margin
    strength, confidence-weighted accuracy, and bootstrap stability.

    Accuracy level:
      >= 0.80 → EXCELLENT
      >= 0.65 → GOOD
      >= 0.55 → MODERATE
      >= 0.45 → POOR
       < 0.45 → INVERTED

    Margin strength (based on mean normalized margin):
      >= 0.20 → strong separation
      >= 0.10 → moderate separation
       < 0.10 → weak separation (even if accuracy is high)

    Confidence-weighted accuracy:
      Reported if it differs substantially from binary accuracy (> 0.05 gap),
      which signals that confidence is (or is not) tracking correctness.

    Stability:
      high_variance → appends a warning to interpret results with caution.
    """
    if accuracy >= 0.80:
        acc_label = "EXCELLENT"
    elif accuracy >= 0.65:
        acc_label = "GOOD"
    elif accuracy >= 0.55:
        acc_label = "MODERATE"
    elif accuracy >= 0.45:
        acc_label = "POOR"
    else:
        acc_label = "INVERTED"

    # Margin thresholds apply to normalized margins
    if mean_margin >= 0.20:
        margin_label = "strong margins"
    elif mean_margin >= 0.10:
        margin_label = "moderate margins"
    elif mean_margin >= 0.0:
        margin_label = "weak margins → low ranking confidence"
    else:
        margin_label = "negative margins → system inverts ordering"

    parts = [f"{acc_label} — {accuracy:.0%} accuracy with {margin_label}"]

    # Confidence-weighted accuracy insight
    if confidence_weighted_accuracy > 0:
        gap = confidence_weighted_accuracy - accuracy
        if gap > 0.05:
            parts.append(
                f"confidence-weighted acc={confidence_weighted_accuracy:.0%} "
                f"(+{gap:.0%} vs binary — confidence tracks correctness well)"
            )
        elif gap < -0.05:
            parts.append(
                f"confidence-weighted acc={confidence_weighted_accuracy:.0%} "
                f"({gap:.0%} vs binary — confidence may anti-track correctness ⚠)"
            )

    # Bootstrap stability
    if stability:
        high_variance = stability.get("high_variance", False)
        std_acc = stability.get("std_accuracy", 0.0)
        if high_variance:
            parts.append(
                f"⚠ unstable across bootstrap resamples (std_acc={std_acc:.3f} > 0.10) "
                f"— collect more samples before drawing conclusions"
            )

    if imbalance_warning:
        parts.append("⚠ imbalanced dataset — interpret with caution")

    return "; ".join(parts)


# -----------------------------------------------------------------------
# Fallback for insufficient data
# -----------------------------------------------------------------------

def _empty_ranking() -> RankingAnalysis:
    return RankingAnalysis(
        accuracy=0.0,
        correct_pairs=0,
        total_pairs=0,
        tied_pairs=0,
        per_class_accuracy={},
        per_class_counts={},
        mean_margin=0.0,
        median_margin=0.0,
        margin_std=0.0,
        per_class_margin={},
        weighted_accuracy=0.0,
        pair_distribution={},
        imbalance_warning=False,
        interpretation="Insufficient data.",
        normalized_mean_margin=0.0,
        normalized_per_class_margin={},
        normalized_weighted_accuracy=0.0,
        clipped_weighted_accuracy=0.0,
        confidence_weighted_accuracy=0.0,
        stability={},
    )


# -----------------------------------------------------------------------
# Sample size estimation (unchanged from original)
# -----------------------------------------------------------------------

def minimum_samples_for_ranking(
    target_accuracy: float = 0.8,
    alpha: float = 0.05,
) -> dict:
    """
    Estimate minimum samples needed per class to detect a given ranking
    accuracy with statistical significance.

    Uses a normal approximation to the binomial.
    """
    from scipy.stats import norm as scipy_norm

    if target_accuracy <= 0.5:
        return {"note": "target_accuracy must be > 0.5"}

    z = scipy_norm.ppf(1 - alpha)
    n_pairs_needed = int(np.ceil((z * 0.5 / (target_accuracy - 0.5)) ** 2))
    n_per_class = int(np.ceil(np.sqrt(n_pairs_needed)))

    return {
        "n_pairs_needed": n_pairs_needed,
        "n_per_class_2_classes": n_per_class,
        "n_per_class_3_classes": n_per_class,
        "note": (
            f"To detect {target_accuracy:.0%} ranking accuracy at α={alpha}, "
            f"need ~{n_pairs_needed} pairs, i.e., ~{n_per_class} samples per class."
        ),
    }
