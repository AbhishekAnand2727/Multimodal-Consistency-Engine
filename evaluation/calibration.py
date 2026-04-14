"""
Calibration module: threshold suggestion, score normalization,
reliability adjustment, weight adjustment suggestions,
and evaluation report generation.

This module does NOT modify the core evaluator — it provides
external calibration recommendations based on observed data.

v2 additions:
  - suggest_weight_adjustments: data-driven weight rebalancing based on
    component discriminative power (separation scores)
  - Extended report: normalized margins, clipped accuracy,
    confidence-weighted accuracy, bootstrap stability, weight suggestions
"""

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from loguru import logger

from evaluation.dataset_loader import EvalResult
from evaluation.analysis import (
    LABELS,
    COMPONENTS,
    MIN_USEFUL_SEPARATION,
    score_distributions,
    component_breakdown,
    confidence_analysis,
    weakest_component,
    ranked_components,
    validate_label_ordering,
    overlap_analysis,
)


# -----------------------------------------------------------------------
# 1. Threshold suggestion — ordering-aware
# -----------------------------------------------------------------------

@dataclass
class ThresholdSuggestion:
    good_threshold: float
    acceptable_threshold: float
    method: str
    notes: str


def suggest_thresholds(results: list[EvalResult]) -> ThresholdSuggestion:
    """
    Suggest decision thresholds based on observed distributions.

    Strategy: midpoint between adjacent class medians, placed on the
    dynamically validated ordering (not the assumed good > acceptable > bad).
    This prevents incorrect threshold placement if class distributions
    are inverted due to data or scoring issues.

    Falls back to IQR boundaries if class medians overlap.
    """
    dists = score_distributions(results)

    # Use the validated ordering — may differ from the nominal LABELS order
    ordered = validate_label_ordering(results)

    if len(ordered) < 2:
        return ThresholdSuggestion(
            good_threshold=0.7,
            acceptable_threshold=0.4,
            method="default_fallback",
            notes="Fewer than 2 label classes have data; using defaults.",
        )

    # We need stats for adjacent pairs in sorted order
    stats_by_label = {label: dists[label] for label in ordered if dists[label] is not None}

    if len(stats_by_label) < 2:
        return ThresholdSuggestion(
            good_threshold=0.7,
            acceptable_threshold=0.4,
            method="default_fallback",
            notes="Insufficient data for threshold estimation; using defaults.",
        )

    # Build thresholds for each adjacent pair in the ordered list
    thresholds: list[float] = []
    for label_a, label_b in zip(ordered[:-1], ordered[1:]):
        sa = stats_by_label.get(label_a)
        sb = stats_by_label.get(label_b)
        if sa and sb:
            thresholds.append((sa.median + sb.median) / 2.0)

    if len(thresholds) < 1:
        return ThresholdSuggestion(
            good_threshold=0.7,
            acceptable_threshold=0.4,
            method="default_fallback",
            notes="Could not compute midpoint thresholds; using defaults.",
        )

    # Map computed thresholds to good/acceptable names
    # Highest threshold separates top class from second
    # Second threshold separates second class from third
    sorted_thresholds = sorted(thresholds, reverse=True)
    good_threshold = sorted_thresholds[0]
    acceptable_threshold = sorted_thresholds[1] if len(sorted_thresholds) > 1 else good_threshold * 0.6

    # Sanity: good_threshold must exceed acceptable_threshold
    if good_threshold <= acceptable_threshold:
        # Fall back to IQR boundaries
        top_stats = stats_by_label.get(ordered[0])
        mid_stats = stats_by_label.get(ordered[1]) if len(ordered) > 1 else None

        good_threshold = top_stats.q25 if top_stats else 0.7
        acceptable_threshold = mid_stats.q25 if mid_stats else 0.4

        return ThresholdSuggestion(
            good_threshold=round(good_threshold, 4),
            acceptable_threshold=round(acceptable_threshold, 4),
            method="iqr_boundary",
            notes=(
                "Medians too close to separate — thresholds placed at Q25 of upper classes. "
                "Collect more diverse samples to improve calibration."
            ),
        )

    return ThresholdSuggestion(
        good_threshold=round(good_threshold, 4),
        acceptable_threshold=round(acceptable_threshold, 4),
        method="median_midpoint",
        notes=(
            f"Thresholds placed at midpoints between adjacent class medians. "
            f"Label order used: {ordered}."
        ),
    )


# -----------------------------------------------------------------------
# 2. Score normalization
# -----------------------------------------------------------------------

def normalize_scores_minmax(
    results: list[EvalResult],
) -> list[tuple[EvalResult, float]]:
    """
    Apply min-max normalization to stretch observed score range to [0, 1].
    Returns list of (original_result, normalized_score) pairs.
    """
    valid = [r for r in results if r.is_valid]
    if not valid:
        return []

    scores = np.array([r.overall_score for r in valid])
    s_min, s_max = scores.min(), scores.max()

    if s_max - s_min < 1e-6:
        logger.warning("Score range too narrow for normalization.")
        return [(r, 0.5) for r in valid]

    normalized = [
        (r, round(float((r.overall_score - s_min) / (s_max - s_min)), 4))
        for r in valid
    ]
    logger.info(
        f"MinMax normalization: [{s_min:.4f}, {s_max:.4f}] → [0, 1] "
        f"({len(normalized)} samples)"
    )
    return normalized


def normalize_scores_sigmoid(
    results: list[EvalResult],
    center: float | None = None,
    scale: float = 10.0,
) -> list[tuple[EvalResult, float]]:
    """
    Apply sigmoid normalization centered at the median score.
    Compresses extremes, spreads the middle — good for skewed distributions.
    """
    valid = [r for r in results if r.is_valid]
    if not valid:
        return []

    scores = np.array([r.overall_score for r in valid])
    if center is None:
        center = float(np.median(scores))

    normalized = []
    for r in valid:
        x = (r.overall_score - center) * scale
        norm_score = 1.0 / (1.0 + np.exp(-x))
        normalized.append((r, round(float(norm_score), 4)))

    logger.info(f"Sigmoid normalization: center={center:.4f}, scale={scale}")
    return normalized


# -----------------------------------------------------------------------
# 3. Reliability adjustment — smooth (no hard None drops)
# -----------------------------------------------------------------------

@dataclass
class ReliabilityResult:
    video_path: str
    label: str
    raw_score: float
    confidence: float
    adjusted_score: float    # always a float — never None
    reliable: bool
    flag: str                # "ok" | "uncertain" | "low_confidence"


def apply_reliability_adjustment(
    results: list[EvalResult],
    confidence_threshold: float = 0.4,
) -> list[ReliabilityResult]:
    """
    Smooth confidence-weighted score adjustment.

    Formula:
        adjusted_score = confidence * raw_score + (1 - confidence) * 0.5

    This continuously pulls low-confidence scores toward the neutral
    midpoint (0.5) rather than making a hard binary decision.
    No samples are dropped — every result gets an adjusted_score.

    Flag categories:
        "ok"             — confidence >= threshold: score mostly unchanged
        "uncertain"      — confidence >= threshold/2: score noticeably pulled toward 0.5
        "low_confidence" — confidence < threshold/2: score pulled heavily toward 0.5

    The smooth formula ensures adjusted_score is always interpretable:
      - A confident bad dub: confidence=0.9, raw=0.1 → adjusted≈0.14
      - An uncertain result: confidence=0.2, raw=0.8 → adjusted=0.56
    """
    adjusted = []
    half_threshold = confidence_threshold * 0.5

    for r in results:
        if not r.is_valid:
            # Pipeline failed — treat as maximum uncertainty
            adjusted.append(ReliabilityResult(
                video_path=r.video_path,
                label=r.label,
                raw_score=0.0,
                confidence=0.0,
                adjusted_score=0.5,    # neutral — no information
                reliable=False,
                flag="low_confidence",
            ))
            continue

        conf = r.confidence
        raw = r.overall_score

        # Smooth blend toward neutral 0.5
        adj = conf * raw + (1.0 - conf) * 0.5

        if conf >= confidence_threshold:
            flag = "ok"
            reliable = True
        elif conf >= half_threshold:
            flag = "uncertain"
            reliable = False
        else:
            flag = "low_confidence"
            reliable = False

        adjusted.append(ReliabilityResult(
            video_path=r.video_path,
            label=r.label,
            raw_score=raw,
            confidence=conf,
            adjusted_score=round(adj, 4),
            reliable=reliable,
            flag=flag,
        ))

    n_ok = sum(1 for a in adjusted if a.flag == "ok")
    n_uncertain = sum(1 for a in adjusted if a.flag == "uncertain")
    n_low = sum(1 for a in adjusted if a.flag == "low_confidence")
    logger.info(
        f"Reliability adjustment: {n_ok} ok, {n_uncertain} uncertain, "
        f"{n_low} low_confidence (threshold={confidence_threshold})"
    )

    return adjusted


# -----------------------------------------------------------------------
# 4. Full evaluation report — expanded
# -----------------------------------------------------------------------

def generate_report(
    results: list[EvalResult],
    output_path: str | Path = "evaluation/results/report.txt",
    include_ranking: bool = True,
) -> str:
    """
    Generate a comprehensive human-readable calibration report.

    Sections:
      - Dataset summary
      - Score distributions (with P10-P90)
      - Label ordering validation
      - Recommended thresholds
      - Separation score (good vs bad)
      - Component ranking by discriminative power
      - Confidence behavior (dual correlation)
      - Reliability breakdown
      - Ranking accuracy (if include_ranking=True)
      - Actionable recommendations with severity warnings
    """
    valid = [r for r in results if r.is_valid]
    failed = [r for r in results if not r.is_valid]

    dists = score_distributions(results)
    ordered_labels = validate_label_ordering(results)
    thresholds = suggest_thresholds(results)
    conf = confidence_analysis(results)
    comp_ranks = ranked_components(results)
    weakest = weakest_component(results)
    breakdown = component_breakdown(results)
    reliability = apply_reliability_adjustment(results)
    overlaps = overlap_analysis(results)

    # Separation: top class mean vs bottom class mean
    top_label = ordered_labels[0] if ordered_labels else "good"
    bot_label = ordered_labels[-1] if len(ordered_labels) > 1 else "bad"
    top_stats = dists.get(top_label)
    bot_stats = dists.get(bot_label)
    overall_separation = (
        round(top_stats.mean - bot_stats.mean, 4)
        if top_stats and bot_stats else None
    )

    lines = []
    lines.append("=" * 70)
    lines.append("  MULTIMODAL DUBBING EVALUATOR — CALIBRATION REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # Dataset summary
    lines.append("")
    lines.append("DATASET SUMMARY")
    lines.append(f"  Total:   {len(results)}  |  Valid: {len(valid)}  |  Failed: {len(failed)}")
    for label in LABELS:
        n = sum(1 for r in valid if r.label == label)
        lines.append(f"  {label:>12}: {n} samples")

    # Label ordering
    lines.append("")
    lines.append("LABEL ORDERING (by observed median)")
    ordering_matches_expected = ordered_labels == [l for l in LABELS if dists[l] is not None]
    status = "OK" if ordering_matches_expected else "WARNING: INVERTED"
    lines.append(f"  Status: {status}")
    lines.append(f"  Order:  {' > '.join(ordered_labels)}")
    for label in ordered_labels:
        s = dists.get(label)
        if s:
            lines.append(f"  {label:>12}: median={s.median:.3f}  P10-P90={s.robust_range_str()}")

    # Score distributions
    lines.append("")
    lines.append("SCORE DISTRIBUTIONS")
    for label in LABELS:
        s = dists.get(label)
        if s:
            lines.append(
                f"  {label:>12}:  mean={s.mean:.3f} ± {s.std:.3f}  "
                f"median={s.median:.3f}  IQR=[{s.q25:.3f}–{s.q75:.3f}]  "
                f"P10-P90={s.robust_range_str()}"
            )
        else:
            lines.append(f"  {label:>12}:  no data")

    # Separation
    lines.append("")
    lines.append("CLASS SEPARATION")
    if overall_separation is not None:
        flag = ""
        if overall_separation < MIN_USEFUL_SEPARATION:
            flag = " ⚠ POOR"
        elif overall_separation < 0.30:
            flag = " △ MODERATE"
        else:
            flag = " ✓ GOOD"
        lines.append(
            f"  {top_label} vs {bot_label}: separation = {overall_separation:+.3f}{flag}"
        )
    for o in overlaps:
        lines.append(
            f"  {o.label_a}/{o.label_b}: "
            f"overlap={o.overlap_width:.3f} [{o.overlap_low:.3f}–{o.overlap_high:.3f}]  "
            f"separation={o.separation:+.3f}  severity={o.severity}"
        )

    # Thresholds
    lines.append("")
    lines.append("RECOMMENDED THRESHOLDS")
    lines.append(f"  Good threshold:       >= {thresholds.good_threshold}")
    lines.append(f"  Acceptable threshold: >= {thresholds.acceptable_threshold}")
    lines.append(f"  Method: {thresholds.method}")
    lines.append(f"  Notes:  {thresholds.notes}")

    # Component ranking
    lines.append("")
    lines.append("COMPONENT DISCRIMINATIVE POWER (good – bad separation)")
    lines.append(f"  {'Rank':<6} {'Component':<10} {'Separation':>12}  {'Strength'}")
    for rank, (comp, sep) in enumerate(comp_ranks, 1):
        if sep >= 0.20:
            strength = "strong"
        elif sep >= 0.10:
            strength = "moderate"
        elif sep >= 0.0:
            strength = "weak"
        else:
            strength = "INVERTED ⚠"
        lines.append(f"  #{rank:<5} {comp:<10} {sep:>+12.3f}  {strength}")
    if weakest:
        lines.append(f"  Weakest: {weakest}")

    lines.append("")
    lines.append("COMPONENT MEANS BY LABEL")
    header = f"  {'':>12}" + "".join(f"  {c:>8}" for c in COMPONENTS)
    lines.append(header)
    for label in LABELS:
        row = f"  {label:>12}"
        for comp in COMPONENTS:
            s = breakdown[label].get(comp)
            row += f"  {s.mean:8.3f}" if s else f"  {'N/A':>8}"
        lines.append(row)

    # Confidence
    lines.append("")
    lines.append("CONFIDENCE BEHAVIOR")
    for label, mc in conf.mean_confidence_by_label.items():
        lines.append(f"  {label:>12}: mean_conf = {mc:.3f}")
    lines.append(f"  Low-confidence samples (<{conf.low_confidence_threshold}): {conf.low_confidence_count}")
    if conf.low_conf_mean_score is not None:
        lines.append(f"  Mean score (low-conf):  {conf.low_conf_mean_score:.3f}")
    if conf.high_conf_mean_score is not None:
        lines.append(f"  Mean score (high-conf): {conf.high_conf_mean_score:.3f}")

    lines.append("")
    lines.append("  Correlation Analysis:")
    if conf.conf_score_correlation is not None:
        lines.append(f"  conf vs score: {conf.conf_score_correlation:+.3f}")
    if conf.conf_error_correlation is not None:
        lines.append(f"  conf vs error: {conf.conf_error_correlation:+.3f}")
        if conf.conf_error_correlation < -0.3:
            lines.append("    → GOOD: high confidence reliably predicts lower error")
        elif conf.conf_error_correlation < 0.0:
            lines.append("    → WEAK: mild tendency for high confidence to reduce error")
        else:
            lines.append("    → POOR: confidence does not track error — needs recalibration ⚠")

    # Ranking accuracy
    ranking = None
    if include_ranking:
        try:
            from evaluation.ranking import pairwise_ranking_analysis
            ranking = pairwise_ranking_analysis(results)
            lines.append("")
            lines.append("RANKING ACCURACY")
            lines.append(f"  {ranking.interpretation}")
            lines.append("")
            lines.append(
                f"  Binary accuracy:        {ranking.accuracy:.1%}  "
                f"({ranking.correct_pairs}/{ranking.total_pairs} pairs, "
                f"{ranking.tied_pairs} ties)"
            )
            lines.append(
                f"  Weighted accuracy:      {ranking.weighted_accuracy:.1%}  "
                f"(raw margin-magnitude weighted)"
            )
            lines.append(
                f"  Norm. weighted acc.:    {ranking.normalized_weighted_accuracy:.1%}  "
                f"(scale-invariant, normalized margins)"
            )
            lines.append(
                f"  Clipped weighted acc.:  {ranking.clipped_weighted_accuracy:.1%}  "
                f"(outlier-robust, clip=0.3)"
            )
            lines.append(
                f"  Conf-weighted acc.:     {ranking.confidence_weighted_accuracy:.1%}  "
                f"(weighted by min(conf_i, conf_j))"
            )
            lines.append("")
            lines.append(
                f"  Mean margin (raw):      {ranking.mean_margin:+.3f}  "
                f"(median={ranking.median_margin:+.3f}, std={ranking.margin_std:.3f})"
            )
            lines.append(
                f"  Mean margin (norm.):    {ranking.normalized_mean_margin:+.3f}  "
                f"(scale-invariant)"
            )
            # Margin strength label based on normalized margin
            norm_mg = ranking.normalized_mean_margin
            if norm_mg >= 0.20:
                lines.append("  Margin strength: STRONG (> 0.20)")
            elif norm_mg >= 0.10:
                lines.append("  Margin strength: MODERATE (0.10 – 0.20)")
            else:
                lines.append("  Margin strength: WEAK (< 0.10) ⚠")
            lines.append("")
            lines.append(
                f"  {'Class pair':<25}  {'Accuracy':>8}  "
                f"{'Norm margin':>12}  {'N pairs':>8}"
            )
            for label_pair in sorted(ranking.per_class_accuracy):
                acc = ranking.per_class_accuracy[label_pair]
                mg = ranking.normalized_per_class_margin.get(label_pair, 0.0)
                n = ranking.per_class_counts.get(label_pair, 0)
                lines.append(
                    f"  {label_pair:<25}  {acc:>8.1%}  {mg:>+12.3f}  {n:>8}"
                )

            # Bootstrap stability
            stab = ranking.stability
            if stab:
                lines.append("")
                lines.append("  BOOTSTRAP STABILITY (n=100 resamples)")
                lines.append(
                    f"  Accuracy:  {stab.get('mean_accuracy', 0):.1%} "
                    f"± {stab.get('std_accuracy', 0):.3f}"
                )
                lines.append(
                    f"  Margin:    {stab.get('mean_margin', 0):+.3f} "
                    f"± {stab.get('std_margin', 0):.3f}"
                )
                if stab.get("high_variance", False):
                    lines.append(
                        "  ⚠ HIGH VARIANCE: std_accuracy > 0.10 — results unstable, "
                        "collect more labeled samples."
                    )
                else:
                    lines.append("  Stability: OK")

            if ranking.imbalance_warning:
                lines.append("")
                lines.append(
                    "  ⚠ IMBALANCE WARNING: one class pair dominates (>60% of pairs). "
                    "Overall accuracy may not reflect balanced performance."
                )
                dist_str = "  Pair distribution: " + "  ".join(
                    f"{k}: {v}" for k, v in ranking.pair_distribution.items()
                )
                lines.append(dist_str)
        except ImportError:
            lines.append("")
            lines.append("RANKING ACCURACY: (ranking.py not available)")

    # Weight adjustment suggestions (based on component separation)
    lines.append("")
    lines.append("WEIGHT ADJUSTMENT SUGGESTIONS")
    try:
        # Build component_separation from ranked_components output
        comp_sep_dict = dict(comp_ranks)
        adjusted_weights = suggest_weight_adjustments(comp_sep_dict)
        lines.append(
            "  Data-driven weight suggestions (50% separation-guided, 50% default priors):"
        )
        lines.append(
            f"  {'Component':<10}  {'Default':>8}  {'Suggested':>10}  {'Separation':>12}"
        )
        for comp in sorted(adjusted_weights):
            default_w = _DEFAULT_WEIGHTS.get(comp, 0.0)
            new_w = adjusted_weights[comp]
            sep = comp_sep_dict.get(comp, 0.0)
            delta = new_w - default_w
            arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "~")
            lines.append(
                f"  {comp:<10}  {default_w:>8.2f}  {new_w:>10.4f}  "
                f"{sep:>+12.3f}  {arrow}"
            )
        lines.append(
            "  Note: apply these suggestions only after validating on a larger dataset."
        )
    except Exception as e:
        lines.append(f"  (Weight suggestion failed: {e})")

    # Reliability
    n_ok = sum(1 for r in reliability if r.flag == "ok")
    n_uncertain = sum(1 for r in reliability if r.flag == "uncertain")
    n_low = sum(1 for r in reliability if r.flag == "low_confidence")
    lines.append("")
    lines.append("RELIABILITY BREAKDOWN (smooth adjustment)")
    lines.append(f"  OK:             {n_ok}")
    lines.append(f"  Uncertain:      {n_uncertain}")
    lines.append(f"  Low-confidence: {n_low}")
    lines.append(f"  (All adjusted toward 0.5 by: adj = conf*score + (1-conf)*0.5)")

    # Recommendations
    lines.append("")
    lines.append("RECOMMENDATIONS")
    recs = _generate_recommendations(
        dists, thresholds, weakest, conf,
        comp_ranks, overall_separation, overlaps
    )
    for i, rec in enumerate(recs, 1):
        lines.append(f"  {i}. {rec}")

    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {output_path}")

    return report


# -----------------------------------------------------------------------
# 5. Calibration feedback — weight adjustment suggestions
# -----------------------------------------------------------------------

# Default scoring weights from CLAUDE.md
_DEFAULT_WEIGHTS = {
    "gender":  0.35,
    "age":     0.30,
    "pitch":   0.20,
    "formant": 0.15,
}


def suggest_weight_adjustments(
    component_separation: dict[str, float],
    blend_ratio: float = 0.5,
) -> dict[str, float]:
    """
    Suggest revised scoring weights based on observed component separation.

    Components with high discriminative power (large good-vs-bad separation)
    are rewarded with higher weight; components with low (or negative)
    separation are reduced.

    Strategy:
      1. Floor negative separations to a small positive value (1e-3) —
         a component that is inverted should still keep *some* weight rather
         than being zeroed, because the underlying model may improve.
      2. Compute separation-proportional weights (floored_sep / total_sep).
      3. Blend 50/50 with the default weights to preserve domain priors
         and avoid over-reacting to noisy estimates.
      4. Re-normalize so all weights sum to 1.0.

    Args:
        component_separation: {component_name: separation_score} — typically
            the good-vs-bad mean difference for each scoring component.
            Positive = component discriminates correctly; negative = inverted.
        blend_ratio: Weight of separation-based assignment vs default weights.
            0.0 = pure default weights; 1.0 = pure separation-based.
            Default 0.5 gives equal influence to data and priors.

    Returns:
        dict[str, float] — suggested weights summing to 1.0.

    Example:
        >>> suggest_weight_adjustments({"gender": 0.40, "age": 0.05,
        ...                             "pitch": 0.20, "formant": -0.02})
        {'gender': 0.37, 'age': 0.22, 'pitch': 0.26, 'formant': 0.15}
    """
    if not component_separation:
        logger.warning("suggest_weight_adjustments: empty input — returning defaults.")
        return dict(_DEFAULT_WEIGHTS)

    known_components = list(component_separation.keys())

    # 1. Floor negative separations to prevent zero/negative weights
    floored = {c: max(s, 1e-3) for c, s in component_separation.items()}
    total_sep = sum(floored.values())
    if total_sep < 1e-9:
        logger.warning("suggest_weight_adjustments: all separations near zero — returning defaults.")
        return dict(_DEFAULT_WEIGHTS)

    # 2. Separation-proportional weights
    sep_weights = {c: v / total_sep for c, v in floored.items()}

    # 3. Blend with default weights
    blended = {}
    for c in known_components:
        default_w = _DEFAULT_WEIGHTS.get(c, 1.0 / len(known_components))
        blended[c] = (1.0 - blend_ratio) * default_w + blend_ratio * sep_weights[c]

    # 4. Re-normalize to sum exactly to 1.0
    total_blended = sum(blended.values())
    adjusted = {c: round(v / total_blended, 4) for c, v in blended.items()}

    # Log comparison vs defaults
    logger.info("Weight adjustment suggestions (separation-guided):")
    for c in sorted(adjusted):
        default_w = _DEFAULT_WEIGHTS.get(c, 0.0)
        sep = component_separation.get(c, 0.0)
        delta = adjusted[c] - default_w
        direction = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "~")
        logger.info(
            f"  {c:>10}: default={default_w:.2f}  sep={sep:+.3f}  "
            f"suggested={adjusted[c]:.4f}  ({direction} {delta:+.4f})"
        )

    return adjusted


def _generate_recommendations(
    dists, thresholds, weakest, conf,
    comp_ranks, separation, overlaps
) -> list[str]:
    """Generate prioritized actionable recommendations."""
    recs = []

    # Separation check
    if separation is not None:
        if separation < MIN_USEFUL_SEPARATION:
            recs.append(
                f"[CRITICAL] Overall separation ({separation:.3f}) is below {MIN_USEFUL_SEPARATION}. "
                f"Scores are nearly indistinguishable between good and bad dubs. "
                f"The scoring formula needs significant recalibration."
            )
        elif separation < 0.30:
            recs.append(
                f"[WARNING] Separation ({separation:.3f}) is moderate. "
                f"Add more diverse labeled samples and review scoring weights."
            )

    # Severe overlap
    severe = [o for o in overlaps if o.severity == "severe"]
    if severe:
        pairs = ", ".join(f"{o.label_a}/{o.label_b}" for o in severe)
        recs.append(
            f"[WARNING] Severe overlap detected between: {pairs}. "
            f"Threshold placement is unreliable in these zones."
        )

    # Inverted components
    inverted = [(c, s) for c, s in comp_ranks if s < 0]
    if inverted:
        names = ", ".join(c for c, _ in inverted)
        recs.append(
            f"[WARNING] Component(s) with inverted separation (bad > good): {names}. "
            f"These components are actively hurting score quality — "
            f"consider zeroing their weight until the underlying model is improved."
        )

    # Weakest component
    if weakest:
        recs.append(
            f"Component '{weakest}' has the lowest discriminative power. "
            f"Investigate the underlying model or reduce its scoring weight."
        )

    # Confidence-error correlation
    if conf.conf_error_correlation is not None and conf.conf_error_correlation >= 0.0:
        recs.append(
            "Confidence does not track prediction error. "
            "Recalibrate the confidence formula — consider increasing weight "
            "of feature_availability and reducing weight of speech_ratio."
        )

    # Low confidence prevalence
    total_valid = sum(1 for label in conf.mean_confidence_by_label if conf.mean_confidence_by_label[label] > 0)
    if conf.low_confidence_count > max(total_valid, 1) * 2:
        recs.append(
            "High proportion of low-confidence outputs. "
            "Check video quality, face detection coverage, and minimum audio length."
        )

    # Threshold method quality
    if thresholds.method == "iqr_boundary":
        recs.append(
            "Threshold estimation fell back to IQR boundaries due to overlapping medians. "
            "Collect more labeled samples with clear quality distinctions."
        )

    if not recs:
        recs.append("System appears well-calibrated for the current dataset.")

    return recs
