"""
Statistical analysis of evaluation results.

Five analysis passes:
  1. Score distributions per label (mean, std, min, max, quartiles, P10/P90)
  2. Overlap analysis — P10-P90 range with separation score
  3. Component breakdown — separation-ranked discriminative power per signal
  4. Confidence analysis — dual correlation (vs score AND vs proxy error)
  5. Label ordering validation — detects inverted distributions early
"""

import numpy as np
from dataclasses import dataclass, field
from loguru import logger

from evaluation.dataset_loader import EvalResult


LABELS = ["good", "acceptable", "bad"]
COMPONENTS = ["gender", "age", "pitch", "formant"]

# Expected score midpoints per label — used as proxy targets for error analysis.
# These reflect what a well-calibrated system should output on average.
LABEL_EXPECTED_SCORE: dict[str, float] = {
    "good": 0.85,
    "acceptable": 0.60,
    "bad": 0.25,
}

# Minimum separation (good_mean - bad_mean) for the system to be considered useful.
MIN_USEFUL_SEPARATION = 0.15


# -----------------------------------------------------------------------
# Core stats dataclass — extended with P10/P90
# -----------------------------------------------------------------------

@dataclass
class DistStats:
    """Descriptive statistics for a score distribution."""
    mean: float
    std: float
    min: float
    max: float
    p10: float      # 10th percentile — robust lower bound
    q25: float
    median: float
    q75: float
    p90: float      # 90th percentile — robust upper bound
    n: int

    def range_str(self) -> str:
        return f"[{self.min:.3f} – {self.max:.3f}]"

    def robust_range_str(self) -> str:
        """P10–P90 range, less sensitive to outliers than full min/max."""
        return f"[{self.p10:.3f} – {self.p90:.3f}]"


def _compute_stats(values: list[float]) -> DistStats | None:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return DistStats(
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        p10=float(np.percentile(arr, 10)),
        q25=float(np.percentile(arr, 25)),
        median=float(np.median(arr)),
        q75=float(np.percentile(arr, 75)),
        p90=float(np.percentile(arr, 90)),
        n=len(arr),
    )


def _group_by_label(results: list[EvalResult]) -> dict[str, list[EvalResult]]:
    groups: dict[str, list[EvalResult]] = {label: [] for label in LABELS}
    for r in results:
        if r.is_valid and r.label in groups:
            groups[r.label].append(r)
    return groups


# -----------------------------------------------------------------------
# 1. Score distributions per label
# -----------------------------------------------------------------------

def score_distributions(results: list[EvalResult]) -> dict[str, DistStats | None]:
    """
    Compute descriptive statistics of overall_score for each label.
    Returns {label: DistStats}.
    """
    groups = _group_by_label(results)
    dists = {}
    for label in LABELS:
        scores = [r.overall_score for r in groups[label]]
        dists[label] = _compute_stats(scores)
    return dists


# -----------------------------------------------------------------------
# Label ordering validation
# -----------------------------------------------------------------------

def validate_label_ordering(
    results: list[EvalResult],
) -> list[str]:
    """
    Validate that label medians follow the expected ordering: good > acceptable > bad.

    If ordering is violated (e.g., 'bad' scores higher than 'acceptable'),
    log a warning and return labels sorted by observed median (highest first).
    This sorted order is used by overlap and threshold logic to stay correct
    regardless of scoring quirks.

    Returns:
        List of labels ordered by descending median score.
        Always contains all LABELS that have data.
    """
    dists = score_distributions(results)

    # Collect labels that have data, sorted by median descending
    labeled_medians = [
        (label, dists[label].median)
        for label in LABELS
        if dists[label] is not None
    ]
    labeled_medians.sort(key=lambda x: x[1], reverse=True)
    observed_order = [label for label, _ in labeled_medians]

    # Expected order (subset of LABELS with data)
    expected_order = [l for l in LABELS if dists[l] is not None]

    if observed_order != expected_order:
        logger.warning(
            f"Label ordering VIOLATED. "
            f"Expected: {expected_order}, "
            f"Observed by median: {observed_order}. "
            f"Medians: { {l: round(m, 3) for l, m in labeled_medians} }. "
            f"Using observed order for thresholding."
        )
    else:
        logger.info(
            f"Label ordering OK: {observed_order} "
            f"Medians: { {l: round(m, 3) for l, m in labeled_medians} }"
        )

    return observed_order


# -----------------------------------------------------------------------
# 2. Overlap analysis — upgraded to P10-P90 with separation score
# -----------------------------------------------------------------------

@dataclass
class OverlapInfo:
    """Overlap between two adjacent label classes."""
    label_a: str          # higher-quality label (higher expected score)
    label_b: str          # lower-quality label
    overlap_low: float    # start of overlap zone
    overlap_high: float   # end of overlap zone
    overlap_width: float
    separation: float     # mean_a - mean_b (positive = system separates them)
    severity: str         # "none", "mild", "significant", "severe"


def overlap_analysis(results: list[EvalResult]) -> list[OverlapInfo]:
    """
    Compute overlap between adjacent class pairs using the P10-P90 range.

    P10-P90 is wider than IQR (Q25-Q75), capturing 80% of the distribution
    while still being robust against extreme outliers. This gives a more
    realistic picture of ambiguity zones than IQR alone.

    Also computes a separation score (mean difference) to quantify how well
    the system distinguishes adjacent classes regardless of overlap.

    Pair ordering is determined dynamically by `validate_label_ordering`
    so this function remains correct even if score distributions are inverted.
    """
    dists = score_distributions(results)
    ordered_labels = validate_label_ordering(results)

    # Build adjacent pairs from the dynamically sorted order
    pairs = list(zip(ordered_labels[:-1], ordered_labels[1:]))

    overlaps = []

    for label_a, label_b in pairs:
        stats_a = dists.get(label_a)
        stats_b = dists.get(label_b)

        if stats_a is None or stats_b is None:
            logger.warning(f"Cannot compute overlap for {label_a}/{label_b}: missing data")
            continue

        # Use P10-P90 for robust overlap detection
        a_low, a_high = stats_a.p10, stats_a.p90
        b_low, b_high = stats_b.p10, stats_b.p90

        overlap_low = max(a_low, b_low)
        overlap_high = min(a_high, b_high)
        overlap_width = max(0.0, overlap_high - overlap_low)

        # Separation: how far apart are the means?
        # Positive means label_a (higher quality) has higher mean — correct.
        separation = stats_a.mean - stats_b.mean

        # Severity: overlap relative to the narrower class's P10-P90 spread
        spread_a = a_high - a_low
        spread_b = b_high - b_low
        reference_spread = min(spread_a, spread_b)  # more conservative than avg

        if reference_spread > 0:
            overlap_ratio = overlap_width / reference_spread
        else:
            overlap_ratio = 0.0

        if overlap_ratio < 0.05:
            severity = "none"
        elif overlap_ratio < 0.20:
            severity = "mild"
        elif overlap_ratio < 0.45:
            severity = "significant"
        else:
            severity = "severe"

        overlaps.append(OverlapInfo(
            label_a=label_a,
            label_b=label_b,
            overlap_low=round(overlap_low, 4),
            overlap_high=round(overlap_high, 4),
            overlap_width=round(overlap_width, 4),
            separation=round(separation, 4),
            severity=severity,
        ))

    return overlaps


# -----------------------------------------------------------------------
# 3. Component breakdown — with separation-based ranking
# -----------------------------------------------------------------------

def component_breakdown(
    results: list[EvalResult],
) -> dict[str, dict[str, DistStats | None]]:
    """
    For each label, compute stats of each component score.
    Returns {label: {component: DistStats}}.
    """
    groups = _group_by_label(results)
    breakdown = {}

    for label in LABELS:
        breakdown[label] = {}
        for comp in COMPONENTS:
            values = [
                r.components[comp]
                for r in groups[label]
                if comp in r.components
            ]
            breakdown[label][comp] = _compute_stats(values)

    return breakdown


def component_separations(results: list[EvalResult]) -> dict[str, float]:
    """
    Compute discriminative power of each component as:
        separation = mean_good - mean_bad

    A high separation means the component clearly differs between
    good and bad dubs — it's contributing meaningfully to scoring.
    A low separation means the component produces similar scores
    regardless of dub quality — it's not discriminating.

    Returns:
        {component: separation_score}, sorted descending by separation.
    """
    breakdown = component_breakdown(results)
    good_stats = breakdown.get("good", {})
    bad_stats = breakdown.get("bad", {})

    separations = {}
    for comp in COMPONENTS:
        gs = good_stats.get(comp)
        bs = bad_stats.get(comp)
        if gs is not None and bs is not None:
            separations[comp] = round(gs.mean - bs.mean, 4)
        else:
            separations[comp] = None

    # Log ranked components
    ranked = sorted(
        [(c, s) for c, s in separations.items() if s is not None],
        key=lambda x: x[1], reverse=True
    )
    logger.info("Component discriminative power (good-bad separation):")
    for comp, sep in ranked:
        bar = "█" * int(max(0, sep) * 20)
        logger.info(f"  {comp:>8}: {sep:+.3f}  {bar}")

    return separations


def weakest_component(results: list[EvalResult]) -> str | None:
    """
    Return the component with the smallest (good - bad) separation.
    Uses mean separation, not variance, as a direct measure of
    discriminative usefulness.
    """
    seps = component_separations(results)
    valid = {c: s for c, s in seps.items() if s is not None}
    if not valid:
        return None
    return min(valid, key=lambda c: valid[c])


def strongest_component(results: list[EvalResult]) -> str | None:
    """Return the component with the largest (good - bad) separation."""
    seps = component_separations(results)
    valid = {c: s for c, s in seps.items() if s is not None}
    if not valid:
        return None
    return max(valid, key=lambda c: valid[c])


def ranked_components(results: list[EvalResult]) -> list[tuple[str, float]]:
    """
    Return all components sorted by discriminative power, descending.
    Returns list of (component_name, separation) tuples.
    """
    seps = component_separations(results)
    ranked = [
        (comp, seps[comp])
        for comp in COMPONENTS
        if seps.get(comp) is not None
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# -----------------------------------------------------------------------
# 4. Confidence analysis — dual correlation (score + error)
# -----------------------------------------------------------------------

@dataclass
class ConfidenceAnalysis:
    """Summary of how confidence relates to score reliability."""
    mean_confidence_by_label: dict[str, float]
    low_confidence_count: int
    low_confidence_threshold: float
    low_conf_mean_score: float | None
    high_conf_mean_score: float | None
    # Correlation between confidence and overall_score
    # High positive = confident samples score higher
    conf_score_correlation: float | None
    # Correlation between confidence and proxy error |score - expected|
    # Negative = confident samples have lower error — desired behavior
    conf_error_correlation: float | None

    # Keep old name as alias for backward compatibility
    @property
    def correlation(self) -> float | None:
        return self.conf_score_correlation


def _proxy_error(result: EvalResult) -> float:
    """
    Compute proxy error for a result: how far the raw compatibility score
    deviates from the expected score for that label.

    Uses raw_score (pre-confidence) to avoid bias — overall_score is
    scaled by confidence, which would conflate reliability with accuracy.

    Expected scores (from LABEL_EXPECTED_SCORE) encode our prior
    on what a well-calibrated system should output per class.
    A well-behaved confidence signal should correlate negatively
    with this error — high confidence → low error.
    """
    expected = LABEL_EXPECTED_SCORE.get(result.label)
    if expected is None:
        return 0.0
    # Prefer stored raw_score; fall back to recovering it from overall_score
    if result.raw_score is not None:
        raw = result.raw_score
    elif result.overall_score is not None and result.confidence is not None:
        raw = result.overall_score / max(result.confidence, 1e-6)
    else:
        return 0.0
    return abs(raw - expected)


def confidence_analysis(
    results: list[EvalResult],
    low_threshold: float = 0.4,
) -> ConfidenceAnalysis:
    """
    Analyze the relationship between confidence and score reliability.

    Two correlation axes:
      1. confidence vs overall_score
         Positive correlation means high confidence → higher scores.
         This is expected if confident detections pick up good dubs.

      2. confidence vs proxy_error (|score - expected_label_score|)
         Negative correlation is the ideal: high confidence → small error.
         If this is near zero or positive, confidence is not a useful
         reliability signal and the confidence formula needs work.
    """
    valid = [r for r in results if r.is_valid]
    if not valid:
        return ConfidenceAnalysis(
            mean_confidence_by_label={},
            low_confidence_count=0,
            low_confidence_threshold=low_threshold,
            low_conf_mean_score=None,
            high_conf_mean_score=None,
            conf_score_correlation=None,
            conf_error_correlation=None,
        )

    groups = _group_by_label(results)
    mean_conf_by_label = {
        label: float(np.mean([r.confidence for r in groups[label]]))
        if groups[label] else 0.0
        for label in LABELS
    }

    low_conf = [r for r in valid if r.confidence < low_threshold]
    high_conf = [r for r in valid if r.confidence >= low_threshold]

    low_conf_score = (
        float(np.mean([r.overall_score for r in low_conf])) if low_conf else None
    )
    high_conf_score = (
        float(np.mean([r.overall_score for r in high_conf])) if high_conf else None
    )

    conf_score_corr = None
    conf_error_corr = None

    if len(valid) >= 3:
        scores = np.array([r.overall_score for r in valid])
        confs = np.array([r.confidence for r in valid])
        errors = np.array([_proxy_error(r) for r in valid])

        # Guard against zero variance (all identical values)
        if confs.std() > 1e-6:
            conf_score_corr = float(np.corrcoef(confs, scores)[0, 1])
            conf_error_corr = float(np.corrcoef(confs, errors)[0, 1])

    # Log interpretation
    if conf_error_corr is not None:
        if conf_error_corr < -0.3:
            interpretation = "GOOD — high confidence reliably predicts lower error"
        elif conf_error_corr < 0.0:
            interpretation = "WEAK — mild tendency for high confidence to reduce error"
        else:
            interpretation = "POOR — confidence does not reduce error (needs recalibration)"
        logger.info(
            f"Confidence-error correlation: {conf_error_corr:.3f} ({interpretation})"
        )

    return ConfidenceAnalysis(
        mean_confidence_by_label=mean_conf_by_label,
        low_confidence_count=len(low_conf),
        low_confidence_threshold=low_threshold,
        low_conf_mean_score=low_conf_score,
        high_conf_mean_score=high_conf_score,
        conf_score_correlation=conf_score_corr,
        conf_error_correlation=conf_error_corr,
    )


# -----------------------------------------------------------------------
# Full analysis runner
# -----------------------------------------------------------------------

def run_full_analysis(results: list[EvalResult]) -> dict:
    """
    Run all analysis passes and return a structured summary dict.
    Logs key findings at each stage.
    """
    valid = [r for r in results if r.is_valid]
    failed = [r for r in results if not r.is_valid]

    logger.info(f"Analysis: {len(valid)} valid, {len(failed)} failed results")

    # 0. Label ordering check
    ordered = validate_label_ordering(results)

    # 1. Score distributions
    dists = score_distributions(results)
    logger.info("--- Score Distributions ---")
    for label in LABELS:
        s = dists[label]
        if s:
            logger.info(
                f"  {label:>12}: mean={s.mean:.3f} ± {s.std:.3f}  "
                f"median={s.median:.3f}  P10-P90={s.robust_range_str()}  n={s.n}"
            )
        else:
            logger.info(f"  {label:>12}: no data")

    # 2. Overlap
    overlaps = overlap_analysis(results)
    logger.info("--- Overlap Analysis (P10-P90) ---")
    for o in overlaps:
        logger.info(
            f"  {o.label_a}/{o.label_b}: "
            f"overlap={o.overlap_width:.3f} [{o.overlap_low:.3f}–{o.overlap_high:.3f}]  "
            f"separation={o.separation:+.3f}  severity={o.severity}"
        )

    # 3. Component breakdown + ranking
    breakdown = component_breakdown(results)
    comp_ranks = ranked_components(results)
    weakest = weakest_component(results)
    logger.info("--- Component Means by Label ---")
    header = f"  {'':>12}" + "".join(f"  {c:>8}" for c in COMPONENTS)
    logger.info(header)
    for label in LABELS:
        row = f"  {label:>12}"
        for comp in COMPONENTS:
            s = breakdown[label].get(comp)
            row += f"  {s.mean:8.3f}" if s else f"  {'N/A':>8}"
        logger.info(row)

    logger.info("--- Component Ranking (by good-bad separation) ---")
    for rank, (comp, sep) in enumerate(comp_ranks, 1):
        logger.info(f"  #{rank} {comp:>8}: separation={sep:+.3f}")
    if weakest:
        logger.info(f"  Weakest: {weakest}")

    # 4. Confidence
    conf = confidence_analysis(results)
    logger.info("--- Confidence Analysis ---")
    for label, mc in conf.mean_confidence_by_label.items():
        logger.info(f"  {label:>12}: mean_conf={mc:.3f}")
    logger.info(f"  Low-conf samples (<{conf.low_confidence_threshold}): {conf.low_confidence_count}")
    if conf.conf_score_correlation is not None:
        logger.info(f"  Conf-score correlation:  {conf.conf_score_correlation:.3f}")
    if conf.conf_error_correlation is not None:
        logger.info(f"  Conf-error correlation:  {conf.conf_error_correlation:.3f}")

    return {
        "label_order": ordered,
        "distributions": dists,
        "overlaps": overlaps,
        "component_breakdown": breakdown,
        "component_separations": dict(comp_ranks),
        "ranked_components": comp_ranks,
        "weakest_component": weakest,
        "confidence": conf,
    }
