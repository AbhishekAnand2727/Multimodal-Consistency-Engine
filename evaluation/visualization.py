"""
Visualization module for evaluation results.

All plots use matplotlib only (no seaborn).
No hardcoded colors — uses matplotlib's default color cycle.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from evaluation.dataset_loader import EvalResult
from evaluation.analysis import LABELS, COMPONENTS


def _valid_results(results: list[EvalResult]) -> list[EvalResult]:
    return [r for r in results if r.is_valid]


def _group_scores_by_label(results: list[EvalResult]) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = {label: [] for label in LABELS}
    for r in _valid_results(results):
        if r.label in groups:
            groups[r.label].append(r.overall_score)
    return groups


# -----------------------------------------------------------------------
# 1. Histogram of overall scores grouped by label
# -----------------------------------------------------------------------

def plot_score_histogram(
    results: list[EvalResult],
    output_path: str | Path | None = None,
    bins: int = 20,
) -> None:
    """
    Overlaid histogram of overall_score for each label class.
    Shows where score distributions sit and whether they separate.
    """
    groups = _group_scores_by_label(results)

    fig, ax = plt.subplots(figsize=(10, 6))

    for label in LABELS:
        scores = groups[label]
        if scores:
            ax.hist(
                scores,
                bins=bins,
                alpha=0.5,
                label=f"{label} (n={len(scores)})",
                range=(0, 1),
            )

    ax.set_xlabel("Overall Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution by Label")
    ax.legend()
    ax.set_xlim(0, 1)

    plt.tight_layout()
    _save_or_show(fig, output_path, "score_histogram")


# -----------------------------------------------------------------------
# 2. Box plots of overall scores per label
# -----------------------------------------------------------------------

def plot_score_boxplots(
    results: list[EvalResult],
    output_path: str | Path | None = None,
) -> None:
    """
    Box plots showing score spread per label class.
    Good for spotting overlap, outliers, and median separation.
    """
    groups = _group_scores_by_label(results)

    data = [groups[label] for label in LABELS]
    labels_with_n = [
        f"{label}\n(n={len(groups[label])})" for label in LABELS
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    bp = ax.boxplot(data, labels=labels_with_n, patch_artist=True)

    # Use default color cycle for box fills
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.6)

    ax.set_ylabel("Overall Score")
    ax.set_title("Score Spread by Label")
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    _save_or_show(fig, output_path, "score_boxplots")


# -----------------------------------------------------------------------
# 3. Component score comparison chart
# -----------------------------------------------------------------------

def plot_component_comparison(
    results: list[EvalResult],
    output_path: str | Path | None = None,
) -> None:
    """
    Grouped bar chart: mean of each component score per label.
    Reveals which components separate classes and which are flat.
    """
    valid = _valid_results(results)
    groups: dict[str, list[EvalResult]] = {label: [] for label in LABELS}
    for r in valid:
        if r.label in groups:
            groups[r.label].append(r)

    # Compute means
    means: dict[str, dict[str, float]] = {}
    stds: dict[str, dict[str, float]] = {}

    for label in LABELS:
        means[label] = {}
        stds[label] = {}
        for comp in COMPONENTS:
            vals = [r.components[comp] for r in groups[label] if comp in r.components]
            means[label][comp] = float(np.mean(vals)) if vals else 0.0
            stds[label][comp] = float(np.std(vals)) if vals else 0.0

    x = np.arange(len(COMPONENTS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, label in enumerate(LABELS):
        offsets = x + (i - 1) * width
        vals = [means[label][c] for c in COMPONENTS]
        errs = [stds[label][c] for c in COMPONENTS]
        ax.bar(
            offsets, vals, width,
            label=f"{label} (n={len(groups[label])})",
            yerr=errs,
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(COMPONENTS)
    ax.set_ylabel("Score")
    ax.set_title("Component Scores by Label")
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    _save_or_show(fig, output_path, "component_comparison")


# -----------------------------------------------------------------------
# 4. Confidence vs score scatter
# -----------------------------------------------------------------------

def plot_confidence_vs_score(
    results: list[EvalResult],
    output_path: str | Path | None = None,
) -> None:
    """
    Scatter plot: confidence (x) vs overall_score (y), colored by label.
    Shows whether low confidence correlates with unreliable scores.
    """
    valid = _valid_results(results)
    if not valid:
        logger.warning("No valid results to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    label_colors = {label: colors[i % len(colors)] for i, label in enumerate(LABELS)}

    for label in LABELS:
        subset = [r for r in valid if r.label == label]
        if subset:
            confs = [r.confidence for r in subset]
            scores = [r.overall_score for r in subset]
            ax.scatter(
                confs, scores,
                label=f"{label} (n={len(subset)})",
                color=label_colors[label],
                alpha=0.7,
                edgecolors="none",
            )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Overall Score")
    ax.set_title("Confidence vs Score")
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Reference lines
    ax.axhline(y=0.5, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.axvline(x=0.4, linestyle="--", linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    _save_or_show(fig, output_path, "confidence_vs_score")


# -----------------------------------------------------------------------
# Generate all plots
# -----------------------------------------------------------------------

def generate_all_plots(
    results: list[EvalResult],
    output_dir: str | Path = "evaluation/plots",
) -> None:
    """Generate all visualization charts and save to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_score_histogram(results, output_dir / "score_histogram.png")
    plot_score_boxplots(results, output_dir / "score_boxplots.png")
    plot_component_comparison(results, output_dir / "component_comparison.png")
    plot_confidence_vs_score(results, output_dir / "confidence_vs_score.png")

    logger.info(f"All plots saved to {output_dir}")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _save_or_show(fig, output_path: str | Path | None, default_name: str) -> None:
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved: {path}")
        plt.close(fig)
    else:
        plt.show()
