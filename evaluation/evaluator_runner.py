"""
Batch evaluation runner.

Runs the core evaluator on every sample in a dataset,
captures results (including failures), and persists them.
Does NOT modify any evaluator logic — purely observational.
"""

import time
import traceback
from pathlib import Path
from loguru import logger

from evaluation.dataset_loader import (
    DatasetSample,
    EvalResult,
    load_dataset,
    save_results_json,
    save_results_csv,
)
from pipelines.face_pipeline import run_face_pipeline
from pipelines.audio_pipeline import run_audio_pipeline
from pipelines.scoring import compute_evaluation
from utils.preprocessing import extract_frames, extract_audio


def evaluate_single(sample: DatasetSample, job_dir: str) -> EvalResult:
    """
    Run the full evaluation pipeline on a single video.
    Returns an EvalResult regardless of success or failure.
    """
    t0 = time.time()

    try:
        # Preprocessing
        frames = extract_frames(sample.video_path, fps=2.0)
        audio_path = extract_audio(sample.video_path, output_dir=job_dir)

        # Pipelines
        face_features = run_face_pipeline(frames)
        voice_features = run_audio_pipeline(audio_path)

        # Scoring
        result = compute_evaluation(face_features, voice_features)

        elapsed = time.time() - t0
        logger.info(
            f"[OK] {Path(sample.video_path).name} | "
            f"label={sample.label} | score={result.overall_score:.3f} | "
            f"conf={result.confidence:.3f} | {elapsed:.1f}s"
        )

        return EvalResult(
            video_path=sample.video_path,
            label=sample.label,
            raw_score=result.raw_score,
            overall_score=result.overall_score,
            confidence=result.confidence,
            components={
                "gender": result.components.gender,
                "age": result.components.age,
                "pitch": result.components.pitch,
                "formant": result.components.formant,
            },
            face_confidence=face_features.confidence,
            voice_confidence=voice_features.confidence,
        )

    except Exception as e:
        elapsed = time.time() - t0
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(
            f"[FAIL] {Path(sample.video_path).name} | "
            f"label={sample.label} | {error_msg} | {elapsed:.1f}s"
        )
        logger.debug(traceback.format_exc())

        return EvalResult(
            video_path=sample.video_path,
            label=sample.label,
            error=error_msg,
        )


def evaluate_dataset(
    dataset_path: str | Path,
    output_dir: str | Path = "evaluation/results",
    output_name: str = "eval_results",
) -> list[EvalResult]:
    """
    Batch evaluate all samples in a dataset file.

    Args:
        dataset_path: Path to the dataset JSON file.
        output_dir: Where to write result files.
        output_name: Base filename for results (without extension).

    Returns:
        List of EvalResult objects.
    """
    samples = load_dataset(dataset_path)
    if not samples:
        logger.warning("No valid samples in dataset — nothing to evaluate.")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-sample temp directories for audio extraction
    tmp_base = output_dir / "tmp"
    tmp_base.mkdir(exist_ok=True)

    results: list[EvalResult] = []
    n_ok, n_fail = 0, 0

    logger.info(f"Starting batch evaluation: {len(samples)} samples")
    t_total = time.time()

    for i, sample in enumerate(samples):
        logger.info(f"--- Sample {i+1}/{len(samples)}: {Path(sample.video_path).name} ---")

        job_dir = str(tmp_base / f"sample_{i:04d}")
        Path(job_dir).mkdir(exist_ok=True)

        result = evaluate_single(sample, job_dir)
        results.append(result)

        if result.is_valid:
            n_ok += 1
        else:
            n_fail += 1

    elapsed_total = time.time() - t_total

    logger.info(
        f"Batch complete: {n_ok} succeeded, {n_fail} failed, "
        f"{elapsed_total:.1f}s total"
    )

    # Save results in both formats
    save_results_json(results, output_dir / f"{output_name}.json")
    save_results_csv(results, output_dir / f"{output_name}.csv")

    # Print per-sample debug summary
    _print_debug_summary(results)

    return results


def _print_debug_summary(results: list[EvalResult]) -> None:
    """Per-sample debug log for quick visual inspection."""
    logger.info("=" * 70)
    logger.info("PER-SAMPLE SUMMARY")
    logger.info("=" * 70)

    for r in results:
        name = Path(r.video_path).name
        if r.is_valid:
            comps = r.components
            logger.info(
                f"  {name:<35} | label={r.label:<12} | "
                f"score={r.overall_score:.3f} | conf={r.confidence:.3f} | "
                f"G={comps['gender']:.2f} A={comps['age']:.2f} "
                f"P={comps['pitch']:.2f} F={comps['formant']:.2f}"
            )
        else:
            logger.info(
                f"  {name:<35} | label={r.label:<12} | "
                f"FAILED: {r.error}"
            )

    logger.info("=" * 70)


def compare_versions(
    results_a_path: str | Path,
    results_b_path: str | Path,
) -> dict:
    """
    Compare evaluation results from two model versions.

    Loads results from two JSON files and computes:
      - per-label mean score delta
      - per-component mean delta
      - overall improvement/regression summary

    Returns a dict with the comparison.
    """
    from evaluation.dataset_loader import load_results_json

    results_a = load_results_json(results_a_path)
    results_b = load_results_json(results_b_path)

    # Index by video_path for alignment
    a_by_path = {r.video_path: r for r in results_a if r.is_valid}
    b_by_path = {r.video_path: r for r in results_b if r.is_valid}

    common_paths = set(a_by_path.keys()) & set(b_by_path.keys())
    if not common_paths:
        logger.warning("No overlapping valid samples between versions.")
        return {"error": "no overlapping samples"}

    deltas_by_label: dict[str, list[float]] = {}
    component_deltas: dict[str, list[float]] = {
        "gender": [], "age": [], "pitch": [], "formant": [],
    }

    for path in common_paths:
        ra, rb = a_by_path[path], b_by_path[path]
        label = ra.label
        delta = rb.overall_score - ra.overall_score

        deltas_by_label.setdefault(label, []).append(delta)

        for comp in component_deltas:
            if comp in ra.components and comp in rb.components:
                component_deltas[comp].append(
                    rb.components[comp] - ra.components[comp]
                )

    import numpy as np

    comparison = {
        "n_compared": len(common_paths),
        "overall_delta_by_label": {
            label: {
                "mean_delta": float(np.mean(ds)),
                "n_samples": len(ds),
            }
            for label, ds in deltas_by_label.items()
        },
        "component_deltas": {
            comp: float(np.mean(ds)) if ds else 0.0
            for comp, ds in component_deltas.items()
        },
    }

    # Log summary
    logger.info("=" * 50)
    logger.info("VERSION COMPARISON (B vs A)")
    logger.info("=" * 50)
    for label, info in comparison["overall_delta_by_label"].items():
        direction = "+" if info["mean_delta"] >= 0 else ""
        logger.info(
            f"  {label}: {direction}{info['mean_delta']:.4f} "
            f"({info['n_samples']} samples)"
        )
    for comp, delta in comparison["component_deltas"].items():
        direction = "+" if delta >= 0 else ""
        logger.info(f"  {comp}: {direction}{delta:.4f}")
    logger.info("=" * 50)

    return comparison
