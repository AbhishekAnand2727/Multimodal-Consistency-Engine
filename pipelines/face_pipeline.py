"""
Face pipeline: extract age/gender distributions from video frames using DeepFace.
"""

import numpy as np
from deepface import DeepFace
from loguru import logger
from scipy.stats import norm

from models.schemas import FaceFeatures, GenderDist, AgeDist


# Age bucket boundaries for Gaussian smoothing
AGE_BUCKETS = [
    ("child",  0, 12),
    ("teen",   13, 19),
    ("adult",  20, 35),
    ("middle", 35, 55),
    ("senior", 55, 100),
]

# Bucket centers for distance calculations
BUCKET_CENTERS = [6, 16, 27.5, 45, 72.5]


def _age_to_distribution(age: float, sigma: float = 8.0) -> np.ndarray:
    """
    Convert a point age prediction to a soft probability distribution
    over age buckets using CDF-based probability mass.

    Uses CDF(upper) - CDF(lower) for each bucket instead of PDF at the
    center point, which correctly integrates probability mass over the
    entire bucket width rather than sampling a single point.
    """
    dist = np.zeros(len(AGE_BUCKETS))
    for i, (_, lower, upper) in enumerate(AGE_BUCKETS):
        dist[i] = norm.cdf(upper, loc=age, scale=sigma) - norm.cdf(lower, loc=age, scale=sigma)

    # Normalize to sum to 1
    total = dist.sum()
    if total > 0:
        dist /= total
    else:
        # Fallback: uniform
        dist = np.ones(len(AGE_BUCKETS)) / len(AGE_BUCKETS)

    return dist


def _analyze_single_frame(frame: np.ndarray) -> dict | None:
    """
    Run DeepFace on a single frame.
    Returns dict with 'age' and 'gender' or None if no face detected.
    """
    try:
        results = DeepFace.analyze(
            frame,
            actions=["age", "gender"],
            enforce_detection=False,
            silent=True,
        )

        if not results:
            return None

        # DeepFace returns a list; take the dominant (largest) face
        result = results[0] if isinstance(results, list) else results

        # DeepFace gender output: {"Man": prob, "Woman": prob}
        gender_raw = result.get("gender", {})
        p_male = gender_raw.get("Man", 0) / 100.0
        p_female = gender_raw.get("Woman", 0) / 100.0

        # If detection confidence is very low, skip
        face_confidence = result.get("face_confidence", 0)
        if face_confidence < 0.5:
            return None

        return {
            "age": float(result["age"]),
            "p_male": p_male,
            "p_female": p_female,
        }

    except Exception as e:
        logger.debug(f"Face analysis failed on frame: {e}")
        return None


def run_face_pipeline(frames: list[np.ndarray]) -> FaceFeatures:
    """
    Process multiple video frames and aggregate face predictions
    into probabilistic distributions.
    """
    if not frames:
        logger.warning("No frames provided to face pipeline")
        return FaceFeatures(
            gender_dist=GenderDist(male=0.5, female=0.5),
            age_dist=AgeDist(child=0.2, teen=0.2, adult=0.2, middle=0.2, senior=0.2),
            confidence=0.0,
        )

    valid_results = []
    for i, frame in enumerate(frames):
        result = _analyze_single_frame(frame)
        if result is not None:
            valid_results.append(result)
        if (i + 1) % 10 == 0:
            logger.info(f"Face pipeline: processed {i + 1}/{len(frames)} frames")

    logger.info(f"Face pipeline: {len(valid_results)}/{len(frames)} frames had valid detections")

    if not valid_results:
        return FaceFeatures(
            gender_dist=GenderDist(male=0.5, female=0.5),
            age_dist=AgeDist(child=0.2, teen=0.2, adult=0.2, middle=0.2, senior=0.2),
            confidence=0.0,
        )

    # Aggregate gender: mean of per-frame distributions
    p_males = [r["p_male"] for r in valid_results]
    p_females = [r["p_female"] for r in valid_results]
    mean_male = float(np.mean(p_males))
    mean_female = float(np.mean(p_females))

    # Normalize
    gender_total = mean_male + mean_female
    if gender_total > 0:
        mean_male /= gender_total
        mean_female /= gender_total

    # Aggregate age: convert each prediction to distribution, then average
    ages = [r["age"] for r in valid_results]
    age_dists = np.array([_age_to_distribution(a) for a in ages])
    mean_age_dist = age_dists.mean(axis=0)
    # Normalize
    mean_age_dist /= mean_age_dist.sum()

    # Confidence: based on detection ratio and prediction stability
    detection_ratio = len(valid_results) / len(frames)
    # Gender stability: low std = high stability
    gender_std = float(np.std(p_males))
    gender_stability = max(0, 1.0 - 2.0 * gender_std)  # penalty for high variance
    # Age stability
    age_std = float(np.std(ages))
    age_stability = max(0, 1.0 - age_std / 20.0)  # normalize by ~20 years

    confidence = float(
        0.5 * detection_ratio +
        0.25 * gender_stability +
        0.25 * age_stability
    )
    confidence = np.clip(confidence, 0.0, 1.0)

    return FaceFeatures(
        gender_dist=GenderDist(male=round(mean_male, 4), female=round(mean_female, 4)),
        age_dist=AgeDist(
            child=round(float(mean_age_dist[0]), 4),
            teen=round(float(mean_age_dist[1]), 4),
            adult=round(float(mean_age_dist[2]), 4),
            middle=round(float(mean_age_dist[3]), 4),
            senior=round(float(mean_age_dist[4]), 4),
        ),
        confidence=round(float(confidence), 4),
    )
