"""
Scoring engine: compute compatibility between face and voice features.
All scoring is smooth/probabilistic — no hard thresholds.

Signal independence:
  - gender_score uses ONLY gender distributions (face vs voice model output)
  - age_score uses ONLY age distributions (face vs voice spectral heuristic)
  - pitch_score uses ONLY pitch vs face-gender-expected range
  - formant_score uses ONLY F1/F2 vs face-gender-expected range

No feature feeds into another scorer — zero double-counting.
"""

import numpy as np
from loguru import logger

from models.schemas import (
    FaceFeatures,
    VoiceFeatures,
    ComponentScores,
    EvaluationResult,
)


# --- Pitch reference ranges (Hz) ---
# Male typical: 85-180, Female typical: 165-255
MALE_PITCH_CENTER = 132.5
MALE_PITCH_SIGMA = 35.0
FEMALE_PITCH_CENTER = 210.0
FEMALE_PITCH_SIGMA = 30.0

# --- Formant reference ranges (Hz) ---
# F2 is the primary discriminator of vocal tract length
MALE_F2_CENTER = 1100.0
MALE_F2_SIGMA = 200.0
FEMALE_F2_CENTER = 1600.0
FEMALE_F2_SIGMA = 250.0

# F1 is supporting (jaw opening / vowel height)
MALE_F1_CENTER = 400.0
MALE_F1_SIGMA = 80.0
FEMALE_F1_CENTER = 500.0
FEMALE_F1_SIGMA = 100.0

# Neutral score returned when a feature is unavailable.
# 0.5 means "no evidence for or against" — does not inflate or deflate.
NEUTRAL_SCORE = 0.5

# --- Scoring weights ---
WEIGHT_GENDER = 0.60
WEIGHT_AGE = 0.10
WEIGHT_PITCH = 0.30
# WEIGHT_FORMANT removed — formant excluded from scoring


def _dot_similarity(a: list[float], b: list[float]) -> float:
    """Dot product similarity between two probability distributions."""
    return float(np.dot(a, b))


def _gaussian_score(value: float, center: float, sigma: float) -> float:
    """
    Smooth score: how well a value fits an expected distribution.
    Returns 0-1. Peak (1.0) at center, falls off with sigma.
    """
    z = (value - center) / sigma
    return float(np.exp(-0.5 * z * z))


def score_gender(face: FaceFeatures, voice: VoiceFeatures) -> float:
    """
    Gender alignment via dot product of distributions.

    Inputs: face gender distribution (from DeepFace)
            voice gender distribution (from wav2vec2 classifier)
    No pitch involved — pure model-vs-model comparison.
    """
    score = _dot_similarity(
        face.gender_dist.as_vector(),
        voice.gender_dist.as_vector(),
    )
    return round(score, 4)


def score_age(face: FaceFeatures, voice: VoiceFeatures) -> float:
    """
    Age similarity via dot product of age distributions.

    Inputs: face age distribution (from DeepFace, Gaussian smoothed)
            voice age distribution (from spectral centroid heuristic, wide sigma)
    No pitch or formant involved.
    """
    score = _dot_similarity(
        face.age_dist.as_vector(),
        voice.age_dist.as_vector(),
    )
    return round(score, 4)


def score_pitch(face: FaceFeatures, voice: VoiceFeatures) -> float:
    """
    Pitch plausibility: does the observed F0 fit the expected range
    for the face's apparent gender?

    Uses ONLY pitch_mean and face_gender_dist.
    Returns NEUTRAL_SCORE if pitch data is unavailable.
    """
    if voice.pitch_mean <= 0:
        logger.debug("Pitch unavailable — returning neutral score")
        return NEUTRAL_SCORE

    pitch = voice.pitch_mean
    p_male = face.gender_dist.male
    p_female = face.gender_dist.female

    male_fit = _gaussian_score(pitch, MALE_PITCH_CENTER, MALE_PITCH_SIGMA)
    female_fit = _gaussian_score(pitch, FEMALE_PITCH_CENTER, FEMALE_PITCH_SIGMA)

    # Weighted blend by face gender probability
    score = p_male * male_fit + p_female * female_fit

    return round(float(score), 4)


def score_formant(face: FaceFeatures, voice: VoiceFeatures) -> float:
    """
    Formant plausibility: do F1/F2 fit expected ranges for the face's
    apparent gender?

    F2 is primary (vocal tract length), F1 is supporting.
    Returns NEUTRAL_SCORE if formant data is entirely unavailable.
    Does NOT use fake fallback values — absence is handled explicitly.
    """
    f2_available = voice.f2_mean > 0
    f1_available = voice.f1_mean > 0

    if not f2_available and not f1_available:
        logger.debug("No formant data available — returning neutral score")
        return NEUTRAL_SCORE

    p_male = face.gender_dist.male
    p_female = face.gender_dist.female

    # F2 scoring (primary — only if available)
    if f2_available:
        f2_male_fit = _gaussian_score(voice.f2_mean, MALE_F2_CENTER, MALE_F2_SIGMA)
        f2_female_fit = _gaussian_score(voice.f2_mean, FEMALE_F2_CENTER, FEMALE_F2_SIGMA)
        f2_score = p_male * f2_male_fit + p_female * f2_female_fit
    else:
        f2_score = None

    # F1 scoring (supporting — only if available)
    if f1_available:
        f1_male_fit = _gaussian_score(voice.f1_mean, MALE_F1_CENTER, MALE_F1_SIGMA)
        f1_female_fit = _gaussian_score(voice.f1_mean, FEMALE_F1_CENTER, FEMALE_F1_SIGMA)
        f1_score = p_male * f1_male_fit + p_female * f1_female_fit
    else:
        f1_score = None

    # Combine available signals with appropriate weights
    if f2_score is not None and f1_score is not None:
        score = 0.7 * f2_score + 0.3 * f1_score
    elif f2_score is not None:
        score = f2_score  # F2 alone (primary signal)
    else:
        score = f1_score  # F1 alone (rare case — weak signal only)

    return round(float(score), 4)


def compute_evaluation(face: FaceFeatures, voice: VoiceFeatures) -> EvaluationResult:
    """
    Full evaluation: compute all component scores, confidence,
    and final weighted score.
    """
    gender = score_gender(face, voice)
    age = score_age(face, voice)
    formant = score_formant(face, voice)  # kept for output; not used in raw_score

    # Pitch: range-based scoring anchored to face-apparent gender
    if voice.pitch_mean > 0:
        face_gender = "female" if face.gender_dist.female > face.gender_dist.male else "male"
        if face_gender == "female":
            expected_low, expected_high = 165, 255
        else:
            expected_low, expected_high = 85, 180

        if expected_low <= voice.pitch_mean <= expected_high:
            pitch = 1.0
        else:
            distance = min(
                abs(voice.pitch_mean - expected_low),
                abs(voice.pitch_mean - expected_high),
            )
            pitch = max(0.0, 1.0 - distance / 200)
        pitch = round(float(pitch), 4)
    else:
        logger.debug("Pitch unavailable — returning neutral score")
        pitch = NEUTRAL_SCORE

    raw_score = (
        WEIGHT_GENDER * gender +
        WEIGHT_PITCH * pitch +
        WEIGHT_AGE * age
    )

    # Confidence: weighted average — less harsh than min()
    confidence = 0.6 * voice.confidence + 0.4 * face.confidence

    # Final score scaled by confidence
    final_score = raw_score * confidence

    # --- Debug logging ---
    logger.info(
        f"Component scores — gender: {gender}, age: {age}, "
        f"pitch: {pitch}, formant: {formant}"
    )
    logger.info(
        f"Feature availability — pitch: {voice.pitch_mean > 0}, "
        f"F1: {voice.f1_mean > 0}, F2: {voice.f2_mean > 0}"
    )
    logger.info(
        f"Confidence — face: {face.confidence}, voice: {voice.confidence}, "
        f"effective: {confidence}"
    )
    logger.info(
        f"Final — raw: {raw_score:.4f}, confidence: {confidence:.4f}, "
        f"score: {final_score:.4f}"
    )

    return EvaluationResult(
        raw_score=round(raw_score, 4),
        overall_score=round(final_score, 4),
        confidence=round(confidence, 4),
        components=ComponentScores(
            gender=gender,
            age=age,
            pitch=pitch,
            formant=formant,
        ),
        face_features=face,
        voice_features=voice,
    )
