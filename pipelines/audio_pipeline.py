"""
Audio pipeline: extract voice gender, age, pitch, and formants
from dubbed audio with VAD and segmentation.

Signal independence contract:
  - Gender  → wav2vec2 classifier (NO pitch dependency)
  - Age     → spectral-centroid heuristic (NO pitch/formant dependency)
  - Pitch   → librosa pyin (independent physical signal)
  - Formant → parselmouth Praat (independent physical signal)

Each feature is extracted independently per segment, then aggregated
using RMS-energy-weighted means across segments.
"""

import numpy as np
import librosa
import torch
from functools import lru_cache
from scipy.stats import norm
from loguru import logger

from models.schemas import VoiceFeatures, GenderDist, AgeDist
from utils.audio_features import (
    apply_vad,
    segment_audio,
    extract_pitch,
    extract_formants,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGE_BUCKET_CENTERS = [6.0, 16.0, 27.5, 45.0, 72.5]
AGE_BUCKET_BOUNDS = [(0, 12), (13, 19), (20, 35), (35, 55), (55, 100)]

# Spectral centroid reference ranges (Hz) for age heuristic.
# Younger speakers tend to have higher spectral centroids.
# These are approximate — the wide sigma ensures this signal stays weak.
SC_YOUNG_CENTER = 2800.0   # children / teens
SC_ADULT_CENTER = 2200.0   # adults
SC_SENIOR_CENTER = 1600.0  # seniors


# ---------------------------------------------------------------------------
# Gender classifier — lazy-loaded, cached singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_gender_classifier():
    """
    Load a pretrained wav2vec2-based gender classifier from HuggingFace.
    Returns a transformers pipeline that outputs [{"label": "male/female", "score": float}].
    Cached after first call — zero cost on subsequent invocations.
    """
    from transformers import pipeline as hf_pipeline

    logger.info("Loading wav2vec2 gender classifier (first call only)...")
    clf = hf_pipeline(
        task="audio-classification",
        model="alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    logger.info("Gender classifier loaded.")
    return clf


def _classify_gender_segment(segment: np.ndarray, sr: int) -> tuple[float, float] | None:
    """
    Classify gender for a single audio segment using the wav2vec2 model.
    Returns (p_male, p_female) or None if classification fails.
    """
    try:
        clf = _load_gender_classifier()

        # The pipeline expects {"raw": array, "sampling_rate": int}
        result = clf({"raw": segment.astype(np.float32), "sampling_rate": sr})

        p_male, p_female = 0.5, 0.5
        for entry in result:
            label = entry["label"].lower()
            if label == "male":
                p_male = entry["score"]
            elif label == "female":
                p_female = entry["score"]

        # Normalize (should already sum to ~1, but be safe)
        total = p_male + p_female
        if total > 0:
            p_male /= total
            p_female /= total

        return float(p_male), float(p_female)

    except Exception as e:
        logger.warning(f"Gender classification failed for segment: {e}")
        return None


# ---------------------------------------------------------------------------
# Age heuristic — spectral centroid based (NOT pitch-based)
# ---------------------------------------------------------------------------

def _estimate_age_from_spectral_centroid(segment: np.ndarray, sr: int) -> float | None:
    """
    Weak age proxy using spectral centroid.

    Rationale: younger vocal tracts are shorter, producing higher spectral
    energy concentration. This is a rough signal — we use a very wide sigma
    downstream so it cannot dominate the age distribution.

    IMPORTANT: This does NOT use pitch or formants, preserving signal
    independence in the scoring engine.
    """
    try:
        centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
        centroid = centroid[centroid > 0]
        if len(centroid) == 0:
            return None

        sc_mean = float(np.median(centroid))

        # Map spectral centroid → rough age estimate
        # Higher centroid → younger, lower → older
        if sc_mean > 2600:
            return 15.0
        elif sc_mean > 2300:
            return 25.0
        elif sc_mean > 2000:
            return 38.0
        elif sc_mean > 1700:
            return 50.0
        else:
            return 62.0

    except Exception as e:
        logger.debug(f"Spectral centroid extraction failed: {e}")
        return None


def _age_to_distribution(age: float, sigma: float = 15.0) -> np.ndarray:
    """
    Convert a point age estimate to a soft probability distribution
    over 5 age buckets using CDF-based probability mass.

    Uses CDF(upper) - CDF(lower) for each bucket instead of PDF at the
    center point, correctly integrating over the full bucket width.

    sigma=15 (wider than face pipeline's sigma=8) because audio-based
    age estimation is inherently less reliable than visual.
    """
    dist = np.zeros(5)
    for i, (lower, upper) in enumerate(AGE_BUCKET_BOUNDS):
        dist[i] = norm.cdf(upper, loc=age, scale=sigma) - norm.cdf(lower, loc=age, scale=sigma)
    total = dist.sum()
    if total > 0:
        dist /= total
    else:
        dist = np.ones(5) / 5.0
    return dist


# ---------------------------------------------------------------------------
# Per-segment analysis
# ---------------------------------------------------------------------------

def _compute_segment_weight(segment: np.ndarray) -> float:
    """
    Compute segment weight based on RMS energy.
    Higher energy → more likely speech → more trustworthy.
    """
    return float(np.sqrt(np.mean(segment ** 2)))


def _analyze_segment(segment: np.ndarray, sr: int) -> dict | None:
    """
    Analyze a single audio segment. Each feature is extracted independently:
      - gender:  wav2vec2 classifier
      - age:     spectral centroid heuristic
      - pitch:   librosa pyin
      - formant: parselmouth Praat

    Returns dict with all extracted features, or None if segment is too quiet.
    """
    rms = np.sqrt(np.mean(segment ** 2))
    if rms < 0.005:
        return None

    # --- Gender (wav2vec2 model — independent of pitch) ---
    gender_result = _classify_gender_segment(segment, sr)
    if gender_result is None:
        p_male, p_female = 0.5, 0.5
        gender_available = False
    else:
        p_male, p_female = gender_result
        gender_available = True

    # --- Pitch (independent physical signal) ---
    pitch = extract_pitch(segment, sr)
    pitch_available = pitch is not None

    # --- Formants (independent physical signal) ---
    # Adjust formant ceiling based on detected gender: 5000 Hz for male, 5500 Hz for female.
    formant_ceiling = 5000.0 if p_male > p_female else 5500.0
    f1, f2 = extract_formants(segment, sr, formant_ceiling=formant_ceiling)
    f1_available = f1 is not None
    f2_available = f2 is not None

    # --- Age (spectral centroid — independent of pitch and formants) ---
    estimated_age = _estimate_age_from_spectral_centroid(segment, sr)
    age_available = estimated_age is not None

    return {
        "p_male": p_male,
        "p_female": p_female,
        "gender_available": gender_available,
        "pitch": pitch,
        "pitch_available": pitch_available,
        "f1": f1,
        "f2": f2,
        "f1_available": f1_available,
        "f2_available": f2_available,
        "age": estimated_age,
        "age_available": age_available,
        "weight": rms,
    }


# ---------------------------------------------------------------------------
# Weighted aggregation helpers
# ---------------------------------------------------------------------------

def _weighted_mean(values: list[float], weights: list[float]) -> float:
    """Weighted mean. Falls back to simple mean if weights sum to zero."""
    w = np.array(weights)
    v = np.array(values)
    w_sum = w.sum()
    if w_sum > 0:
        return float(np.dot(v, w) / w_sum)
    return float(np.mean(v))


def _weighted_dist_mean(dists: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted mean of distribution vectors."""
    w = np.array(weights)
    d = np.array(dists)
    w_sum = w.sum()
    if w_sum > 0:
        result = (d.T @ w) / w_sum
    else:
        result = d.mean(axis=0)
    # Normalize to sum to 1
    total = result.sum()
    if total > 0:
        result /= total
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_audio_pipeline(audio_path: str) -> VoiceFeatures:
    """
    Full audio pipeline: load → VAD → segment → analyze → weighted aggregate.

    Each feature (gender, age, pitch, formant) is extracted independently.
    Segments are weighted by RMS energy for aggregation.
    Missing features are tracked and penalize confidence.
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio) / sr
    logger.info(f"Audio loaded: {duration:.1f}s at {sr}Hz")

    if duration < 0.5:
        logger.warning("Audio too short for analysis")
        return _empty_voice_features()

    # VAD
    audio_clean, speech_ratio = apply_vad(audio, sr)

    if speech_ratio < 0.05:
        logger.warning("Almost no speech detected in audio")
        return _empty_voice_features()

    # Segment
    segments = segment_audio(audio_clean, sr)
    if not segments:
        return _empty_voice_features()

    # Analyze each segment
    results = []
    for i, seg in enumerate(segments):
        result = _analyze_segment(seg, sr)
        if result is not None:
            results.append(result)

    n_total = len(segments)
    n_valid = len(results)
    logger.info(f"Audio pipeline: {n_valid}/{n_total} segments produced results")

    if not results:
        return _empty_voice_features()

    # --- Debug: feature extraction success rates ---
    n_gender = sum(1 for r in results if r["gender_available"])
    n_pitch = sum(1 for r in results if r["pitch_available"])
    n_f1 = sum(1 for r in results if r["f1_available"])
    n_f2 = sum(1 for r in results if r["f2_available"])
    n_age = sum(1 for r in results if r["age_available"])
    logger.info(
        f"Feature success rates — gender: {n_gender}/{n_valid}, "
        f"pitch: {n_pitch}/{n_valid}, F1: {n_f1}/{n_valid}, "
        f"F2: {n_f2}/{n_valid}, age: {n_age}/{n_valid}"
    )

    weights = [r["weight"] for r in results]

    # ---- Aggregate gender (weighted by RMS) ----
    gender_results = [r for r in results if r["gender_available"]]
    if gender_results:
        g_weights = [r["weight"] for r in gender_results]
        mean_male = _weighted_mean([r["p_male"] for r in gender_results], g_weights)
        mean_female = _weighted_mean([r["p_female"] for r in gender_results], g_weights)
        g_total = mean_male + mean_female
        if g_total > 0:
            mean_male /= g_total
            mean_female /= g_total
    else:
        mean_male, mean_female = 0.5, 0.5

    # ---- Aggregate pitch (weighted, only segments with valid pitch) ----
    pitch_results = [r for r in results if r["pitch_available"]]
    if pitch_results:
        p_weights = [r["weight"] for r in pitch_results]
        pitch_mean = _weighted_mean([r["pitch"] for r in pitch_results], p_weights)
    else:
        pitch_mean = 0.0  # 0.0 signals "unavailable" to scoring

    # ---- Aggregate formants (weighted, only segments with valid data) ----
    # F1 and F2 tracked independently — no fake fallback values
    f1_results = [r for r in results if r["f1_available"]]
    f2_results = [r for r in results if r["f2_available"]]

    if f1_results:
        f1_weights = [r["weight"] for r in f1_results]
        f1_mean = _weighted_mean([r["f1"] for r in f1_results], f1_weights)
    else:
        f1_mean = 0.0  # 0.0 signals "unavailable" to scoring

    if f2_results:
        f2_weights = [r["weight"] for r in f2_results]
        f2_mean = _weighted_mean([r["f2"] for r in f2_results], f2_weights)
    else:
        f2_mean = 0.0  # 0.0 signals "unavailable" to scoring

    # ---- Aggregate age (weighted, only segments with valid age) ----
    age_results = [r for r in results if r["age_available"]]
    if age_results:
        a_weights = [r["weight"] for r in age_results]
        age_dists = [_age_to_distribution(r["age"]) for r in age_results]
        mean_age_dist = _weighted_dist_mean(age_dists, a_weights)
    else:
        # Uniform — no information
        mean_age_dist = np.ones(5) / 5.0

    # ---- Confidence estimation ----
    confidence = _compute_confidence(
        speech_ratio=speech_ratio,
        n_valid=n_valid,
        n_total=n_total,
        n_gender=n_gender,
        n_pitch=n_pitch,
        n_f2=n_f2,
        results=results,
    )

    return VoiceFeatures(
        gender_dist=GenderDist(male=round(mean_male, 4), female=round(mean_female, 4)),
        age_dist=AgeDist(
            child=round(float(mean_age_dist[0]), 4),
            teen=round(float(mean_age_dist[1]), 4),
            adult=round(float(mean_age_dist[2]), 4),
            middle=round(float(mean_age_dist[3]), 4),
            senior=round(float(mean_age_dist[4]), 4),
        ),
        pitch_mean=round(pitch_mean, 2),
        f1_mean=round(f1_mean, 2),
        f2_mean=round(f2_mean, 2),
        confidence=round(confidence, 4),
    )


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def _compute_confidence(
    speech_ratio: float,
    n_valid: int,
    n_total: int,
    n_gender: int,
    n_pitch: int,
    n_f2: int,
    results: list[dict],
) -> float:
    """
    Confidence reflects how much we trust the extracted features.

    Components:
      1. speech_ratio   — proportion of audio that is speech (from VAD)
      2. segment_ratio  — proportion of segments that produced any result
      3. feature_avail  — average availability across gender/pitch/formant
      4. consistency    — stability of gender predictions across segments

    Missing features (no pitch, no formants) directly reduce confidence.
    """
    if n_valid == 0:
        return 0.0

    # 1. Speech presence
    speech_score = min(speech_ratio / 0.5, 1.0)  # saturates at 50% speech

    # 2. Segment success ratio
    segment_score = n_valid / max(n_total, 1)

    # 3. Feature availability (each penalizes if missing)
    gender_avail = n_gender / max(n_valid, 1)
    pitch_avail = n_pitch / max(n_valid, 1)
    formant_avail = n_f2 / max(n_valid, 1)
    feature_score = (gender_avail + pitch_avail + formant_avail) / 3.0

    # 4. Gender consistency across segments
    gender_results = [r for r in results if r["gender_available"]]
    if len(gender_results) >= 2:
        p_males = [r["p_male"] for r in gender_results]
        gender_std = float(np.std(p_males))
        consistency = max(0.0, 1.0 - 2.0 * gender_std)
    else:
        consistency = 0.5  # insufficient data for consistency check

    confidence = (
        0.25 * speech_score +
        0.20 * segment_score +
        0.30 * feature_score +
        0.25 * consistency
    )

    return float(np.clip(confidence, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _empty_voice_features() -> VoiceFeatures:
    """Return neutral features with zero confidence when audio analysis fails."""
    return VoiceFeatures(
        gender_dist=GenderDist(male=0.5, female=0.5),
        age_dist=AgeDist(child=0.2, teen=0.2, adult=0.2, middle=0.2, senior=0.2),
        pitch_mean=0.0,
        f1_mean=0.0,
        f2_mean=0.0,
        confidence=0.0,
    )
