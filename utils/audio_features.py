"""
Audio feature extraction: pitch (F0), formants (F1/F2), VAD, segmentation.
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from loguru import logger


def apply_vad(audio: np.ndarray, sr: int, frame_length: int = 2048,
              hop_length: int = 512, energy_threshold: float = 0.01) -> np.ndarray:
    """
    Simple energy-based Voice Activity Detection.
    Returns audio with non-speech regions zeroed out,
    plus a boolean mask of speech frames.
    """
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    # Adaptive threshold: use the higher of the fixed floor and a
    # data-driven threshold based on the 30th percentile of RMS energy.
    # This prevents silent/low-amplitude audio from being entirely marked as speech.
    adaptive_threshold = max(
        energy_threshold,
        float(np.percentile(rms, 30)) * 3.0,
    )
    speech_frames = rms > adaptive_threshold

    # Convert frame-level mask to sample-level
    mask = np.repeat(speech_frames, hop_length)[:len(audio)]

    speech_ratio = np.mean(speech_frames)
    logger.info(f"VAD: {speech_ratio:.1%} speech detected")

    return audio * mask, speech_ratio


def segment_audio(audio: np.ndarray, sr: int,
                  min_duration: float = 2.0, max_duration: float = 5.0) -> list[np.ndarray]:
    """
    Split audio into segments of 2-5 seconds.
    """
    min_samples = int(min_duration * sr)
    max_samples = int(max_duration * sr)
    target_samples = int(3.0 * sr)  # aim for ~3s segments

    segments = []
    offset = 0

    while offset < len(audio):
        remaining = len(audio) - offset
        if remaining < min_samples:
            # Too short for a standalone segment — merge with last if possible,
            # but clamp to max_duration to prevent overflow.
            if segments:
                merged = np.concatenate([segments[-1], audio[offset:]])
                segments[-1] = merged[:max_samples]
            break

        seg_len = min(target_samples, remaining)
        # Don't exceed max
        seg_len = min(seg_len, max_samples)
        segments.append(audio[offset:offset + seg_len])
        offset += seg_len

    logger.info(f"Audio segmented into {len(segments)} chunks")
    return segments


def extract_pitch(audio: np.ndarray, sr: int) -> float | None:
    """
    Extract mean fundamental frequency (F0) using librosa's pyin.
    Returns mean pitch in Hz, or None if unvoiced.
    """
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),  # ~65 Hz
        fmax=librosa.note_to_hz("C7"),  # ~2093 Hz
        sr=sr,
    )

    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) == 0:
        return None

    # Remove outliers (below 50 Hz or above 500 Hz for speech)
    voiced_f0 = voiced_f0[(voiced_f0 >= 50) & (voiced_f0 <= 500)]
    if len(voiced_f0) == 0:
        return None

    return float(np.median(voiced_f0))


def extract_formants(
    audio: np.ndarray, sr: int, formant_ceiling: float = 5500.0,
) -> tuple[float | None, float | None]:
    """
    Extract F1 and F2 formant frequencies using Praat via parselmouth.
    Returns (F1_mean, F2_mean) in Hz, or None if extraction fails.

    formant_ceiling: maximum formant frequency for Burg analysis.
        Use 5000 Hz for male speakers, 5500 Hz for female speakers.
    """
    try:
        snd = parselmouth.Sound(audio, sampling_frequency=sr)
        formant = call(snd, "To Formant (burg)", 0.0, 5, formant_ceiling, 0.025, 50)

        n_frames = call(formant, "Get number of frames")
        if n_frames == 0:
            return None, None

        f1_values = []
        f2_values = []

        for i in range(1, n_frames + 1):
            t = call(formant, "Get time from frame number", i)
            f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")

            if f1 and not np.isnan(f1) and 200 < f1 < 1000:
                f1_values.append(f1)
            if f2 and not np.isnan(f2) and 500 < f2 < 3500:
                f2_values.append(f2)

        f1_mean = float(np.median(f1_values)) if f1_values else None
        f2_mean = float(np.median(f2_values)) if f2_values else None

        return f1_mean, f2_mean

    except Exception as e:
        logger.warning(f"Formant extraction failed: {e}")
        return None, None
