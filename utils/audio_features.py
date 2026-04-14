"""
Audio feature extraction: pitch (F0), formants (F1/F2), VAD, segmentation.
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from loguru import logger


def apply_vad(audio: np.ndarray, sr: int, frame_length: int = 2048,
              hop_length: int = 512, energy_threshold: float = 0.005) -> tuple[np.ndarray, float]:
    """
    Energy-based Voice Activity Detection with segment merging and filtering.

    Steps:
      1. Compute per-frame RMS energy.
      2. Threshold (fixed floor + adaptive component) to get raw speech frames.
      3. Convert frames → speech regions (start/end sample pairs).
      4. Merge regions whose gap is < 0.3 s (captures natural pauses in speech).
      5. Drop regions shorter than 0.5 s (removes noise bursts).
      6. Rebuild sample-level mask from cleaned regions.
      7. Compute speech_ratio as total_speech_samples / total_samples.

    Returns:
      audio_masked  — original audio with non-speech zeroed out
      speech_ratio  — fraction of audio that is speech (0–1)
    """
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    # Lower adaptive multiplier (was 3.0) so softer speech frames pass.
    adaptive_threshold = max(
        energy_threshold,
        float(np.percentile(rms, 30)) * 1.5,
    )
    speech_frames = rms > adaptive_threshold

    # ── Convert frame mask → (start_sample, end_sample) regions ──────────
    regions: list[tuple[int, int]] = []
    in_speech = False
    seg_start = 0
    for i, active in enumerate(speech_frames):
        sample = i * hop_length
        if active and not in_speech:
            seg_start = sample
            in_speech = True
        elif not active and in_speech:
            regions.append((seg_start, sample))
            in_speech = False
    if in_speech:
        regions.append((seg_start, min(len(speech_frames) * hop_length, len(audio))))

    # ── Merge gaps < 0.3 s ────────────────────────────────────────────────
    merge_gap = int(0.3 * sr)
    if regions:
        merged: list[tuple[int, int]] = [regions[0]]
        for start, end in regions[1:]:
            if start - merged[-1][1] < merge_gap:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        regions = merged

    # ── Remove regions shorter than 0.5 s ────────────────────────────────
    min_samples = int(0.5 * sr)
    regions = [(s, e) for s, e in regions if e - s >= min_samples]

    # ── Rebuild sample-level mask ─────────────────────────────────────────
    mask = np.zeros(len(audio), dtype=np.float32)
    for start, end in regions:
        mask[start:min(end, len(audio))] = 1

    # ── Speech ratio based on merged/filtered regions ─────────────────────
    total_speech = sum(e - s for s, e in regions)
    total_samples = len(audio)
    speech_ratio = total_speech / total_samples if total_samples > 0 else 0.0
    total_dur = total_samples / sr
    speech_dur = total_speech / sr
    logger.info(
        f"VAD: speech {speech_dur:.1f}s / {total_dur:.1f}s "
        f"({speech_ratio:.1%}, {len(regions)} regions after merge/filter)"
    )

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
