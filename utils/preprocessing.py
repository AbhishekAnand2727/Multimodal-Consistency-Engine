"""
Preprocessing utilities: frame extraction and audio extraction.
"""

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


TARGET_FRAMES = 50


def extract_frames(video_path: str, fps: float = 2.0) -> list[np.ndarray]:
    """
    Extract frames from video using uniform sampling (constant frame count).
    Returns list of BGR frames (OpenCV format).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < TARGET_FRAMES:
            indices = np.arange(total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, TARGET_FRAMES).astype(int)

        logger.info(
            f"Uniform sampling: total_frames={total_frames}, sampled={len(indices)}"
        )

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    finally:
        cap.release()


def extract_audio(video_path: str, output_dir: str | None = None) -> str:
    """
    Extract audio from video as mono WAV at 16kHz using ffmpeg.
    Returns path to the extracted WAV file.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    output_path = str(Path(output_dir) / "audio.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                  # no video
        "-acodec", "pcm_s16le", # 16-bit PCM
        "-ar", "16000",         # 16kHz sample rate
        "-ac", "1",             # mono
        output_path,
    ]

    logger.info(f"Extracting audio: {video_path} -> {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    return output_path
