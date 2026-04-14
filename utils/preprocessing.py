"""
Preprocessing utilities: frame extraction and audio extraction.
"""

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


def extract_frames(video_path: str, fps: float = 2.0) -> list[np.ndarray]:
    """
    Extract frames from video at the given FPS rate.
    Returns list of BGR frames (OpenCV format).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            raise ValueError(f"Invalid video FPS: {video_fps}")

        # Sample every N-th frame to achieve target FPS
        frame_interval = max(1, int(round(video_fps / fps)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Video: {video_fps:.1f} FPS, {total_frames} frames. "
            f"Sampling every {frame_interval} frames (~{fps} FPS)"
        )

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frames.append(frame)
            frame_idx += 1

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
