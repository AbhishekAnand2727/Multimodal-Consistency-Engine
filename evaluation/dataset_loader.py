"""
Dataset loader for evaluation and calibration.

Dataset format (JSON):
[
    {
        "video_path": "path/to/video.mp4",
        "label": "good",          # good | acceptable | bad
        "notes": "optional text"
    },
    ...
]

Results format (per-sample, stored after evaluation):
{
    "video_path": "...",
    "label": "good",
    "overall_score": 0.76,
    "confidence": 0.71,
    "components": {"gender": 0.92, "age": 0.64, "pitch": 0.72, "formant": 0.66},
    "face_confidence": 0.85,
    "voice_confidence": 0.71,
    "error": null
}
"""

import json
import csv
from pathlib import Path
from dataclasses import dataclass, field, asdict
from loguru import logger


VALID_LABELS = {"good", "acceptable", "bad"}


@dataclass
class DatasetSample:
    video_path: str
    label: str
    notes: str = ""

    def __post_init__(self):
        if self.label not in VALID_LABELS:
            raise ValueError(
                f"Invalid label '{self.label}'. Must be one of: {VALID_LABELS}"
            )


@dataclass
class EvalResult:
    """Result of running the evaluator on a single sample."""
    video_path: str
    label: str
    raw_score: float | None = None
    overall_score: float | None = None
    confidence: float | None = None
    components: dict[str, float] = field(default_factory=dict)
    face_confidence: float | None = None
    voice_confidence: float | None = None
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.error is None and self.overall_score is not None


def load_dataset(path: str | Path) -> list[DatasetSample]:
    """
    Load evaluation dataset from a JSON file.
    Validates labels and checks that video paths exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Dataset must be a JSON array of sample objects")

    samples = []
    skipped = 0

    for i, entry in enumerate(raw):
        try:
            sample = DatasetSample(
                video_path=entry["video_path"],
                label=entry["label"],
                notes=entry.get("notes", ""),
            )

            if not Path(sample.video_path).exists():
                logger.warning(
                    f"Sample {i}: video not found at '{sample.video_path}', skipping"
                )
                skipped += 1
                continue

            samples.append(sample)

        except (KeyError, ValueError) as e:
            logger.warning(f"Sample {i}: invalid entry — {e}, skipping")
            skipped += 1

    logger.info(f"Loaded {len(samples)} samples ({skipped} skipped) from {path}")
    return samples


def save_results_json(results: list[EvalResult], path: str | Path) -> None:
    """Save evaluation results as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info(f"Results saved to {path} ({len(results)} samples)")


def load_results_json(path: str | Path) -> list[EvalResult]:
    """Load previously saved evaluation results."""
    with open(path, "r") as f:
        raw = json.load(f)
    return [EvalResult(**r) for r in raw]


def save_results_csv(results: list[EvalResult], path: str | Path) -> None:
    """Save evaluation results as CSV for spreadsheet analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_path", "label", "overall_score", "confidence",
        "gender", "age", "pitch", "formant",
        "face_confidence", "voice_confidence", "error",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {
                "video_path": r.video_path,
                "label": r.label,
                "overall_score": r.overall_score,
                "confidence": r.confidence,
                "gender": r.components.get("gender"),
                "age": r.components.get("age"),
                "pitch": r.components.get("pitch"),
                "formant": r.components.get("formant"),
                "face_confidence": r.face_confidence,
                "voice_confidence": r.voice_confidence,
                "error": r.error,
            }
            writer.writerow(row)

    logger.info(f"CSV results saved to {path}")


def create_sample_dataset(path: str | Path) -> None:
    """Create an example dataset file as a starting template."""
    template = [
        {
            "video_path": "videos/good_match_male.mp4",
            "label": "good",
            "notes": "Male speaker, male voice, matching age"
        },
        {
            "video_path": "videos/acceptable_slight_age_gap.mp4",
            "label": "acceptable",
            "notes": "Gender matches but voice sounds older"
        },
        {
            "video_path": "videos/bad_gender_mismatch.mp4",
            "label": "bad",
            "notes": "Female face with male dubbed voice"
        },
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(template, f, indent=2)
    logger.info(f"Sample dataset template created at {path}")
