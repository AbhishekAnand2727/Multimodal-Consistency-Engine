"""
FastAPI backend for the Multimodal Dubbing Evaluator.
Async job-based: upload returns job_id, poll for result.
"""

import uuid
import time
import shutil
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from models.schemas import JobResponse, JobStatus, EvaluationResult
from pipelines.face_pipeline import run_face_pipeline
from pipelines.audio_pipeline import run_audio_pipeline
from pipelines.scoring import compute_evaluation
from utils.preprocessing import extract_frames, extract_audio


# In-memory job store (swap for Redis/DB in production)
jobs: dict[str, JobResponse] = {}
# Timestamps for TTL-based cleanup (job_id → creation time)
job_timestamps: dict[str, float] = {}
# Original filenames for dashboard listing
job_filenames: dict[str, str] = {}

JOBS_DIR = Path("jobs")
JOB_TTL_SECONDS = 3600  # 1 hour
ROOT_DIR = Path(__file__).parent.parent  # project root (where index.html lives)


@asynccontextmanager
async def lifespan(app: FastAPI):
    JOBS_DIR.mkdir(exist_ok=True)
    logger.info("Dubbing Evaluator API started")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Multimodal Dubbing Evaluator",
    description="Reference-less voice-face compatibility scoring",
    version="1.0.0",
    lifespan=lifespan,
)


def _cleanup_expired_jobs() -> None:
    """Remove completed/failed jobs older than JOB_TTL_SECONDS and their files."""
    now = time.time()
    expired = [
        jid for jid, ts in job_timestamps.items()
        if now - ts > JOB_TTL_SECONDS
        and jid in jobs
        and jobs[jid].status in (JobStatus.COMPLETED, JobStatus.FAILED)
    ]
    for jid in expired:
        jobs.pop(jid, None)
        job_timestamps.pop(jid, None)
        job_dir = JOBS_DIR / jid
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
        logger.info(f"Cleaned up expired job {jid}")


def _run_evaluation(job_id: str, video_path: str) -> None:
    """
    Synchronous evaluation pipeline. Runs in a thread pool.
    """
    job_dir = JOBS_DIR / job_id

    def set_progress(p: int) -> None:
        jobs[job_id] = JobResponse(job_id=job_id, status=JobStatus.PROCESSING, progress=p)

    try:
        logger.info(f"[{job_id}] Starting evaluation for {video_path}")

        logger.info(f"[{job_id}] Extracting frames...")
        frames = extract_frames(video_path, fps=2.0)
        set_progress(1)

        logger.info(f"[{job_id}] Extracting audio...")
        audio_path = extract_audio(video_path, output_dir=str(job_dir))
        set_progress(2)

        logger.info(f"[{job_id}] Running face pipeline...")
        face_features = run_face_pipeline(frames)
        set_progress(3)

        logger.info(f"[{job_id}] Running audio pipeline...")
        voice_features = run_audio_pipeline(audio_path)
        set_progress(4)

        logger.info(f"[{job_id}] Computing scores...")
        result = compute_evaluation(face_features, voice_features)

        jobs[job_id] = JobResponse(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            progress=5,
            result=result,
        )
        logger.info(f"[{job_id}] Evaluation complete. Score: {result.overall_score}")

    except Exception as e:
        logger.error(f"[{job_id}] Evaluation failed: {e}")
        jobs[job_id] = JobResponse(
            job_id=job_id,
            status=JobStatus.FAILED,
            error=str(e),
        )


@app.post("/evaluate", response_model=JobResponse)
async def evaluate_video(file: UploadFile = File(...)):
    """
    Upload a video file for dubbing quality evaluation.
    Returns a job_id to poll for results.
    """
    if not file.filename or not file.filename.lower().endswith((".mp4", ".mkv", ".avi", ".mov")):
        raise HTTPException(400, "Unsupported video format. Use .mp4, .mkv, .avi, or .mov")

    # Clean up expired jobs before creating new ones
    _cleanup_expired_jobs()

    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job_timestamps[job_id] = time.time()

    # Sanitize filename to prevent path traversal attacks
    safe_name = Path(file.filename).name
    video_path = str(job_dir / safe_name)
    job_filenames[job_id] = safe_name
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Initialize job
    jobs[job_id] = JobResponse(job_id=job_id, status=JobStatus.PENDING)

    # Update to processing and launch background task
    jobs[job_id] = JobResponse(job_id=job_id, status=JobStatus.PROCESSING)

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_evaluation, job_id, video_path)

    return JobResponse(job_id=job_id, status=JobStatus.PROCESSING)


@app.get("/result/{job_id}", response_model=JobResponse)
async def get_result(job_id: str):
    """
    Poll for evaluation result by job_id.
    """
    if job_id not in jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    return jobs[job_id]


@app.get("/videos")
async def list_videos():
    """
    Return summary of all completed evaluation jobs for the dashboard.
    """
    result = []
    for job_id, job in jobs.items():
        if job.status == JobStatus.COMPLETED and job.result:
            r = job.result
            result.append({
                "job_id":       job_id,
                "filename":     job_filenames.get(job_id, "unknown"),
                "overall_score": r.overall_score,
                "confidence":   r.confidence,
                "components": {
                    "gender":  r.components.gender,
                    "age":     r.components.age,
                    "pitch":   r.components.pitch,
                    "formant": r.components.formant,
                },
            })
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/dashboard")
async def serve_dashboard():
    return FileResponse(ROOT_DIR / "dashboard.html")


@app.get("/")
async def serve_ui():
    return FileResponse(ROOT_DIR / "index.html")
