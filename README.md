# Multimodal Consistency Engine

Reference-less voice-face compatibility scoring for dubbed video.

Given a dubbed video, it answers: **does this voice plausibly belong to this face?**

No ground truth. No original audio. Pure cross-modal consistency.

---

## How it works

```
Video
  ├─► Frame extraction (cv2, 2 FPS)
  │       └─► Face Pipeline
  │               DeepFace → gender_dist, age_dist, confidence
  │
  └─► Audio extraction (ffmpeg, 16kHz mono)
          └─► Audio Pipeline
                  VAD → segments → pitch, F1, F2
                  wav2vec2 → gender_dist
                  → voice_confidence

Face features + Voice features
          └─► Scoring Engine
                  gender_score  (weight 0.35) — dot product of distributions
                  age_score     (weight 0.30) — dot product of distributions
                  pitch_score   (weight 0.20) — Gaussian plausibility
                  formant_score (weight 0.15) — F1/F2 Gaussian plausibility
                  → raw_score × confidence = overall_score
```

---

## Output

```json
{
  "overall_score": 0.61,
  "raw_score": 0.74,
  "confidence": 0.82,
  "components": {
    "gender": 0.91,
    "age": 0.68,
    "pitch": 0.72,
    "formant": 0.55
  }
}
```

| Field | Description |
|---|---|
| `overall_score` | Final reliability-weighted score (`raw × confidence`) |
| `raw_score` | Face-voice compatibility, ignoring signal quality |
| `confidence` | Trust in extracted features — `min(face_conf, voice_conf)` |
| `components` | Per-signal breakdown |

---

## Stack

| Layer | Tool |
|---|---|
| Face analysis | DeepFace |
| Gender classification | wav2vec2 (HuggingFace Transformers) |
| Pitch extraction | librosa / pyin |
| Formant extraction | Praat (parselmouth) |
| VAD | librosa energy-based |
| Backend | FastAPI (async, job-based) |
| Frontend | Vanilla HTML/CSS/JS |

---

## Setup

```bash
pip install -r requirements.txt
```

Requires **ffmpeg** on PATH.

```bash
uvicorn api.main:app --reload
```

Open `index.html` in a browser (or serve it statically).

---

## API

### `POST /evaluate`
Upload a video file. Returns `job_id`.

```bash
curl -X POST http://localhost:8000/evaluate \
  -F "file=@video.mp4"
```

### `GET /result/{job_id}`
Poll for result.

```bash
curl http://localhost:8000/result/{job_id}
```

Supported formats: `.mp4` `.mkv` `.avi` `.mov`

---

## Dashboard

The `index.html` dashboard shows:

- Face & audio gender/age distributions side by side
- Gender score breakdown (face dist · voice dist = similarity)
- Mismatch warning when face and voice predict different genders
- Consistency Signals panel (Strong / Moderate / Weak per component)
- Segment consistency indicator
- Confidence impact warning
- **Explain Score** toggle — shows the full math with actual values

---

## Design principles

- All scores are **probabilistic** (0–1), never binary
- Signals are **independent** — pitch does not feed gender, formants do not feed age
- Missing features return **0.5** (neutral) not 0 — absence of evidence ≠ evidence of absence
- Confidence penalizes low speech ratio, few segments, and weak face detection
