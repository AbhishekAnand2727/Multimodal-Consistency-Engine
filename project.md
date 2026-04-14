# Multimodal Dubbing Evaluator — Complete Technical Documentation

**Version:** 2.0
**Last updated:** 2026-04-14

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Preprocessing](#2-preprocessing)
3. [Face Pipeline](#3-face-pipeline)
4. [Audio Pipeline](#4-audio-pipeline)
5. [Scoring Engine](#5-scoring-engine)
6. [Final Score Computation](#6-final-score-computation)
7. [Evaluation Framework](#7-evaluation-framework)
8. [Edge Cases and Failure Modes](#8-edge-cases-and-failure-modes)
9. [Design Principles](#9-design-principles)
10. [API Reference](#10-api-reference)

---

## 1. System Overview

### 1.1 What the system does

The Multimodal Dubbing Evaluator scores how **natural a dubbed audio track sounds for the visible speaker** in a video. Given a dubbed video, it asks: does this voice plausibly belong to this face?

It produces a single scalar score in `[0, 1]`:
- `1.0` — voice and face are highly compatible
- `0.5` — no evidence either way (neutral / uncertain)
- `0.0` — voice and face are highly incompatible

### 1.2 What "reference-less" means

Most audio quality systems require a **reference** — a ground-truth recording of how the audio *should* sound (e.g., the original un-dubbed voice). This system has no such reference. It evaluates **absolute compatibility** between two independent signals:

- What does the **face** tell us about the expected speaker? (gender, age)
- What does the **voice** actually sound like? (gender, age, pitch, formants)

No ground truth. No original audio. Just cross-modal consistency.

### 1.3 Input and output

**Input:** A single video file (`.mp4`, `.mkv`, `.avi`, `.mov`)

**Output:**
```json
{
  "raw_score": 0.74,
  "overall_score": 0.61,
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
| `raw_score` | Pure compatibility — how well face and voice match, ignoring signal quality |
| `overall_score` | Reliability-weighted output: `raw_score × confidence` |
| `confidence` | How much we trust the extracted features (0 = no data, 1 = full trust) |
| `components` | Breakdown by scoring signal |

### 1.4 High-level pipeline

```
Video
  │
  ├─► Frame extraction (cv2)
  │       │
  │       └─► Face Pipeline
  │               DeepFace → gender_dist, age_dist, face_confidence
  │
  └─► Audio extraction (ffmpeg)
          │
          └─► Audio Pipeline
                  VAD → segmentation → per-segment features
                  → voice_confidence
                  gender_dist, age_dist, pitch_mean, F1, F2

Face features + Voice features
          │
          └─► Scoring Engine
                  → gender_score, age_score, pitch_score, formant_score
                  → raw_score, confidence, overall_score
```

---

## 2. Preprocessing

### 2.1 Frame extraction

**File:** `utils/preprocessing.py` — `extract_frames()`

Video frames are extracted using OpenCV at a target rate of **2 FPS** (configurable). Sampling at 2 FPS rather than the native frame rate (typically 24–30 FPS) avoids redundant computation while still capturing face appearance changes over time.

**Logic:**

```python
frame_interval = max(1, round(video_fps / target_fps))
```

For a 30 FPS video at target 2 FPS:
```
frame_interval = round(30 / 2) = 15
→ sample every 15th frame
→ ~2 frames per second
```

Every `frame_interval`-th frame is appended to the output list. The result is a list of BGR `numpy.ndarray` frames in OpenCV format.

**Resource safety:** The `VideoCapture` object is always released in a `try/finally` block, even if frame reading raises an exception:

```python
cap = cv2.VideoCapture(video_path)
try:
    ...
finally:
    cap.release()
```

### 2.2 Audio extraction

**File:** `utils/preprocessing.py` — `extract_audio()`

Audio is extracted using **ffmpeg** with fixed settings:

| Parameter | Value | Reason |
|---|---|---|
| Sample rate | 16 kHz | Standard for speech models (wav2vec2, pyin) |
| Channels | Mono | Reduces complexity; speech models expect mono |
| Codec | PCM 16-bit | Lossless, widely compatible |
| Output | WAV | Raw format — no codec artifacts |

The ffmpeg command:
```bash
ffmpeg -y -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav
```

If ffmpeg returns a non-zero exit code, the error is raised immediately — no silent failures.

---

## 3. Face Pipeline

**File:** `pipelines/face_pipeline.py`

The face pipeline processes a list of frames and aggregates **gender** and **age** distributions across all frames with valid face detections.

### 3.1 Model: DeepFace

Each frame is analyzed with `DeepFace.analyze()` requesting:
- `age` — predicted age in years (scalar)
- `gender` — `{"Man": probability, "Woman": probability}` (percentages 0–100)

Settings:
- `enforce_detection=False` — returns a result even if detection confidence is low (handled downstream)
- `silent=True` — suppresses progress output

A frame is discarded if `face_confidence < 0.5`. This removes frames where DeepFace found no clear face, preventing noisy low-quality detections from polluting the aggregate.

### 3.2 Gender distribution

For each valid frame, DeepFace returns percentages that are normalized to `[0, 1]` probabilities:

```python
p_male   = gender_raw["Man"]   / 100.0
p_female = gender_raw["Woman"] / 100.0
```

Across all valid frames, the per-frame probabilities are averaged:

```python
mean_male   = mean([p_male_frame_1, p_male_frame_2, ...])
mean_female = mean([p_female_frame_1, ...])
```

Then re-normalized so they sum to 1:
```
mean_male   /= (mean_male + mean_female)
mean_female /= (mean_male + mean_female)
```

This gives a **soft gender distribution**: e.g., `{male: 0.87, female: 0.13}` for a clearly male face, or `{male: 0.52, female: 0.48}` for an ambiguous face.

### 3.3 Age distribution (CDF-based Gaussian smoothing)

A point age prediction (e.g., `age = 34`) is converted to a **soft probability distribution** over five age buckets:

| Bucket | Range |
|---|---|
| child | 0–12 |
| teen | 13–19 |
| adult | 20–35 |
| middle | 35–55 |
| senior | 55–100 |

**Method:** For each bucket, compute the probability mass that a Gaussian centered at the predicted age places within the bucket's boundaries, using the CDF:

```
P(bucket_i) = CDF(upper_i, μ=age, σ=8) − CDF(lower_i, μ=age, σ=8)
```

This integrates over the full bucket width rather than sampling a single point, giving accurate probability mass even for wide buckets (e.g., "senior" spans 45 years). The distribution is then normalized to sum to 1.

**Why σ=8?** The face pipeline uses σ=8 years because DeepFace's age estimate is reasonably accurate (within ~5–10 years), so the distribution should be moderately peaked.

**Example:** Predicted age = 34, σ = 8:
- child (0–12): ≈ 0.00
- teen (13–19): ≈ 0.01
- adult (20–35): ≈ 0.52
- middle (35–55): ≈ 0.46
- senior (55+): ≈ 0.01

### 3.4 Aggregation across frames

Multiple frames produce multiple age predictions. Each prediction is converted to a distribution, then averaged:

```python
age_dists = [age_to_distribution(age) for age in ages]
mean_age_dist = mean(age_dists, axis=0)
mean_age_dist /= mean_age_dist.sum()  # normalize
```

This averaging is equivalent to **Bayesian opinion pooling**: each frame casts a "vote" in distribution space, and the result reflects the consensus across the full video.

### 3.5 Face confidence

Face confidence reflects how much we trust the extracted face features. Three sub-signals:

**1. Detection ratio** — fraction of frames with valid face detections:
```
detection_ratio = n_valid_frames / n_total_frames
```

**2. Gender stability** — consistency of gender prediction across frames. High variance means the face appearance is ambiguous or the video switches speakers:
```
gender_std = std([p_male_frame_1, p_male_frame_2, ...])
gender_stability = max(0, 1.0 − 2.0 × gender_std)
```
A std of 0 (perfectly consistent) gives stability = 1.0. A std of 0.5 (random) gives stability = 0.

**3. Age stability** — age predictions normalized by a 20-year reference span:
```
age_std = std([age_frame_1, age_frame_2, ...])
age_stability = max(0, 1.0 − age_std / 20.0)
```

**Final formula:**
```python
confidence = 0.50 × detection_ratio
           + 0.25 × gender_stability
           + 0.25 × age_stability
```

Confidence is clipped to `[0, 1]`.

---

## 4. Audio Pipeline

**File:** `pipelines/audio_pipeline.py`, `utils/audio_features.py`

The audio pipeline extracts four independent features from speech segments: gender classification, age heuristic, fundamental frequency (pitch), and formants (F1/F2).

### 4.1 Voice Activity Detection (VAD)

**File:** `utils/audio_features.py` — `apply_vad()`

Before feature extraction, speech regions are isolated using energy-based VAD.

**Step 1:** Compute frame-level RMS energy using librosa:
```python
rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)
```
Each frame covers ~128 ms of audio at 16 kHz.

**Step 2:** Adaptive thresholding. A fixed threshold fails on low-amplitude audio (e.g., quiet recordings where the RMS of speech is below the typical "silence" floor). Instead:

```python
adaptive_threshold = max(
    energy_threshold,           # absolute floor (default 0.01)
    percentile(rms, 30) * 3.0  # data-driven: 3× the 30th percentile
)
speech_frames = rms > adaptive_threshold
```

The 30th percentile represents the quieter end of the recording — silence or background noise. Multiplying by 3.0 ensures only frames clearly above the noise floor are marked as speech.

**Step 3:** The frame-level boolean mask is expanded to sample-level and multiplied with the audio:
```python
mask = repeat(speech_frames, hop_length)[:len(audio)]
audio_clean = audio * mask
```

**Output:** Cleaned audio with silences zeroed out, plus `speech_ratio` — the fraction of frames classified as speech.

### 4.2 Segmentation

**File:** `utils/audio_features.py` — `segment_audio()`

The cleaned audio is split into segments of **2–5 seconds** for per-segment analysis. Target segment length is 3 seconds.

**Logic:**
```
while audio remaining:
    if remaining < min_samples (2s):
        merge tail into last segment, clamped to max_samples (5s)
        break
    seg_len = min(target_samples, remaining, max_samples)
    append segment
    advance offset
```

The merge-and-clamp on the tail prevents a 1.5s leftover from being wasted (too short for formant/pitch extraction) while ensuring the merged segment never exceeds the 5s maximum. This maximum matters because very long segments degrade Praat's formant estimation accuracy.

**Why 2–5 seconds?** This range balances:
- Minimum: librosa's pyin and Praat need ≥2s for reliable pitch/formant extraction
- Maximum: Longer segments average over more temporal variation, reducing sensitivity to short-duration events

### 4.3 Per-segment feature extraction

**File:** `pipelines/audio_pipeline.py` — `_analyze_segment()`

Each segment goes through four **independent** extractors:

#### 4.3.1 Gender classification (wav2vec2)

Model: `alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech`

This is a fine-tuned wav2vec2 model that classifies segments as male or female. It operates on raw waveforms (no spectrogram needed) and is loaded once via `@lru_cache` singleton.

```python
result = classifier({"raw": segment, "sampling_rate": 16000})
# Returns: [{"label": "male", "score": 0.93}, {"label": "female", "score": 0.07}]
```

Scores are re-normalized so they sum exactly to 1. If classification fails (model error, too-short segment), the fallback is `{male: 0.5, female: 0.5}` — neutral, no gender claim.

**Critical independence contract:** This signal uses ONLY the raw waveform content. It does NOT use pitch, formants, or any other derived feature. This prevents double-counting.

#### 4.3.2 Pitch extraction (librosa pyin)

Fundamental frequency (F0) is extracted using the probabilistic YIN (pYIN) algorithm:

```python
f0, voiced_flag, _ = librosa.pyin(
    segment,
    fmin=65 Hz,     # C2 — below typical speech range
    fmax=2093 Hz,   # C7 — above typical speech range
    sr=16000,
)
```

Only voiced frames (where `voiced_flag=True`) contribute to the pitch estimate. Outliers outside `[50, 500]` Hz are discarded (non-speech sounds). The median of remaining F0 values is returned.

**Why median?** Median is more robust than mean to transient pitch errors in unvoiced transitions.

If no voiced frames remain: returns `None`, which is stored as `0.0` — the sentinel value meaning "pitch unavailable".

#### 4.3.3 Formant extraction (Praat via parselmouth)

Formant frequencies F1 and F2 are extracted using the Burg LPC method via Praat:

```python
formant = call(snd, "To Formant (burg)",
    0.0,              # time step (0 = auto)
    5,                # max formants
    ceiling_hz,       # formant ceiling (gender-adaptive)
    0.025,            # window length (25ms)
    50,               # pre-emphasis from (Hz)
)
```

**Gender-adaptive ceiling:** The formant ceiling is a critical parameter. Setting it too low causes formants to be merged; too high introduces spurious high-frequency formants.

```python
formant_ceiling = 5000.0 if p_male > p_female else 5500.0
```

Males have shorter vocal tracts, placing formants at lower absolute frequencies. The standard 5000 Hz ceiling is appropriate for male speakers; 5500 Hz is standard for female speakers.

Valid F1 values: `200–1000 Hz`
Valid F2 values: `500–3500 Hz`

The median of valid frames is returned for each. If no valid frames exist: `None` (stored as `0.0`).

#### 4.3.4 Age heuristic (spectral centroid)

A rough age estimate is derived from the **spectral centroid** — the "center of mass" of the spectrum:

```python
centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
sc_mean = median(centroid[centroid > 0])
```

Younger speakers (especially children) have shorter, higher-resonating vocal tracts, producing spectra with energy concentrated at higher frequencies. The mapping:

| Spectral centroid (Hz) | Estimated age |
|---|---|
| > 2600 | 15 (teen) |
| 2300–2600 | 25 (young adult) |
| 2000–2300 | 38 (adult) |
| 1700–2000 | 50 (middle age) |
| < 1700 | 62 (senior) |

This estimate is intentionally coarse. It is converted to a distribution using σ=15 years (much wider than the face pipeline's σ=8), reflecting the low reliability of audio-based age estimation compared to visual.

**Critical independence contract:** This uses ONLY the spectral centroid — not pitch (F0) or formants (F1/F2). No double-counting with other signals.

### 4.4 Weighted aggregation

All per-segment features are aggregated using **RMS-energy-weighted means**. The RMS energy of each segment serves as its weight — louder (more speech-rich) segments are trusted more than quiet ones.

```python
weight_i = sqrt(mean(segment_i²))
```

**Gender aggregation:**
```python
mean_male = weighted_mean([p_male_i], [weight_i])
```
Then re-normalized to sum to 1.

**Pitch aggregation:**
```python
pitch_mean = weighted_mean([pitch_i], [weight_i])
            # only segments where pitch is available
```

**Formant aggregation:** F1 and F2 are tracked independently. A segment where F1 is unavailable does not affect the F2 aggregate.

**Age aggregation:** Each segment's age estimate is converted to a 5-bucket distribution, then weighted-averaged:
```python
mean_age_dist = weighted_dist_mean([dist_i], [weight_i])
```

### 4.5 Voice confidence

Voice confidence reflects how much we trust the extracted voice features.

**Components:**

```
1. speech_score    = min(speech_ratio / 0.5, 1.0)
                   # saturates at 50% speech — more speech = more signal

2. segment_score   = n_valid_segments / n_total_segments
                   # fraction of segments that produced any results

3. feature_score   = (gender_avail + pitch_avail + formant_avail) / 3.0
                   # average feature extraction success rate

4. consistency     = max(0, 1.0 − 2.0 × std(p_male across segments))
                   # gender prediction stability
```

**Final formula:**
```python
confidence = 0.25 × speech_score
           + 0.20 × segment_score
           + 0.30 × feature_score
           + 0.25 × consistency
```

Feature availability (component 3) has the highest weight (0.30) because missing features directly degrade the scoring engine's accuracy. A video with no detectable pitch and no formants should score low confidence even if it has clear speech.

---

## 5. Scoring Engine

**File:** `pipelines/scoring.py`

The scoring engine takes `FaceFeatures` and `VoiceFeatures` and computes four independent component scores. **No feature feeds into another scorer.** Gender score uses only gender distributions; pitch score uses only pitch and face-gender; formant score uses only formants and face-gender.

### 5.1 Gender score

**Signal:** Face gender distribution vs voice gender distribution
**Method:** Dot product

```
gender_score = face.gender_dist · voice.gender_dist
             = face.p_male × voice.p_male + face.p_female × voice.p_female
```

**Intuition:** If the face is clearly male (`{male: 0.95, female: 0.05}`) and the voice is also clearly male (`{male: 0.92, female: 0.08}`):
```
score = 0.95 × 0.92 + 0.05 × 0.08 = 0.874 + 0.004 = 0.878
```

If the face is male but the voice is female:
```
score = 0.95 × 0.08 + 0.05 × 0.92 = 0.076 + 0.046 = 0.122
```

The dot product naturally handles uncertainty: if either distribution is near uniform (50/50), the score is pulled toward 0.5 regardless of the other signal — exactly the right behavior.

**Range:** `[0, 1]`. Maximum when both distributions are identical and peaked; minimum when they are perfectly opposite; 0.5 when either is uniform.

### 5.2 Age score

**Signal:** Face age distribution vs voice age distribution
**Method:** Dot product (same as gender)

```
age_score = face.age_dist · voice.age_dist
          = Σ_i (face.bucket_i × voice.bucket_i)
```

Both distributions are over the same five buckets (child, teen, adult, middle, senior), so the dot product measures how much probability mass they share — i.e., how similar the implied age range is.

**Asymmetry awareness:** The face pipeline uses σ=8; the audio pipeline uses σ=15. This means the audio distribution is always flatter (more spread). A flat audio distribution produces a naturally lower dot product with a peaked face distribution, which correctly reflects the lower reliability of audio-based age estimation.

### 5.3 Pitch score

**Signal:** Voice pitch (`pitch_mean`) vs face gender expectation
**Method:** Gaussian plausibility

Pitch is scored by asking: how probable is this pitch value under the expected distribution for this face's apparent gender?

**Reference ranges:**

| Parameter | Male | Female |
|---|---|---|
| Center (Hz) | 132.5 | 210.0 |
| Sigma (Hz) | 35.0 | 30.0 |

These ranges are based on speech science literature for adult speakers:
Male typical: 85–180 Hz; Female typical: 165–255 Hz.

**Gaussian plausibility:**
```python
def gaussian_score(value, center, sigma):
    z = (value - center) / sigma
    return exp(−0.5 × z²)
```
Returns 1.0 at the center, falls off symmetrically with `sigma`.

**Blending by face gender probability:**
```python
male_fit    = gaussian_score(pitch, 132.5, 35.0)
female_fit  = gaussian_score(pitch, 210.0, 30.0)
pitch_score = p_male × male_fit + p_female × female_fit
```

This blending is important for ambiguous faces. If a face is 60% male / 40% female, the expected pitch distribution is a 60/40 mixture of male and female ranges, so a pitch of 150 Hz is reasonably plausible even though it sits between the two centers.

**Unavailability:** If `pitch_mean == 0` (sentinel for "unavailable"), returns `NEUTRAL_SCORE = 0.5`. This neither inflates nor deflates the final score.

### 5.4 Formant score

**Signal:** Voice F1/F2 frequencies vs face gender expectation
**Method:** Gaussian plausibility (same approach as pitch)

**Reference ranges:**

| Parameter | Male center | Male σ | Female center | Female σ |
|---|---|---|---|---|
| F2 (Hz) | 1100 | 200 | 1600 | 250 |
| F1 (Hz) | 400 | 80 | 500 | 100 |

F2 is the primary discriminator because it is most sensitive to vocal tract length differences between sexes. F1 (jaw opening / vowel height) is a supporting signal.

**Scoring:**
```python
f2_score = p_male × gaussian(F2, 1100, 200) + p_female × gaussian(F2, 1600, 250)
f1_score = p_male × gaussian(F1, 400, 80)   + p_female × gaussian(F1, 500, 100)
```

**Combining F1 and F2:**
```
if both available:   score = 0.7 × f2_score + 0.3 × f1_score
if only F2:          score = f2_score
if only F1:          score = f1_score    # weak signal — low confidence penalizes this
if neither:          score = NEUTRAL_SCORE (0.5)
```

F2 carries 70% of the weight when both are available because it is more reliably sex-differentiated than F1.

---

## 6. Final Score Computation

**File:** `pipelines/scoring.py` — `compute_evaluation()`

### 6.1 Raw score (weighted sum)

```python
raw_score = (
    0.35 × gender_score  +
    0.30 × age_score     +
    0.20 × pitch_score   +
    0.15 × formant_score
)
```

**Weight rationale:**

| Component | Weight | Reason |
|---|---|---|
| gender | 0.35 | Highest perceptual salience — gender mismatch is immediately obvious |
| age | 0.30 | Strong perceptual cue — a child face with an elderly voice is jarring |
| pitch | 0.20 | Physical signal directly tied to vocal cord vibration frequency |
| formant | 0.15 | More technical — vocal tract shape, less directly audible |

Weights sum to 1.0. The raw score is a pure **compatibility score** — it measures how well face and voice match, independent of how much data was available.

### 6.2 Confidence

```python
confidence = min(face.confidence, voice.confidence)
```

Taking the minimum means the final confidence is limited by the weaker of the two pipelines. If the face is clear but the audio is degraded, confidence will be low. This is conservative by design: if we can't trust either signal, we can't trust the result.

### 6.3 Final score (reliability-weighted)

```python
overall_score = raw_score × confidence
```

This collapses the compatibility and reliability dimensions into a single deployable score. A highly compatible pair with low-quality data outputs a moderate score, reflecting real uncertainty.

### 6.4 raw_score vs overall_score — why both matter

| Field | Question it answers | When to use |
|---|---|---|
| `raw_score` | "Do face and voice match, assuming we trust the data?" | Comparing system calibration; proxy error calculations |
| `overall_score` | "How confident should we be in this compatibility assessment?" | Production output; ranking dubs; user-facing display |

**Critical note for analysis:** When computing how far a prediction deviates from an expected label score (the "proxy error"), always use `raw_score`, not `overall_score`. Using `overall_score` would conflate two effects:
1. Whether the system correctly assessed compatibility
2. Whether the data quality was sufficient

For example: a good dub with poor video quality might have `raw_score = 0.82` (correct) but `overall_score = 0.41` (penalized by low confidence). If you compare `overall_score` to the expected value for "good" (0.85), you get a large apparent error — but the scoring engine was correct; the data was just bad.

---

## 7. Evaluation Framework

The evaluation framework measures how well the scoring system separates labeled dubs into "good", "acceptable", and "bad" categories.

### 7.1 Dataset format

Each labeled sample is a JSON entry:
```json
{"video_path": "path/to/video.mp4", "label": "good", "notes": "optional"}
```

Labels must be one of: `good`, `acceptable`, `bad`.

After evaluation, each sample gets an `EvalResult`:
```python
EvalResult(
    video_path, label,
    raw_score, overall_score, confidence,
    components,   # {gender, age, pitch, formant}
    face_confidence, voice_confidence,
    error          # None if successful
)
```

### 7.2 Score distributions

For each label, compute descriptive statistics:
- mean, std, min, max
- P10, Q25, median, Q75, P90

The **P10–P90 range** (the 80th percentile spread) is used instead of full min/max for overlap analysis. This is more robust to outliers — an extreme bad dub scoring coincidentally high (rare event) should not inflate the apparent overlap.

### 7.3 Label ordering validation

Before any analysis, the system checks whether the observed score ordering matches expectation:
```
expected: good > acceptable > bad (by median)
```

If this ordering is violated (e.g., bad scores higher than acceptable), a warning is logged and **all downstream analysis uses the observed ordering**, not the assumed one. This prevents incorrect threshold placement in poorly-calibrated early iterations.

### 7.4 Overlap analysis

For each adjacent class pair (good/acceptable and acceptable/bad), the overlap is computed using P10–P90 ranges:

```
overlap_low  = max(p10_class_a, p10_class_b)
overlap_high = min(p90_class_a, p90_class_b)
overlap_width = max(0, overlap_high − overlap_low)
```

**Severity:**

| Ratio of overlap to narrower spread | Severity |
|---|---|
| < 5% | none |
| 5–20% | mild |
| 20–45% | significant |
| > 45% | severe |

**Separation** (mean_a − mean_b) is reported alongside overlap. A system can have zero overlap but a small separation — or vice versa.

### 7.5 Pairwise ranking analysis

**File:** `evaluation/ranking.py`

This is the most important evaluation metric. It asks: for every pair of samples where one has a higher-quality label, does the system correctly give it a higher score?

**Method:** Exhaustive pairwise comparison across all label-ordered pairs.

For labels ordered `[good, acceptable, bad]`, comparable pairs are:
- good vs acceptable
- good vs bad
- acceptable vs bad

For each comparable pair `(r_high, r_low)`:
```
margin = r_high.overall_score − r_low.overall_score
```

- `margin > tie_margin`: correct ranking
- `margin < −tie_margin`: incorrect ranking
- `|margin| ≤ tie_margin`: tie (excluded from accuracy)

**Adaptive tie margin:**
```
tie_margin = 0.01 × score_range
```
Scales with the observed spread of scores — prevents an arbitrary fixed value from being too loose or too strict.

**This is equivalent to a normalized Mann-Whitney U statistic:**
`accuracy = 0.5` means the system ranks randomly; `1.0` means perfect ordering.

### 7.6 Margin analysis

Beyond binary accuracy, margins measure *how confidently* rankings are correct.

**Raw margin statistics:** Mean, median, standard deviation of `(score_high − score_low)`.

**Normalized margins:** Divided by `score_range` for scale invariance:
```
normalized_margin = margin / score_range
```
This allows comparison across datasets with different score distributions.

**Margin-weighted accuracy (raw):**
```
weighted_accuracy = sum(margin if correct else 0) / sum(|margin|)
```
Large correct margins raise the score; large incorrect margins lower it more than small ones.

**Clipped margin-weighted accuracy:** Same formula, but margins are clipped at ±0.3:
```
clipped_margin = sign(margin) × min(|margin|, 0.3)
```
Prevents a handful of extreme pairs from dominating the score. Two samples that score 0.95 vs 0.05 shouldn't count for more than two samples scoring 0.70 vs 0.40 by a factor of 9×.

### 7.7 Confidence-weighted accuracy

Pairs are weighted by the minimum confidence of the two samples:
```
pair_weight = min(confidence_high, confidence_low)
conf_weighted_accuracy = Σ(weight × |norm_margin| if correct) / Σ(weight × |norm_margin|)
```

**What this measures:** Does the system make its correct rankings on the samples it is *most confident about*? If `conf_weighted_accuracy >> accuracy`, confidence is tracking correctness well — high-confidence judgments are more reliable. If `conf_weighted_accuracy << accuracy`, something is wrong: the system is most confident on pairs it gets wrong.

### 7.8 Bootstrap stability

100 bootstrap resamples of the valid result pool (with replacement) are used to estimate:
- `mean_accuracy` — stable estimate of ranking accuracy
- `std_accuracy` — how much accuracy varies across resamples
- `high_variance = (std_accuracy > 0.10)` — if true, the dataset is too small or class distributions too overlapping for reliable conclusions

This directly answers: "Can I trust these ranking numbers, or are they just noise?"

### 7.9 Component breakdown and separation

For each component (gender, age, pitch, formant), compute:
```
separation = mean_score_for_good − mean_score_for_bad
```

High separation means the component correctly assigns higher scores to good dubs than bad ones — it is pulling its weight in the final formula.
Low or negative separation means the component is uninformative or counterproductive.

### 7.10 Calibration report

`generate_report()` produces a human-readable text report covering:
1. Dataset summary and label counts
2. Score distributions with P10–P90
3. Label ordering validation
4. Recommended decision thresholds (midpoints between adjacent class medians)
5. Component ranking by discriminative power
6. Confidence behavior (dual correlation)
7. Full ranking analysis with bootstrap stability
8. Weight adjustment suggestions
9. Actionable recommendations with severity flags

### 7.11 Score normalization

Two normalization strategies are available:

**Min-max:**
```
normalized = (score − min) / (max − min)
```
Stretches observed range to [0, 1]. Good when absolute score values are not interpretable.

**Sigmoid:**
```
normalized = 1 / (1 + exp(−(score − median) × scale))
```
Centered at median, compresses extremes. Good for skewed distributions.

### 7.12 Reliability adjustment

A smooth confidence-weighted score adjustment for downstream use:
```
adjusted_score = confidence × raw_score + (1 − confidence) × 0.5
```

This continuously pulls low-confidence scores toward the neutral midpoint. No samples are dropped — every result gets an adjusted score.

| Flag | Condition | Effect |
|---|---|---|
| `ok` | confidence ≥ threshold (0.4) | Score mostly unchanged |
| `uncertain` | confidence ≥ threshold/2 | Score noticeably pulled toward 0.5 |
| `low_confidence` | confidence < threshold/2 | Score heavily pulled toward 0.5 |

### 7.13 Weight adjustment suggestions

Based on observed component separation scores, the system can suggest revised scoring weights:

1. Floor negative separations to `1e-3` (a badly-performing component keeps a small weight rather than being zeroed — it may improve)
2. Compute separation-proportional weights
3. Blend 50/50 with default weights (preserves domain priors, prevents over-reaction to noisy estimates)
4. Renormalize to sum to 1.0

```
adjusted_weight_c = (0.5 × default_weight_c + 0.5 × (sep_c / total_sep))
                    normalized to sum to 1
```

---

## 8. Edge Cases and Failure Modes

### 8.1 No speech detected

**Condition:** `speech_ratio < 0.05` after VAD.

**Behavior:** Returns `_empty_voice_features()`:
```python
VoiceFeatures(
    gender_dist={male: 0.5, female: 0.5},
    age_dist={uniform},
    pitch_mean=0.0, f1_mean=0.0, f2_mean=0.0,
    confidence=0.0
)
```

With `voice.confidence = 0.0`, `confidence = min(face_conf, 0.0) = 0.0`, so `overall_score = raw_score × 0 = 0.0`. The system outputs zero, signaling complete unreliability rather than a false compatibility claim.

### 8.2 No face detected

**Condition:** All frames fail detection or `face_confidence < 0.5`.

**Behavior:** Returns `FaceFeatures` with uniform distributions and `confidence = 0.0`. Same outcome as no speech: `overall_score = 0.0`.

### 8.3 Short audio (< 0.5 seconds)

**Condition:** `duration < 0.5` seconds after loading.

**Behavior:** Returns `_empty_voice_features()` immediately. Pitch and formant extraction require at least 2 seconds of audio — attempting extraction on shorter clips would produce unreliable values.

### 8.4 Features unavailable individually

Features can be missing without failing the whole pipeline:

| Missing | Effect | Score impact |
|---|---|---|
| pitch_mean = 0 | `score_pitch()` returns 0.5 | Neutral — no contribution |
| F2 unavailable, F1 available | formant score uses F1 alone | Weakened but not absent |
| Neither F1 nor F2 | `score_formant()` returns 0.5 | Neutral |
| Gender unavailable | Fallback to {male: 0.5, female: 0.5} | Near-neutral |

All missing features also reduce `voice_confidence` via the feature availability component.

### 8.5 Multiple speakers

The pipeline has no speaker diarization. If the video contains multiple speakers:
- Face pipeline: averages over the dominant face detected in each frame
- Audio pipeline: aggregates over all speech, including off-screen speakers

**Effect:** Scores become unreliable. The confidence signal will partially reflect this (if the dominant detected face is inconsistent across frames, `gender_stability` drops). However, the system cannot distinguish "wrong gender because of bad dubbing" from "multiple speakers present."

**Recommendation:** Apply diarization upstream before evaluating; or use the system only on single-speaker segments.

### 8.6 Low-amplitude audio

**Pre-fix problem:** A fixed RMS threshold of 0.01 classified nearly all frames as silence for quiet recordings.

**Fix:** Adaptive VAD threshold (`max(fixed_floor, percentile_30 × 3.0)`) ensures the threshold scales with the recording's actual energy level.

---

## 9. Design Principles

### 9.1 Probabilistic modeling — no hard thresholds in feature extraction

Every feature is expressed as a **distribution or a continuous score**, never as a binary label. DeepFace gender output becomes `{male: 0.87, female: 0.13}`, not "male". Age becomes a 5-bucket probability distribution, not "adult".

**Why:** Hard thresholds introduce discontinuities — a face that is 49% male and 51% female would be treated identically to a 1% male / 99% female face if both are classified as "female". Soft distributions preserve the uncertainty information downstream.

### 9.2 Signal independence — zero double-counting

Each scoring component uses exactly one type of information:

| Component | Uses | Does NOT use |
|---|---|---|
| gender_score | gender distributions only | pitch, formants, age |
| age_score | age distributions only | pitch, formants, gender |
| pitch_score | pitch_mean + face gender | formants, voice gender |
| formant_score | F1/F2 + face gender | pitch, voice gender |

Pitch is NOT used to infer gender. Formants are NOT used to estimate age. This independence means each component contributes genuinely new information to the final score.

**Why:** If pitch were used to classify voice gender, and the gender score then used that classification, a single physical measurement (pitch) would be counted twice under two different names — inflating the apparent reliability of gender-related scoring.

### 9.3 Neutral fallback for missing data — never fabricate

When a feature cannot be extracted, the score falls back to `NEUTRAL_SCORE = 0.5` — the midpoint. This has no impact on the final score (it contributes 0 after subtracting the neutral prior), and it does NOT invent a value.

The alternative — imputing "typical" values — would fabricate evidence, causing the system to express false confidence. For example, imputing a typical male pitch for an audio segment where pitch extraction failed would make the pitch score look good for a male face, when in reality we have no pitch data.

### 9.4 Confidence as reliability modifier — not as filter

Confidence does not gate the output (no samples are dropped for low confidence). Instead, it smoothly scales the output:

```
overall_score = raw_score × confidence
```

A raw score of 0.8 with confidence 0.3 yields 0.24 — signaling "this looks compatible but we can't trust the data".

**Why not filter?** Filtering creates a bimodal output — either a score or nothing. A downstream system (e.g., a flagging system) would then need to handle "no output" as a special case. A continuous score is always interpretable.

### 9.5 Extend, don't rewrite — module independence

Each pipeline module (`face_pipeline.py`, `audio_pipeline.py`, `scoring.py`) is independently testable and can be swapped without modifying the others. The scoring engine only sees `FaceFeatures` and `VoiceFeatures` schemas — it doesn't care how they were computed. A face pipeline using a different model (e.g., InsightFace instead of DeepFace) is a drop-in replacement.

### 9.6 Pairwise ranking over classification accuracy

The evaluation system measures **ranking accuracy** (does score_good > score_bad?) rather than binary classification accuracy (does score_good > threshold?).

**Why:** Thresholds are dataset-dependent and change during system development. Ranking is threshold-free — it measures the fundamental ordering property that any useful scoring system must have, regardless of the absolute scale of the scores.

---

## 10. API Reference

**File:** `api/main.py`

The system exposes a job-based REST API built with FastAPI.

### POST `/evaluate`

Upload a video file for evaluation. Returns immediately with a job ID.

**Request:** `multipart/form-data` with field `file` (video file)

**Supported formats:** `.mp4`, `.mkv`, `.avi`, `.mov`

**Response:**
```json
{"job_id": "uuid-string", "status": "processing", "result": null, "error": null}
```

**Security:** The uploaded filename is sanitized to prevent path traversal:
```python
safe_name = Path(file.filename).name  # strips any directory components
video_path = str(job_dir / safe_name)
```

**Async execution:** The evaluation runs in a thread pool executor so the API remains responsive:
```python
loop = asyncio.get_running_loop()
loop.run_in_executor(None, _run_evaluation, job_id, video_path)
```

### GET `/result/{job_id}`

Poll for evaluation result.

**Response (pending):**
```json
{"job_id": "...", "status": "processing", "result": null, "error": null}
```

**Response (completed):**
```json
{
  "job_id": "...",
  "status": "completed",
  "result": {
    "raw_score": 0.74,
    "overall_score": 0.61,
    "confidence": 0.82,
    "components": {"gender": 0.91, "age": 0.68, "pitch": 0.72, "formant": 0.55}
  }
}
```

### GET `/health`

Returns `{"status": "ok"}`. Used for liveness checks.

### Job cleanup

Completed and failed jobs older than **1 hour** are automatically deleted on the next upload request. This includes:
- In-memory job data
- Job files on disk (`jobs/<job_id>/`)

---

*End of documentation*
