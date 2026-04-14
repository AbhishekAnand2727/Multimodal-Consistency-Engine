from pydantic import BaseModel, Field
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class GenderDist(BaseModel):
    male: float = Field(ge=0, le=1)
    female: float = Field(ge=0, le=1)

    def as_vector(self) -> list[float]:
        return [self.male, self.female]


class AgeDist(BaseModel):
    """Soft probability distribution over age buckets."""
    child: float = Field(ge=0, le=1, description="0-12")
    teen: float = Field(ge=0, le=1, description="13-19")
    adult: float = Field(ge=0, le=1, description="20-35")
    middle: float = Field(ge=0, le=1, description="35-55")
    senior: float = Field(ge=0, le=1, description="55+")

    def as_vector(self) -> list[float]:
        return [self.child, self.teen, self.adult, self.middle, self.senior]


class FaceFeatures(BaseModel):
    gender_dist: GenderDist
    age_dist: AgeDist
    confidence: float = Field(ge=0, le=1)


class VoiceFeatures(BaseModel):
    gender_dist: GenderDist
    age_dist: AgeDist
    pitch_mean: float
    f1_mean: float
    f2_mean: float
    confidence: float = Field(ge=0, le=1)


class ComponentScores(BaseModel):
    gender: float = Field(ge=0, le=1)
    age: float = Field(ge=0, le=1)
    pitch: float = Field(ge=0, le=1)
    formant: float = Field(ge=0, le=1)


class EvaluationResult(BaseModel):
    raw_score: float = Field(ge=0, le=1)
    overall_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    components: ComponentScores
    face_features: FaceFeatures | None = None
    voice_features: VoiceFeatures | None = None


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: EvaluationResult | None = None
    error: str | None = None
