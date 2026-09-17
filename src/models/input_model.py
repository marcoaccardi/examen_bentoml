"""Pydantic models shared by the API service and the workers."""

from pydantic import BaseModel, Field

# The feature order used at training time - predictions are positional.
FEATURE_ORDER = [
    "gre_score",
    "toefl_score",
    "university_rating",
    "sop",
    "lor",
    "cgpa",
    "research",
]


class AdmissionInput(BaseModel):
    """One student, with the valid ranges from the dataset."""

    gre_score: int = Field(..., ge=0, le=340)
    toefl_score: int = Field(..., ge=0, le=120)
    university_rating: int = Field(..., ge=1, le=5)
    sop: float = Field(..., ge=0, le=5)
    lor: float = Field(..., ge=0, le=5)
    cgpa: float = Field(..., ge=0, le=10)
    research: int = Field(..., ge=0, le=1)

    def as_row(self) -> list[float]:
        """The features as a list, in training order."""
        return [float(getattr(self, name)) for name in FEATURE_ORDER]


class AdmissionBatchInput(BaseModel):
    """A batch of students: {"predictions": [ {...}, {...} ]}."""

    predictions: list[AdmissionInput]


class Credentials(BaseModel):
    username: str
    password: str
