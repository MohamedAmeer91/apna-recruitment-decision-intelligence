from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experience_months: float = Field(ge=0, le=600)
    interview_reschedules_90d: float = Field(ge=0, le=30)
    salary_gap_ratio: float = Field(ge=0, le=1)
    expected_salary_inr_thousands: float = Field(ge=1, le=1000)
    profile_completeness_score: float = Field(ge=0, le=100)
    competing_offer_flag: Literal[0, 1]
    referred_flag: Literal[0, 1]
    tau: Optional[float] = Field(default=None, ge=0, le=1)


class Contribution(BaseModel):
    feature: str
    x_name: str
    value: float
    z: float
    weight: float
    contribution: float


class ScoreResponse(BaseModel):
    probability: float
    logit: float
    bias: float
    tau: float
    flagged: bool
    contributions: List[Contribution]
    segment: str
    note: str
