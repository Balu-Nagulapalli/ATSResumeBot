"""Local matching compatibility layer; no provider imports or API calls."""
from typing import Literal
from pydantic import Field
from models.schemas import StrictModel, JobDescriptionData, ResumeData, MatchResult


class Decision(StrictModel):
    requirement_id: int = Field(ge=0)
    candidate_id: int = Field(ge=0)
    relation: Literal['equivalent', 'partial', 'unrelated']


class Decisions(StrictModel):
    decisions: list[Decision]



def refine_semantically(result: MatchResult, jd: JobDescriptionData,
                        resume: ResumeData, *, client=None) -> MatchResult:
    """Retain local exact/alias/TF-IDF evidence already computed by match_documents.

    The legacy client argument is accepted for compatibility and never used.
    No model-generated equivalence judgments are added to local match results.
    """
    return result.model_copy(deep=True)
