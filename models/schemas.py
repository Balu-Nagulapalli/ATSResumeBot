"""Strict extraction and scoring contracts. Quotes must be verbatim source text."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class Fact(StrictModel):
    value: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class Years(StrictModel):
    value: float = Field(ge=0, le=80)
    quote: str = Field(min_length=1)


class Education(StrictModel):
    level: Literal['high_school', 'associate', 'bachelor', 'master', 'doctorate', 'other']
    field: str | None
    related_field_allowed: bool
    quote: str = Field(min_length=1)


class Role(StrictModel):
    title: str
    start_date: str | None
    end_date: str | None
    quote: str = Field(min_length=1)


class JobDescriptionData(StrictModel):
    job_title: Fact | None = None
    seniority: Fact | None = None
    required_skills: list[Fact] = Field(default_factory=list)
    preferred_skills: list[Fact] = Field(default_factory=list)
    required_tools: list[Fact] = Field(default_factory=list)
    preferred_tools: list[Fact] = Field(default_factory=list)
    minimum_experience_years: Years | None = None
    maximum_experience_years: Years | None = None
    education_requirements: list[Education] = Field(default_factory=list)
    certifications: list[Fact] = Field(default_factory=list)
    responsibilities: list[Fact] = Field(default_factory=list)
    domain: Fact | None = None
    keywords: list[Fact] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def check_years(self):
        if (self.minimum_experience_years and self.maximum_experience_years
                and self.minimum_experience_years.value > self.maximum_experience_years.value):
            raise ValueError('Minimum experience exceeds maximum experience.')
        return self


class ResumeData(StrictModel):
    candidate_name: Fact | None = None
    headline: Fact | None = None
    total_experience_years: Years | None = None
    skills: list[Fact] = Field(default_factory=list)
    tools: list[Fact] = Field(default_factory=list)
    programming_languages: list[Fact] = Field(default_factory=list)
    frameworks: list[Fact] = Field(default_factory=list)
    databases: list[Fact] = Field(default_factory=list)
    cloud_platforms: list[Fact] = Field(default_factory=list)
    roles: list[Role] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[Fact] = Field(default_factory=list)
    projects: list[Fact] = Field(default_factory=list)
    achievements: list[Fact] = Field(default_factory=list)
    responsibilities: list[Fact] = Field(default_factory=list)
    domains: list[Fact] = Field(default_factory=list)
    contact_present: bool = False
    linkedin_present: bool = False
    github_present: bool = False
    evidence: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class ValidationResult(StrictModel):
    valid: bool
    confidence: float = Field(ge=0, le=1)
    reasons: list[str]


class RequirementMatch(StrictModel):
    requirement: str
    required: bool = True
    credit: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)
    method: str


class CriterionResult(StrictModel):
    applicable: bool
    score: float = Field(ge=0, le=100)
    matches: list[RequirementMatch] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    explanation: str
    confidence: float = Field(ge=0, le=1)


class MatchResult(StrictModel):
    criteria: dict[str, CriterionResult]
    candidate_years: float | None
    warnings: list[str] = Field(default_factory=list)


class ScoreResult(StrictModel):
    overall: float = Field(ge=0, le=100)
    effective_weights: dict[str, float]
    contributions: dict[str, float]
    classification: str
    confidence: float = Field(ge=0, le=1)
    scoring_version: str
