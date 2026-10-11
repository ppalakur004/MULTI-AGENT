"""Validated contracts and checkpoint state."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def now() -> str:
    return datetime.now(UTC).isoformat()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class Application(Contract):
    applicant_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    full_name: str = Field(min_length=1, max_length=120)
    age: int = Field(ge=18, le=100, strict=True)
    coverage_tier: Literal["BASIC", "STANDARD", "PREMIUM"]
    smoker: bool = Field(strict=True)
    declared_chronic_conditions: int = Field(ge=0, le=10, strict=True)
    hospitalizations_last_2_years: int = Field(ge=0, le=20, strict=True)
    occupation: str | None = Field(default=None, max_length=120)

    @field_validator("coverage_tier", mode="before")
    @classmethod
    def normalized_coverage_tier(cls, value):
        return value.upper() if isinstance(value, str) else value


class IntakeResult(Contract):
    validation_notes: list[str] = Field(max_length=10)
    missing_fields: list[str] = Field(default_factory=list, max_length=10)


class EligibilityVerification(Contract):
    identity_verified: bool = Field(strict=True)
    fraud_flag: bool = Field(strict=True)
    source: str = "mock_verification"
    synthetic: bool = True


class ClaimsHistory(Contract):
    claims_last_2_years: int = Field(ge=0, strict=True)
    high_cost_claims_last_2_years: int = Field(ge=0, strict=True)
    source: str = "mock_claims"
    synthetic: bool = True

    @model_validator(mode="after")
    def consistent(self) -> "ClaimsHistory":
        if self.high_cost_claims_last_2_years > self.claims_last_2_years:
            raise ValueError("high-cost claims cannot exceed total claims")
        return self


class EnrichmentResult(Contract):
    verification: EligibilityVerification
    claims: ClaimsHistory


class EnrichmentSummary(Contract):
    summary: str = Field(min_length=1, max_length=500)
    evidence: list[str] = Field(min_length=1, max_length=20)


class Band(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskFactor(StrEnum):
    SMOKER = "smoker"
    ONE_CHRONIC_CONDITION = "one_chronic_condition"
    MULTIPLE_CHRONIC_CONDITIONS = "multiple_chronic_conditions"
    RECENT_HOSPITALIZATION = "recent_hospitalization"
    THREE_OR_MORE_RECENT_CLAIMS = "three_or_more_recent_claims"
    RECENT_HIGH_COST_CLAIM = "recent_high_cost_claim"


class RiskRuleDecision(Contract):
    code: RiskFactor
    triggered: bool = Field(strict=True)
    evidence_path: Literal[
        "application.smoker",
        "application.declared_chronic_conditions",
        "application.hospitalizations_last_2_years",
        "claims.claims_last_2_years",
        "claims.high_cost_claims_last_2_years",
    ]
    explanation: str = Field(min_length=1, max_length=300)


class LLMRiskReview(Contract):
    rule_decisions: list[RiskRuleDecision] = Field(min_length=6, max_length=6)
    data_quality_issues: list[str] = Field(default_factory=list, max_length=10)
    reasoning: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def complete_rule_set(self) -> "LLMRiskReview":
        expected = set(RiskFactor)
        actual = [decision.code for decision in self.rule_decisions]
        if len(set(actual)) != len(actual) or set(actual) != expected:
            raise ValueError("risk review must contain every rule exactly once")
        expected_paths = {
            RiskFactor.SMOKER: "application.smoker",
            RiskFactor.ONE_CHRONIC_CONDITION: "application.declared_chronic_conditions",
            RiskFactor.MULTIPLE_CHRONIC_CONDITIONS: "application.declared_chronic_conditions",
            RiskFactor.RECENT_HOSPITALIZATION: "application.hospitalizations_last_2_years",
            RiskFactor.THREE_OR_MORE_RECENT_CLAIMS: "claims.claims_last_2_years",
            RiskFactor.RECENT_HIGH_COST_CLAIM: "claims.high_cost_claims_last_2_years",
        }
        if any(
            decision.evidence_path != expected_paths[decision.code]
            for decision in self.rule_decisions
        ):
            raise ValueError("risk rule uses an unsupported evidence path")
        return self


class Action(StrEnum):
    APPROVE = "RECOMMEND_APPROVE"
    DENY = "RECOMMEND_DENY"
    REVIEW = "HUMAN_REVIEW"


class PolicyRisk(Contract):
    score: float = Field(ge=0, le=1)
    band: Band
    factors: list[str] = Field(min_length=1, max_length=20)


class LLMRiskAssessment(PolicyRisk):
    reasoning: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0, le=1)
    rule_decisions: list[RiskRuleDecision] = Field(min_length=6, max_length=6)
    data_quality_issues: list[str] = Field(default_factory=list, max_length=10)


class RiskResult(PolicyRisk):
    confidence: float = Field(ge=0, le=1)
    llm_assessment: LLMRiskAssessment
    agreement: bool
    review_flags: list[Literal["identity_unverified", "fraud_review"]] = Field(
        default_factory=list, max_length=2
    )


class Recommendation(Contract):
    action: Action
    reason: str = Field(min_length=1, max_length=500)
    evidence: list[str] = Field(min_length=1, max_length=20)
    confidence: float = Field(ge=0, le=1)


class RecommendationNarrative(Contract):
    """LLM-authored explanation; policy-controlled fields are added by Python."""

    reason: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0, le=1)


class Status(StrEnum):
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    REVIEW = "HUMAN_REVIEW_REQUIRED"


class CaseState(TypedDict, total=False):
    case_id: str
    application: dict[str, Any]
    intake: dict[str, Any]
    enrichment: dict[str, Any]
    enrichment_summary: dict[str, Any]
    risk: dict[str, Any]
    risk_comparison: dict[str, Any]
    recommendation: dict[str, Any]
    status: str
    errors: list[dict[str, Any]]
    attempts: dict[str, int]
    review_reason: str
    created_at: str
    updated_at: str
    policy_version: str
    backend: str
