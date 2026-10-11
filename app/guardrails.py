"""Deterministic demo policy and semantic output validation."""

from typing import Literal

from app.models import (
    Action,
    Application,
    Band,
    EnrichmentResult,
    LLMRiskAssessment,
    LLMRiskReview,
    PolicyRisk,
    Recommendation,
    RiskFactor,
    RiskResult,
)

POLICY_VERSION = "synthetic-health-reviewed-v3"
BASE_RISK_POINTS = 5
RISK_WEIGHTS = {
    RiskFactor.SMOKER: 20,
    RiskFactor.ONE_CHRONIC_CONDITION: 10,
    RiskFactor.MULTIPLE_CHRONIC_CONDITIONS: 25,
    RiskFactor.RECENT_HOSPITALIZATION: 20,
    RiskFactor.THREE_OR_MORE_RECENT_CLAIMS: 15,
    RiskFactor.RECENT_HIGH_COST_CLAIM: 15,
}
RISK_RULES = (
    {
        "code": RiskFactor.SMOKER.value,
        "evidence_path": "application.smoker",
        "operator": "equals",
        "threshold": True,
        "weight_points": 20,
    },
    {
        "code": RiskFactor.ONE_CHRONIC_CONDITION.value,
        "evidence_path": "application.declared_chronic_conditions",
        "operator": "equals",
        "threshold": 1,
        "weight_points": 10,
    },
    {
        "code": RiskFactor.MULTIPLE_CHRONIC_CONDITIONS.value,
        "evidence_path": "application.declared_chronic_conditions",
        "operator": "greater_than_or_equal",
        "threshold": 2,
        "weight_points": 25,
    },
    {
        "code": RiskFactor.RECENT_HOSPITALIZATION.value,
        "evidence_path": "application.hospitalizations_last_2_years",
        "operator": "greater_than_or_equal",
        "threshold": 1,
        "weight_points": 20,
    },
    {
        "code": RiskFactor.THREE_OR_MORE_RECENT_CLAIMS.value,
        "evidence_path": "claims.claims_last_2_years",
        "operator": "greater_than_or_equal",
        "threshold": 3,
        "weight_points": 15,
    },
    {
        "code": RiskFactor.RECENT_HIGH_COST_CLAIM.value,
        "evidence_path": "claims.high_cost_claims_last_2_years",
        "operator": "greater_than_or_equal",
        "threshold": 1,
        "weight_points": 15,
    },
)


class GuardrailError(ValueError):
    pass


class RiskDisagreement(GuardrailError):
    def __init__(self, expected: PolicyRisk, candidate: LLMRiskAssessment):
        super().__init__("LLM risk review disagrees with deterministic policy")
        self.expected = expected
        self.candidate = candidate


def active_risk_factors(application: Application, facts: EnrichmentResult) -> list[RiskFactor]:
    factors: list[RiskFactor] = []
    if application.smoker:
        factors.append(RiskFactor.SMOKER)
    if application.declared_chronic_conditions >= 2:
        factors.append(RiskFactor.MULTIPLE_CHRONIC_CONDITIONS)
    elif application.declared_chronic_conditions == 1:
        factors.append(RiskFactor.ONE_CHRONIC_CONDITION)
    if application.hospitalizations_last_2_years >= 1:
        factors.append(RiskFactor.RECENT_HOSPITALIZATION)
    if facts.claims.claims_last_2_years >= 3:
        factors.append(RiskFactor.THREE_OR_MORE_RECENT_CLAIMS)
    if facts.claims.high_cost_claims_last_2_years >= 1:
        factors.append(RiskFactor.RECENT_HIGH_COST_CLAIM)
    return factors


def policy_from_factors(factors: list[RiskFactor]) -> PolicyRisk:
    ordered = [factor for factor in RiskFactor if factor in set(factors)]
    points = min(BASE_RISK_POINTS + sum(RISK_WEIGHTS[factor] for factor in ordered), 100)
    score = points / 100
    band = Band.LOW if points < 30 else Band.MEDIUM if points < 65 else Band.HIGH
    return PolicyRisk(
        score=score,
        band=band,
        factors=["application", "claims", *(factor.value for factor in ordered)],
    )


def risk_policy(application: Application, facts: EnrichmentResult) -> PolicyRisk:
    return policy_from_factors(active_risk_factors(application, facts))


def risk_review_flags(
    facts: EnrichmentResult,
) -> list[Literal["identity_unverified", "fraud_review"]]:
    flags: list[Literal["identity_unverified", "fraud_review"]] = []
    if not facts.verification.identity_verified:
        flags.append("identity_unverified")
    if facts.verification.fraud_flag:
        flags.append("fraud_review")
    return flags


def assessment_from_review(review: LLMRiskReview) -> LLMRiskAssessment:
    factors = [decision.code for decision in review.rule_decisions if decision.triggered]
    policy = policy_from_factors(factors)
    # Half of review confidence comes from complete evidence/rule coverage. Agreement is added
    # only after comparison with the independent policy result. This is not a probability.
    confidence = 0.5 if not review.data_quality_issues else 0.0
    return LLMRiskAssessment(
        **policy.model_dump(mode="json"),
        reasoning=review.reasoning,
        confidence=confidence,
        rule_decisions=review.rule_decisions,
        data_quality_issues=review.data_quality_issues,
    )


def risk_mismatches(candidate: LLMRiskAssessment, expected: PolicyRisk) -> list[str]:
    mismatches: list[str] = []
    if candidate.score != expected.score:
        mismatches.append(f"score expected {expected.score:.2f}, got {candidate.score:.2f}")
    if candidate.band != expected.band:
        candidate_band = getattr(candidate.band, "value", str(candidate.band))
        mismatches.append(f"band expected {expected.band.value}, got {candidate_band}")
    missing = sorted(set(expected.factors) - set(candidate.factors))
    extra = sorted(set(candidate.factors) - set(expected.factors))
    if missing:
        mismatches.append("missing factors: " + ", ".join(missing))
    if extra:
        mismatches.append("unexpected factors: " + ", ".join(extra))
    if candidate.data_quality_issues:
        mismatches.append("unsupported data-quality issues reported")
    return mismatches


def with_review_confidence(candidate: LLMRiskAssessment, expected: PolicyRisk) -> LLMRiskAssessment:
    confidence = 1.0 if not risk_mismatches(candidate, expected) else candidate.confidence
    return candidate.model_copy(update={"confidence": confidence})


def permitted_action(risk: PolicyRisk | RiskResult) -> Action:
    if isinstance(risk, RiskResult) and risk.review_flags:
        return Action.REVIEW
    if risk.band == Band.MEDIUM:
        return Action.REVIEW
    return Action.DENY if risk.band == Band.HIGH else Action.APPROVE


def check_evidence(references: list[str], allowed: set[str]) -> None:
    if not references or not set(references).issubset(allowed):
        raise GuardrailError("unsupported evidence references")


def check_risk(candidate: LLMRiskAssessment, expected: PolicyRisk) -> None:
    if risk_mismatches(candidate, expected):
        raise RiskDisagreement(expected, candidate)


def check_recommendation(result: Recommendation, risk: RiskResult, minimum: float) -> None:
    check_evidence(result.evidence, set(risk.factors) | set(getattr(risk, "review_flags", [])))
    if result.action != permitted_action(risk):
        raise GuardrailError("recommendation contradicts policy")
    if result.action != Action.REVIEW and (
        result.confidence < minimum or risk.confidence < minimum
    ):
        raise GuardrailError("insufficient confidence for automated recommendation")
