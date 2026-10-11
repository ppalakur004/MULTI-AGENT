import pytest

from app.guardrails import (
    GuardrailError,
    assessment_from_review,
    check_evidence,
    check_recommendation,
    check_risk,
    permitted_action,
    risk_policy,
    risk_review_flags,
)
from app.models import (
    Action,
    Application,
    ClaimsHistory,
    EligibilityVerification,
    EnrichmentResult,
    LLMRiskReview,
    Recommendation,
    RiskFactor,
    RiskRuleDecision,
)


def facts(claims=0, high_cost=0, fraud=False, verified=True):
    return EnrichmentResult(
        verification=EligibilityVerification(identity_verified=verified, fraud_flag=fraud),
        claims=ClaimsHistory(claims_last_2_years=claims, high_cost_claims_last_2_years=high_cost),
    )


def complete_review(triggered=()):
    paths = {
        RiskFactor.SMOKER: "application.smoker",
        RiskFactor.ONE_CHRONIC_CONDITION: "application.declared_chronic_conditions",
        RiskFactor.MULTIPLE_CHRONIC_CONDITIONS: "application.declared_chronic_conditions",
        RiskFactor.RECENT_HOSPITALIZATION: "application.hospitalizations_last_2_years",
        RiskFactor.THREE_OR_MORE_RECENT_CLAIMS: "claims.claims_last_2_years",
        RiskFactor.RECENT_HIGH_COST_CLAIM: "claims.high_cost_claims_last_2_years",
    }
    return LLMRiskReview(
        rule_decisions=[
            RiskRuleDecision(
                code=code,
                triggered=code in triggered,
                evidence_path=paths[code],
                explanation="Compared evidence with the threshold.",
            )
            for code in RiskFactor
        ],
        reasoning="Reviewed every rule.",
    )


@pytest.mark.parametrize(
    "smoker,conditions,hospitalizations,claims,high_cost,band,action",
    [
        (False, 0, 0, 0, 0, "LOW", Action.APPROVE),
        (True, 1, 0, 3, 0, "MEDIUM", Action.REVIEW),
        (True, 2, 1, 5, 2, "HIGH", Action.DENY),
    ],
)
def test_policy(smoker, conditions, hospitalizations, claims, high_cost, band, action):
    app = Application(
        full_name="Test Applicant",
        applicant_id="A",
        age=35,
        coverage_tier="STANDARD",
        smoker=smoker,
        declared_chronic_conditions=conditions,
        hospitalizations_last_2_years=hospitalizations,
    )
    risk = risk_policy(app, facts(claims, high_cost))
    assert risk.band == band
    assert permitted_action(risk) == action


def test_operational_flags_are_separate_from_health_risk():
    assert risk_review_flags(facts(fraud=True)) == ["fraud_review"]
    assert risk_review_flags(facts(verified=False)) == ["identity_unverified"]
    assert (
        risk_policy(
            Application(
                full_name="Test",
                applicant_id="A",
                age=35,
                coverage_tier="STANDARD",
                smoker=False,
                declared_chronic_conditions=0,
                hospitalizations_last_2_years=0,
            ),
            facts(fraud=True, verified=False),
        ).score
        == 0.05
    )


def test_contradictions_and_unsupported_evidence():
    app = Application(
        full_name="Test Applicant",
        applicant_id="A",
        age=35,
        coverage_tier="STANDARD",
        smoker=False,
        declared_chronic_conditions=0,
        hospitalizations_last_2_years=0,
    )
    risk = risk_policy(app, facts())
    candidate = assessment_from_review(complete_review())
    for update in [{"score": 0.9}, {"band": "HIGH"}, {"factors": ["invented"]}]:
        with pytest.raises(GuardrailError):
            check_risk(candidate.model_copy(update=update), risk)
    for evidence in [[], ["invented"]]:
        with pytest.raises(GuardrailError):
            check_evidence(evidence, {"verification"})
    with pytest.raises(GuardrailError):
        check_recommendation(
            Recommendation(action=Action.DENY, reason="bad", evidence=risk.factors, confidence=0.9),
            risk,
            0.75,
        )


@pytest.mark.parametrize(
    "smoker,conditions,hospitalizations,claims,high_cost,score,band",
    [
        (False, 0, 0, 0, 0, 0.05, "LOW"),
        (True, 0, 0, 0, 0, 0.25, "LOW"),
        (False, 1, 0, 0, 0, 0.15, "LOW"),
        (False, 2, 0, 0, 0, 0.30, "MEDIUM"),
        (False, 0, 1, 0, 0, 0.25, "LOW"),
        (False, 0, 0, 3, 0, 0.20, "LOW"),
        (False, 0, 0, 1, 1, 0.20, "LOW"),
        (True, 1, 1, 0, 0, 0.55, "MEDIUM"),
    ],
)
def test_policy_threshold_boundaries(
    smoker, conditions, hospitalizations, claims, high_cost, score, band
):
    application = Application(
        applicant_id="A",
        full_name="Test",
        age=35,
        coverage_tier="STANDARD",
        smoker=smoker,
        declared_chronic_conditions=conditions,
        hospitalizations_last_2_years=hospitalizations,
    )
    result = risk_policy(application, facts(claims, high_cost))
    assert result.score == score
    assert result.band == band


def test_low_confidence_is_rejected():
    application = Application(
        applicant_id="A",
        full_name="Test",
        age=35,
        coverage_tier="STANDARD",
        smoker=False,
        declared_chronic_conditions=0,
        hospitalizations_last_2_years=0,
    )
    risk = risk_policy(application, facts())
    with pytest.raises(GuardrailError):
        check_recommendation(
            Recommendation(
                action=Action.APPROVE, reason="uncertain", evidence=risk.factors, confidence=0.1
            ),
            risk,
            0.75,
        )
