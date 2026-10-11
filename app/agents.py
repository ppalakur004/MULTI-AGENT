"""Four bounded reasoning functions, independent of graph orchestration."""

import hashlib
from pathlib import Path
from typing import Any

from app.adapters import ClaimsAdapter, LLMAdapter, MalformedOutput, VerificationAdapter
from app.guardrails import (
    POLICY_VERSION,
    RISK_RULES,
    RiskDisagreement,
    assessment_from_review,
    check_evidence,
    check_recommendation,
    check_risk,
    permitted_action,
    risk_mismatches,
    risk_policy,
    risk_review_flags,
    with_review_confidence,
)
from app.models import (
    Application,
    ClaimsHistory,
    EligibilityVerification,
    EnrichmentResult,
    EnrichmentSummary,
    IntakeResult,
    LLMRiskReview,
    Recommendation,
    RecommendationNarrative,
    RiskResult,
)
from app.observability import Observer

PROMPT_DIR = Path(__file__).parent / "prompts"


async def reason(
    stage: str, payload: dict[str, Any], schema: type, llm: LLMAdapter, observer: Observer
):
    prompt = (PROMPT_DIR / f"{stage}.md").read_text(encoding="utf-8")
    for attempt in range(2):
        version = hashlib.sha256(prompt.encode()).hexdigest()
        with observer.span(stage + "_generation", model=llm.model, version=version) as trace:
            trace["input"] = observer.capture(payload)
            try:
                result = await llm.generate(stage, prompt, payload, schema)
            except MalformedOutput:
                if attempt:
                    raise
                observer.event("model_repair_scheduled", stage=stage, attempt=1)
                prompt += (
                    "\nPrevious output was invalid. Return only JSON matching the supplied schema."
                )
                continue
            trace.update(
                output=observer.capture(result.value.model_dump()),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
            return result.value


async def intake_agent(
    application: Application, llm: LLMAdapter, observer: Observer
) -> IntakeResult:
    result = await reason("intake", application.model_dump(), IntakeResult, llm, observer)
    if result.missing_fields:
        from app.guardrails import GuardrailError

        raise GuardrailError("model identified missing application evidence")
    return result


async def enrichment_agent(
    application: Application,
    verification: VerificationAdapter,
    claims: ClaimsAdapter,
    llm: LLMAdapter,
    observer: Observer,
) -> tuple[EnrichmentResult, EnrichmentSummary]:
    with observer.span("verification_lookup"):
        verified = EligibilityVerification.model_validate(
            await verification.get_verification(application.applicant_id)
        )
    with observer.span("claims_lookup"):
        history = ClaimsHistory.model_validate(await claims.get_history(application.applicant_id))
    facts = EnrichmentResult(verification=verified, claims=history)
    summary = await reason("enrichment", facts.model_dump(), EnrichmentSummary, llm, observer)
    check_evidence(summary.evidence, {"verification", "claims"})
    return facts, summary


async def risk_scoring_agent(
    application: Application, facts: EnrichmentResult, llm: LLMAdapter, observer: Observer
) -> RiskResult:
    expected = risk_policy(application, facts)
    payload = {
        "validated_evidence": {
            "application": {
                "smoker": application.smoker,
                "declared_chronic_conditions": application.declared_chronic_conditions,
                "hospitalizations_last_2_years": application.hospitalizations_last_2_years,
            },
            "claims": {
                "claims_last_2_years": facts.claims.claims_last_2_years,
                "high_cost_claims_last_2_years": facts.claims.high_cost_claims_last_2_years,
            },
        },
        "rules": list(RISK_RULES),
        "policy_version": POLICY_VERSION,
    }
    review = await reason(
        "risk",
        payload,
        LLMRiskReview,
        llm,
        observer,
    )
    candidate = with_review_confidence(assessment_from_review(review), expected)
    try:
        check_risk(candidate, expected)
    except RiskDisagreement:
        observer.event(
            "risk_review_repair_scheduled",
            mismatches=risk_mismatches(candidate, expected),
        )
        repaired_review = await reason(
            "risk",
            {
                **payload,
                "correction_context": {
                    "previous_review": review.model_dump(mode="json"),
                    "mismatches": risk_mismatches(candidate, expected),
                    "instruction": (
                        "Re-evaluate only the identified rule decisions, then return all six rules."
                    ),
                },
            },
            LLMRiskReview,
            llm,
            observer,
        )
        candidate = with_review_confidence(assessment_from_review(repaired_review), expected)
        check_risk(candidate, expected)
    return RiskResult(
        **expected.model_dump(mode="json"),
        confidence=1.0,
        llm_assessment=candidate,
        agreement=True,
        review_flags=risk_review_flags(facts),
    )


async def recommendation_agent(
    risk: RiskResult, llm: LLMAdapter, observer: Observer, minimum: float
) -> Recommendation:
    action = permitted_action(risk)
    narrative = await reason(
        "recommendation",
        {
            "risk": {
                "score": risk.score,
                "band": risk.band.value,
                "factors": risk.factors,
                "review_flags": risk.review_flags,
            },
            "policy_action": action.value,
        },
        RecommendationNarrative,
        llm,
        observer,
    )
    evidence = list(dict.fromkeys([*risk.factors, *risk.review_flags]))
    result = Recommendation(
        action=action,
        reason=narrative.reason,
        evidence=evidence,
        confidence=narrative.confidence,
    )
    check_recommendation(result, risk, minimum)
    return result
