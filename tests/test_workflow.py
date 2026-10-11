import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.adapters import (
    FakeLLMAdapter,
    MockClaimsAdapter,
    MockVerificationAdapter,
    load_application,
)
from app.config import Settings
from app.guardrails import POLICY_VERSION
from app.models import Application
from app.observability import Observer
from app.workflow import build_graph, execute


def initial(applicant="A-1001", case_id="APP-111111111111"):
    application = (
        load_application(applicant)
        if applicant != "UNKNOWN"
        else Application(
            full_name="Test Applicant",
            applicant_id=applicant,
            age=35,
            coverage_tier="STANDARD",
            smoker=False,
            declared_chronic_conditions=0,
            hospitalizations_last_2_years=0,
        )
    )
    return {
        "case_id": case_id,
        "application": application.model_dump(),
        "status": "RUNNING",
        "errors": [],
        "attempts": {},
        "policy_version": POLICY_VERSION,
        "backend": "fake",
    }


def graph(
    tmp_path, llm=None, saver=None, interrupt=None, case_id="APP-111111111111", verification=None
):
    return build_graph(
        Settings(trace_dir=str(tmp_path), retry_backoff_seconds=0),
        llm or FakeLLMAdapter(),
        verification or MockVerificationAdapter(),
        MockClaimsAdapter(),
        Observer(Settings(trace_dir=str(tmp_path)), case_id),
        saver or InMemorySaver(),
        interrupt_after=interrupt,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "applicant,status,action",
    [
        ("A-1001", "COMPLETE", "RECOMMEND_APPROVE"),
        ("A-1003", "COMPLETE", "RECOMMEND_DENY"),
        ("A-1002", "HUMAN_REVIEW_REQUIRED", "HUMAN_REVIEW"),
        ("A-1006", "HUMAN_REVIEW_REQUIRED", "HUMAN_REVIEW"),
        ("UNKNOWN", "HUMAN_REVIEW_REQUIRED", None),
    ],
)
async def test_scenarios(tmp_path, applicant, status, action):
    result = await execute(graph(tmp_path), "APP-111111111111", initial(applicant))
    assert result["status"] == status
    if action:
        assert result["recommendation"]["action"] == action
    else:
        assert "risk" not in result and "recommendation" not in result
        assert result["attempts"]["enrichment"] == 1


@pytest.mark.asyncio
async def test_verification_timeout_is_injected_by_test(tmp_path):
    from app.adapters import TransientError

    class UnavailableVerification:
        async def get_verification(self, applicant_id):
            raise TransientError("simulated verification timeout")

    result = await execute(
        graph(tmp_path, verification=UnavailableVerification()), "APP-111111111111", initial()
    )
    assert result["status"] == "HUMAN_REVIEW_REQUIRED"
    assert result["attempts"]["enrichment"] == 3
    assert "risk" not in result


@pytest.mark.asyncio
async def test_risk_disagreement_preserves_both_assessments_and_escalates(tmp_path):
    class DisagreeingLLM(FakeLLMAdapter):
        async def generate(self, stage, prompt, payload, schema):
            response = await super().generate(stage, prompt, payload, schema)
            if stage == "risk":
                decisions = list(response.value.rule_decisions)
                decisions[0] = decisions[0].model_copy(update={"triggered": True})
                response.value = response.value.model_copy(update={"rule_decisions": decisions})
            return response

    result = await execute(graph(tmp_path, llm=DisagreeingLLM()), "APP-111111111111", initial())
    assert result["status"] == "HUMAN_REVIEW_REQUIRED"
    assert result["review_reason"] == "risk: RiskDisagreement"
    assert result["risk_comparison"]["authoritative"]["score"] == 0.05
    assert result["risk_comparison"]["llm_assessment"]["score"] == 0.25
    assert result["risk_comparison"]["agreement"] is False
    assert "risk" not in result and "recommendation" not in result


@pytest.mark.asyncio
async def test_risk_disagreement_gets_one_targeted_repair(tmp_path):
    class CorrectedLLM(FakeLLMAdapter):
        risk_calls = 0

        async def generate(self, stage, prompt, payload, schema):
            response = await super().generate(stage, prompt, payload, schema)
            if stage == "risk":
                self.risk_calls += 1
                if self.risk_calls == 1:
                    decisions = list(response.value.rule_decisions)
                    decisions[0] = decisions[0].model_copy(update={"triggered": True})
                    response.value = response.value.model_copy(update={"rule_decisions": decisions})
            return response

    llm = CorrectedLLM()
    result = await execute(graph(tmp_path, llm=llm), "APP-111111111111", initial())
    assert result["status"] == "COMPLETE"
    assert result["risk"]["agreement"] is True
    assert llm.calls.count("risk") == 2


@pytest.mark.asyncio
async def test_transient_recovery_and_no_replay(tmp_path):
    llm = FakeLLMAdapter("risk", 2)
    workflow = graph(tmp_path, llm)
    result = await execute(workflow, "APP-111111111111", initial())
    assert result["status"] == "COMPLETE" and result["attempts"]["risk"] == 3
    calls = list(llm.calls)
    assert await execute(workflow, "APP-111111111111") == result
    assert llm.calls == calls
    with pytest.raises(ValueError):
        await execute(workflow, "APP-111111111111", initial())


@pytest.mark.asyncio
async def test_all_stages_recover_within_twenty_steps(tmp_path):
    from app.adapters import TransientError

    class TwiceFailingLLM(FakeLLMAdapter):
        def __init__(self):
            super().__init__()
            self.attempts = {}

        async def generate(self, stage, prompt, payload, schema):
            self.attempts[stage] = self.attempts.get(stage, 0) + 1
            if self.attempts[stage] <= 2:
                raise TransientError("simulated timeout")
            return await super().generate(stage, prompt, payload, schema)

    result = await execute(graph(tmp_path, llm=TwiceFailingLLM()), "APP-111111111111", initial())
    assert result["status"] == "COMPLETE"
    assert sum(result["attempts"].values()) == 12


@pytest.mark.asyncio
async def test_resume_after_enrichment_and_isolation(tmp_path):
    saver = InMemorySaver()
    llm = FakeLLMAdapter()
    interrupted = graph(tmp_path, llm, saver, ["enrichment"])
    partial = await execute(interrupted, "APP-111111111111", initial())
    assert "enrichment" in partial and "risk" not in partial
    resumed = await execute(graph(tmp_path, llm, saver), "APP-111111111111")
    assert resumed["status"] == "COMPLETE"
    assert llm.calls.count("intake") == 1 and llm.calls.count("enrichment") == 1
    with pytest.raises(ValueError):
        await execute(graph(tmp_path, saver=saver), "APP-222222222222")


@pytest.mark.asyncio
async def test_retry_budget_survives_resume(tmp_path):
    saver, llm = InMemorySaver(), FakeLLMAdapter("risk", 10)
    partial = await execute(graph(tmp_path, llm, saver, ["risk"]), "APP-111111111111", initial())
    assert partial["attempts"]["risk"] == 1
    resumed = await execute(graph(tmp_path, llm, saver), "APP-111111111111")
    assert resumed["status"] == "HUMAN_REVIEW_REQUIRED"
    assert resumed["attempts"]["risk"] == 3
    assert "risk" not in resumed
