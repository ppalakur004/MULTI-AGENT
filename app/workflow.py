"""Checkpointed stages with explicit resume and durable retry counters."""

import asyncio
import random
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.adapters import (
    AdapterError,
    ClaimsAdapter,
    LLMAdapter,
    TransientError,
    VerificationAdapter,
)
from app.agents import enrichment_agent, intake_agent, recommendation_agent, risk_scoring_agent
from app.config import Settings
from app.guardrails import POLICY_VERSION, GuardrailError, RiskDisagreement
from app.models import Action, Application, CaseState, EnrichmentResult, RiskResult, Status, now
from app.observability import Observer

STAGES = ("intake", "enrichment", "risk", "recommendation")


def build_graph(
    settings: Settings,
    llm: LLMAdapter,
    verification: VerificationAdapter,
    claims: ClaimsAdapter,
    observer: Observer,
    checkpointer: Any,
    interrupt_after: list[str] | None = None,
):
    graph = StateGraph(CaseState)

    def node(stage: str):
        async def run(state: CaseState) -> dict[str, Any]:
            attempts = dict(state.get("attempts", {}))
            attempt = attempts.get(stage, 0) + 1
            attempts[stage] = attempt
            if attempt > 1:
                await asyncio.sleep(
                    settings.retry_backoff_seconds * 2 ** (attempt - 2)
                    + random.uniform(0, settings.retry_backoff_seconds)
                )
            try:
                with observer.span(stage) as trace:
                    update: dict[str, Any]
                    application = Application.model_validate(state["application"])
                    if stage == "risk":
                        stage_input = {
                            "application": state["application"],
                            "enrichment": state["enrichment"],
                        }
                    elif stage == "recommendation":
                        stage_input = {"risk": state["risk"]}
                    else:
                        stage_input = {"application": state["application"]}
                    trace["input"] = observer.capture(stage_input)
                    if stage == "intake":
                        result = await intake_agent(application, llm, observer)
                        update = {"intake": result.model_dump(mode="json")}
                    elif stage == "enrichment":
                        facts, summary = await enrichment_agent(
                            application, verification, claims, llm, observer
                        )
                        update = {
                            "enrichment": facts.model_dump(mode="json"),
                            "enrichment_summary": summary.model_dump(mode="json"),
                        }
                    elif stage == "risk":
                        risk_result = await risk_scoring_agent(
                            application,
                            EnrichmentResult.model_validate(state["enrichment"]),
                            llm,
                            observer,
                        )
                        update = {"risk": risk_result.model_dump(mode="json")}
                    else:
                        recommendation = await recommendation_agent(
                            RiskResult.model_validate(state["risk"]),
                            llm,
                            observer,
                            settings.min_recommendation_confidence,
                        )
                        update = {
                            "recommendation": recommendation.model_dump(mode="json"),
                            "status": Status.REVIEW
                            if recommendation.action == Action.REVIEW
                            else Status.COMPLETE,
                        }
                    trace["output"] = observer.capture(update)
                return {**update, "attempts": attempts, "updated_at": now()}
            except (AdapterError, GuardrailError, ValidationError) as exc:
                transient = isinstance(exc, TransientError)
                exhausted = attempt >= settings.max_agent_retries + 1
                errors = [
                    *state.get("errors", []),
                    {
                        "stage": stage,
                        "kind": type(exc).__name__,
                        "attempt": attempt,
                        "retryable": transient,
                    },
                ]
                update = {"attempts": attempts, "errors": errors, "updated_at": now()}
                if isinstance(exc, RiskDisagreement):
                    update["risk_comparison"] = {
                        "authoritative": exc.expected.model_dump(mode="json"),
                        "llm_assessment": exc.candidate.model_dump(mode="json"),
                        "agreement": False,
                    }
                if transient and not exhausted:
                    observer.event("retry_scheduled", stage=stage, attempt=attempt)
                else:
                    update.update(
                        status=Status.REVIEW, review_reason=f"{stage}: {type(exc).__name__}"
                    )
                    observer.event(
                        "human_review_required", stage=stage, error_class=type(exc).__name__
                    )
                return update

        return run

    for index, stage in enumerate(STAGES):
        graph.add_node(stage, node(stage))
        next_stage = STAGES[index + 1] if index < len(STAGES) - 1 else END

        def route(state, current=stage, following=next_stage):
            if state.get("status") in {Status.REVIEW, Status.COMPLETE}:
                return END
            if state.get("errors") and state["errors"][-1]["stage"] == current:
                last = state["errors"][-1]
                if last["attempt"] == state.get("attempts", {}).get(current):
                    return current
            return following

        graph.add_conditional_edges(stage, route, {stage: stage, next_stage: next_stage, END: END})
    graph.add_edge(START, "intake")
    return graph.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)


async def execute(graph, case_id: str, initial: CaseState | None = None) -> CaseState:
    # With the default two retries, four stages need at most twelve node executions.
    # Twenty allows headroom while bounding accidental routing loops.
    config = {"configurable": {"thread_id": case_id}, "recursion_limit": 20}
    snapshot = await graph.aget_state(config)
    if initial is not None:
        if snapshot.values:
            raise ValueError("case already exists; use resume")
        if initial["case_id"] != case_id:
            raise ValueError("case/thread identifiers differ")
        return await graph.ainvoke(initial, config)
    if not snapshot.values:
        raise ValueError("case not found")
    if not snapshot.next:
        return snapshot.values
    if snapshot.values.get("policy_version") != POLICY_VERSION:
        raise ValueError("policy changed; case requires manual review before resume")
    return await graph.ainvoke(None, config)
