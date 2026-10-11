"""CLI commands for submit, resume, status, and deterministic demonstrations."""

import asyncio
import hashlib
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import typer
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from pydantic import ConfigDict, TypeAdapter, ValidationError

from app.adapters import (
    FakeLLMAdapter,
    LLMAdapter,
    MockClaimsAdapter,
    MockVerificationAdapter,
    OllamaAdapter,
    load_application,
)
from app.agents import risk_scoring_agent
from app.config import Settings
from app.guardrails import (
    POLICY_VERSION,
    RiskDisagreement,
    assessment_from_review,
    risk_mismatches,
    risk_policy,
    risk_review_flags,
    with_review_confidence,
)
from app.models import (
    Application,
    CaseState,
    ClaimsHistory,
    EligibilityVerification,
    EnrichmentResult,
    LLMRiskReview,
    Status,
    now,
)
from app.observability import Observer
from app.workflow import build_graph, execute

cli = typer.Typer(
    help="Synthetic health-insurance pre-underwriting workflow. Recommendations are demonstrations."
)


def identifier(case_id: str) -> str:
    if not re.fullmatch(r"APP-[A-F0-9]{12}", case_id):
        raise ValueError("invalid case ID")
    return case_id


@asynccontextmanager
async def persistence(settings: Settings, case_id: str, memory: bool, read_only: bool = False):
    if memory:
        yield InMemorySaver()
        return
    if read_only:
        async with AsyncPostgresSaver.from_conn_string(settings.database_url) as saver:
            yield saver
        return
    # Session-scoped advisory lock prevents concurrent executions of the same case.
    lock_key = int.from_bytes(hashlib.sha256(case_id.encode()).digest()[:8], "big", signed=True)
    async with await AsyncConnection.connect(
        settings.database_url, autocommit=True, connect_timeout=10
    ) as conn:
        cursor = await conn.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,))
        acquired = await cursor.fetchone()
        if not acquired or not acquired[0]:
            raise ValueError("case is already being processed")
        async with AsyncPostgresSaver.from_conn_string(settings.database_url) as saver:
            await saver.setup()
            yield saver


async def run_case(
    application: Application | None,
    case_id: str,
    *,
    backend: str,
    memory: bool = False,
    read_only: bool = False,
    trace_synthetic: bool = False,
    llm_override: LLMAdapter | None = None,
) -> CaseState:
    settings = Settings()
    if trace_synthetic and (
        application is None or application != load_application(application.applicant_id)
    ):
        raise ValueError("detailed tracing requires an unchanged fictional dataset application")
    observer = Observer(settings, case_id, synthetic=trace_synthetic)
    try:
        async with persistence(settings, case_id, memory, read_only) as saver:
            if application:
                selected = backend
            else:
                snapshot = await saver.aget_tuple({"configurable": {"thread_id": case_id}})
                if snapshot is None:
                    raise ValueError("case not found")
                selected = snapshot.checkpoint["channel_values"].get("backend", "ollama")
            llm: LLMAdapter = (
                FakeLLMAdapter()
                if selected == "fake"
                else OllamaAdapter(
                    settings.ollama_base_url, settings.ollama_model, settings.ollama_timeout_seconds
                )
            )
            if llm_override is not None:
                llm = llm_override
            graph = build_graph(
                settings, llm, MockVerificationAdapter(), MockClaimsAdapter(), observer, saver
            )
            if read_only:
                snapshot = await graph.aget_state({"configurable": {"thread_id": case_id}})
                if not snapshot.values:
                    raise ValueError("case not found")
                return snapshot.values
            initial: CaseState | None = None
            if application:
                initial = {
                    "case_id": case_id,
                    "application": application.model_dump(mode="json"),
                    "status": Status.RUNNING,
                    "errors": [],
                    "attempts": {},
                    "created_at": now(),
                    "updated_at": now(),
                    "policy_version": POLICY_VERSION,
                    "backend": selected,
                }
            observer.event("case_started" if initial else "case_resumed", backend=selected)
            with observer.span(f"Underwriting / {case_id}", root=True) as trace:
                trace["input"] = {
                    "case_id": case_id,
                    "backend": selected,
                    "operation": "submit" if initial else "resume",
                }
                result = await execute(graph, case_id, initial)
                trace["output"] = {
                    "case_id": case_id,
                    "status": result["status"],
                    "action": result.get("recommendation", {}).get("action"),
                    "error_count": len(result.get("errors", [])),
                }
            observer.event("case_finished", status=result["status"])
            return result
    finally:
        observer.flush()


def display(result: CaseState, json_output: bool = False) -> None:
    if json_output:
        typer.echo(json.dumps(result, indent=2, default=str))
        return
    application = result.get("application", {})
    typer.echo(
        f"\nApplicant: {application.get('full_name', 'Unknown')} ({application.get('applicant_id', 'Unknown')})"
    )
    typer.echo(f"Case ID: {result.get('case_id', 'Unknown')}")
    typer.echo(f"Status: {result['status']}")
    if risk := result.get("risk"):
        typer.echo(f"Risk: {risk['band']} (score {risk['score']:.2f})")
        if risk.get("review_flags"):
            typer.echo(f"Review flags: {', '.join(risk['review_flags'])}")
    if recommendation := result.get("recommendation"):
        typer.echo(f"Recommendation: {recommendation['action']}")
        typer.echo(
            f"Model-reported confidence: {recommendation['confidence']:.0%} (not calibrated)"
        )
        typer.echo(f"Reason: {recommendation['reason']}")
    if reason := result.get("review_reason"):
        typer.echo(f"Review reason: {reason}")
    if result.get("case_id"):
        typer.echo(f"Trace: {Settings().trace_dir}/{result['case_id']}.jsonl")


def invoke(coroutine, json_output: bool = False) -> None:
    try:
        display(asyncio.run(coroutine), json_output)
    except Exception as exc:
        # Exception text can contain DSNs, SQL, or applicant data.
        typer.echo(
            f"Operation failed ({type(exc).__name__}). Check configuration and service availability.",
            err=True,
        )
        raise typer.Exit(1) from None


def prompt_field(field: str, label: str, value_type=str, default=None):
    definition = Application.model_fields[field]
    annotation = (
        Annotated[definition.annotation, *definition.metadata]
        if definition.metadata
        else definition.annotation
    )
    adapter: TypeAdapter[Any] = TypeAdapter(
        annotation,
        config=ConfigDict(allow_inf_nan=False, str_strip_whitespace=True),
    )
    while True:
        value = typer.prompt(label, type=value_type, default=default)
        try:
            return adapter.validate_python(value)
        except ValidationError:
            typer.echo(f"Invalid {label.lower()}. Please check its allowed range or format.")


def prompt_application() -> Application:
    while True:
        typer.echo("\n1. Existing applicant\n2. New applicant")
        choice = typer.prompt("Choose 1 or 2", type=int, default=1)
        if choice not in {1, 2}:
            typer.echo("Please choose 1 or 2.")
            continue
        entered_id = prompt_field("applicant_id", "Applicant ID")
        try:
            existing = load_application(entered_id)
        except ValueError:
            existing = None
        if choice == 1:
            if existing is None:
                typer.echo("Applicant not found. Try another ID or choose New applicant.")
                continue
            typer.echo(
                f"Loaded {existing.full_name} ({existing.applicant_id}). Starting underwriting."
            )
            return existing
        if existing is not None:
            typer.echo("That ID already exists. Choose Existing applicant or use a new ID.")
            continue
        typer.echo("New applicants without verification/claims records require human review.")
        return Application(
            applicant_id=entered_id,
            full_name=prompt_field("full_name", "Full name"),
            age=prompt_field("age", "Age", int),
            coverage_tier=prompt_field("coverage_tier", "Coverage tier", str, default="STANDARD"),
            smoker=prompt_field("smoker", "Smoker", bool, default=False),
            declared_chronic_conditions=prompt_field(
                "declared_chronic_conditions", "Declared chronic conditions", int, default=0
            ),
            hospitalizations_last_2_years=prompt_field(
                "hospitalizations_last_2_years", "Hospitalizations in last 2 years", int, default=0
            ),
            occupation=prompt_field("occupation", "Occupation", str, default="") or None,
        )


@cli.command()
def submit(
    file: Path | None = typer.Option(
        None, "--file", help="JSON application; omit for interactive input."
    ),
    fake: bool = typer.Option(False, "--fake", help="Use deterministic fake reasoning."),
    applicant_id: str | None = typer.Option(
        None, "--applicant-id", help="Load an application from data/applications.json."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the full case as JSON."),
):
    """Submit a new application with durable PostgreSQL checkpoints."""
    try:
        if file and applicant_id:
            raise ValueError("choose file or applicant ID")
        if applicant_id:
            application = load_application(applicant_id)
        elif file:
            application = Application.model_validate_json(file.read_text())
        else:
            application = prompt_application()
    except (ValueError, OSError):
        typer.echo("Invalid application: check required fields, types, and ranges.", err=True)
        raise typer.Exit(2) from None
    case_id = "APP-" + uuid.uuid4().hex[:12].upper()
    typer.echo(f"Case ID: {case_id}", err=json_output)
    invoke(run_case(application, case_id, backend="fake" if fake else "ollama"), json_output)


@cli.command()
def resume(case_id: str, json_output: bool = typer.Option(False, "--json")):
    """Resume pending stages without replaying completed stages."""
    try:
        identifier(case_id)
    except ValueError:
        raise typer.BadParameter("expected APP- followed by 12 hexadecimal characters") from None
    invoke(run_case(None, case_id, backend="ollama"), json_output)


@cli.command()
def status(case_id: str, json_output: bool = typer.Option(False, "--json")):
    """Read the latest persisted case state."""
    try:
        identifier(case_id)
    except ValueError:
        raise typer.BadParameter("invalid case ID") from None
    invoke(run_case(None, case_id, backend="ollama", read_only=True), json_output)


@cli.command()
def run(
    applicant_id: str = typer.Argument(..., help="Applicant ID from data/applications.json."),
    fake: bool = typer.Option(False, "--fake", help="Use deterministic fake reasoning."),
    json_output: bool = typer.Option(False, "--json", help="Print the full case as JSON."),
    trace_synthetic: bool = typer.Option(
        False, "--trace-synthetic", help="Include fictional dataset inputs and outputs in traces."
    ),
):
    """Run an in-memory synthetic case; no database needed and no cross-process resume."""
    try:
        application = load_application(applicant_id)
    except (ValueError, OSError):
        raise typer.BadParameter("applicant ID not found or invalid application dataset") from None
    case_id = "APP-" + uuid.uuid4().hex[:12].upper()
    invoke(
        run_case(
            application,
            case_id,
            backend="fake" if fake else "ollama",
            memory=True,
            trace_synthetic=trace_synthetic,
        ),
        json_output,
    )


@cli.command()
def evaluate(
    file: Path = typer.Option(Path("evals/cases.jsonl"), "--file"),
    live: bool = typer.Option(False, "--live"),
    trace_synthetic: bool = typer.Option(False, "--trace-synthetic"),
    output: Path | None = typer.Option(None, "--output", help="Save measured evaluation results."),
):
    """Run golden cases; report routing, measured latency, and recorded tokens."""

    async def run():
        report = []
        for line in file.read_text().splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            case_id = "APP-" + uuid.uuid4().hex[:12].upper()
            started = time.perf_counter()
            result = await run_case(
                Application.model_validate(
                    {
                        **load_application(case["applicant_id"]).model_dump(),
                        **case.get("overrides", {}),
                    }
                ),
                case_id,
                backend="ollama" if live else "fake",
                memory=True,
                trace_synthetic=trace_synthetic and not case.get("overrides"),
            )
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            records = [
                json.loads(item)
                for item in (Path(Settings().trace_dir) / f"{case_id}.jsonl")
                .read_text()
                .splitlines()
            ]
            action = result.get("recommendation", {}).get("action")
            report.append(
                {
                    "name": case["name"],
                    "case_id": case_id,
                    "status": result["status"],
                    "passed": result["status"] == case["expected_status"]
                    and ("expected_action" not in case or action == case["expected_action"]),
                    "latency_ms": elapsed,
                    "input_tokens": sum(item.get("input_tokens", 0) for item in records),
                    "output_tokens": sum(item.get("output_tokens", 0) for item in records),
                    "provider_cost_usd": 0,
                    "backend": "ollama" if live else "fake",
                    "trace_file": str(Path(Settings().trace_dir) / f"{case_id}.jsonl"),
                    "stages_ms": {
                        item["stage"]: item["duration_ms"]
                        for item in records
                        if item.get("event") == "stage_finished"
                        and item.get("stage") in {"intake", "enrichment", "risk", "recommendation"}
                    },
                }
            )
        typer.echo(json.dumps(report, indent=2))
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
        if not report or not all(item["passed"] for item in report):
            raise typer.Exit(1)

    asyncio.run(run())


@cli.command("evaluate-risk")
def evaluate_risk(
    file: Path = typer.Option(Path("evals/risk_cases.jsonl"), "--file"),
    live: bool = typer.Option(False, "--live", help="Use Qwen instead of the fake adapter."),
    output: Path | None = typer.Option(None, "--output", help="Save the benchmark report."),
):
    """Benchmark independent LLM rule review against the authoritative Python policy."""

    async def run():
        settings = Settings()
        cases: list[dict[str, Any]] = []
        for line in file.read_text().splitlines():
            if not line.strip():
                continue
            definition = json.loads(line)
            application = Application.model_validate(definition["application"])
            facts = EnrichmentResult(
                verification=EligibilityVerification.model_validate(definition["verification"]),
                claims=ClaimsHistory.model_validate(definition["claims"]),
            )
            expected = risk_policy(application, facts)
            case_id = "APP-" + uuid.uuid4().hex[:12].upper()
            observer = Observer(settings, case_id, synthetic=True)
            llm: LLMAdapter = (
                OllamaAdapter(
                    settings.ollama_base_url, settings.ollama_model, settings.ollama_timeout_seconds
                )
                if live
                else FakeLLMAdapter()
            )
            started = time.perf_counter()
            candidate = None
            error = None
            agreement = False
            try:
                result = await risk_scoring_agent(application, facts, llm, observer)
                candidate = result.llm_assessment
                agreement = result.agreement
            except RiskDisagreement as exc:
                candidate = exc.candidate
            except Exception as exc:
                error = type(exc).__name__
            finally:
                observer.flush()
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            records = [
                json.loads(item)
                for item in (Path(settings.trace_dir) / f"{case_id}.jsonl").read_text().splitlines()
            ]
            review_outputs = [
                LLMRiskReview.model_validate(item["output"])
                for item in records
                if item.get("event") == "stage_finished"
                and item.get("stage") == "risk_generation"
                and item.get("output")
            ]
            first_candidate = (
                with_review_confidence(assessment_from_review(review_outputs[0]), expected)
                if review_outputs
                else None
            )
            first_agreement = bool(
                first_candidate and not risk_mismatches(first_candidate, expected)
            )
            repair_used = any(
                item.get("event") == "risk_review_repair_scheduled" for item in records
            )
            cases.append(
                {
                    "name": definition["name"],
                    "case_id": case_id,
                    "python": expected.model_dump(mode="json"),
                    "review_flags": risk_review_flags(facts),
                    "first_llm_review": (
                        first_candidate.model_dump(mode="json") if first_candidate else None
                    ),
                    "first_pass_agreement": first_agreement,
                    "first_pass_band_agreement": bool(
                        first_candidate and first_candidate.band == expected.band
                    ),
                    "first_pass_factor_agreement": bool(
                        first_candidate and set(first_candidate.factors) == set(expected.factors)
                    ),
                    "first_pass_absolute_score_error": (
                        round(abs(first_candidate.score - expected.score), 4)
                        if first_candidate
                        else None
                    ),
                    "repair_used": repair_used,
                    "llm": candidate.model_dump(mode="json") if candidate else None,
                    "exact_agreement": agreement,
                    "band_agreement": bool(candidate and candidate.band == expected.band),
                    "factor_agreement": bool(
                        candidate and set(candidate.factors) == set(expected.factors)
                    ),
                    "absolute_score_error": round(abs(candidate.score - expected.score), 4)
                    if candidate
                    else None,
                    "error": error,
                    "latency_ms": elapsed,
                    "input_tokens": sum(item.get("input_tokens", 0) for item in records),
                    "output_tokens": sum(item.get("output_tokens", 0) for item in records),
                    "trace_file": str(Path(settings.trace_dir) / f"{case_id}.jsonl"),
                }
            )
        first_scored = [item for item in cases if item["first_llm_review"] is not None]
        scored = [item for item in cases if item["llm"] is not None]
        total = len(cases)
        summary = {
            "backend": "ollama" if live else "fake",
            "total_cases": total,
            "scored_cases": len(scored),
            "first_pass_agreements": sum(item["first_pass_agreement"] for item in cases),
            "first_pass_agreement_rate": round(
                sum(item["first_pass_agreement"] for item in cases) / total, 4
            )
            if total
            else 0,
            "first_pass_band_agreement_rate": round(
                sum(item["first_pass_band_agreement"] for item in first_scored) / len(first_scored),
                4,
            )
            if first_scored
            else 0,
            "first_pass_factor_agreement_rate": round(
                sum(item["first_pass_factor_agreement"] for item in first_scored)
                / len(first_scored),
                4,
            )
            if first_scored
            else 0,
            "first_pass_mean_absolute_score_error": round(
                sum(item["first_pass_absolute_score_error"] for item in first_scored)
                / len(first_scored),
                4,
            )
            if first_scored
            else None,
            "repairs_attempted": sum(item["repair_used"] for item in cases),
            "repairs_succeeded": sum(
                item["repair_used"] and item["exact_agreement"] for item in cases
            ),
            "operational_review_cases": sum(bool(item["review_flags"]) for item in cases),
            "exact_agreements": sum(item["exact_agreement"] for item in scored),
            "exact_agreement_rate": round(
                sum(item["exact_agreement"] for item in scored) / len(scored), 4
            )
            if scored
            else 0,
            "band_agreements": sum(item["band_agreement"] for item in scored),
            "band_agreement_rate": round(
                sum(item["band_agreement"] for item in scored) / len(scored), 4
            )
            if scored
            else 0,
            "factor_agreements": sum(item["factor_agreement"] for item in scored),
            "factor_agreement_rate": round(
                sum(item["factor_agreement"] for item in scored) / len(scored), 4
            )
            if scored
            else 0,
            "mean_absolute_score_error": round(
                sum(item["absolute_score_error"] for item in scored) / len(scored), 4
            )
            if scored
            else None,
            "model_errors": total - len(scored),
            "total_input_tokens": sum(item["input_tokens"] for item in cases),
            "total_output_tokens": sum(item["output_tokens"] for item in cases),
            "total_latency_ms": round(sum(item["latency_ms"] for item in cases), 2),
        }
        report = {"summary": summary, "cases": cases}
        typer.echo(json.dumps(report, indent=2))
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")

    asyncio.run(run())


@cli.command("failure-demo")
def failure_demo(applicant_id: str = "A-1001", json_output: bool = typer.Option(False, "--json")):
    """Inject three risk-model timeouts; demonstrate bounded retries and human review."""
    application = load_application(applicant_id)
    case_id = "APP-" + uuid.uuid4().hex[:12].upper()
    invoke(
        run_case(
            application,
            case_id,
            backend="fake",
            memory=True,
            trace_synthetic=True,
            llm_override=FakeLLMAdapter("risk", 3),
        ),
        json_output,
    )


if __name__ == "__main__":
    cli()
