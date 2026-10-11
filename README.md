# Synthetic Health-Insurance Underwriting Pipeline

CLI-first, four-agent health-insurance pre-underwriting demonstration with LangGraph,
Ollama/Qwen, PostgreSQL checkpoints, guardrails, and optional Langfuse tracing. Eligibility
verification and claims evidence are fictional.

## Install

Activate your existing virtual environment and install dependencies:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Python 3.11+ is required. Python 3.12 is the development-container default.

## Try without infrastructure

```bash
python -m app.cli run A-1001 --fake
python -m app.cli run A-1003 --fake
python -m app.cli run A-1004 --fake
python -m app.cli evaluate
python -m app.cli evaluate-risk
```

These commands use deterministic fake model responses and in-memory checkpoints. No database or model download is needed. Fake token usage is zero and fake latency does not predict live model performance. Each run writes detailed redacted JSONL traces to .runs/. The terminal shows one progress line per agent and a concise final summary.

`evaluate-risk` is a separate 20-case benchmark for Python policy/Qwen rule-review agreement.
Without `--live` it validates the harness with the fake adapter; with `--live` it measures real
Qwen, including first-pass agreement and targeted-repair results.

Add --json to run, submit, status, or resume to print the full case as JSON. Use
CONSOLE_LOGS=json for raw terminal logs or CONSOLE_LOGS=quiet to silence progress.
These settings do not disable file traces. Redacted values appear as "[redacted]";
actual missing values remain null. Whole-stage timing includes its nested model call,
so these durations must not be added together.

## Run with Qwen

```bash
docker compose up -d postgres ollama
docker compose exec ollama ollama pull qwen3:8b
python -m app.cli run A-1001
python -m app.cli evaluate --live
python -m app.cli evaluate --live --trace-synthetic --output .runs/submission-live-evaluation.json
python -m app.cli evaluate-risk --live --output .runs/risk-evaluation-live-v3.json
```

The model download is several GB. If Ollama is installed on the host, it can be used instead of the Ollama container.

For a durable case:

```bash
python -m app.cli submit
python -m app.cli submit --applicant-id A-1001
python -m app.cli submit --file tests/fixtures/application.json
python -m app.cli status APP-012345ABCDEF
python -m app.cli resume APP-012345ABCDEF
```

Use the actual case ID printed on submission. Add --fake to submit for a real PostgreSQL run without Ollama. A submitted case uses its recorded backend on resume.

Interactive submit offers Existing applicant or New applicant. Existing applicants need only
their dataset ID; their stored details are loaded automatically. New applicants enter all fields
and must use an unused ID. Missing external records safely route to human review.

## Synthetic data

- data/applications.json contains fictional applicants, including full_name.
- data/eligibility_verifications.json contains identity/fraud records keyed by applicant_id.
- data/claims_history.json contains claims records keyed by applicant_id.
- evals/cases.jsonl references those applicants and records expected outcomes.

Names identify applicants for display and never affect policy scoring. Age, occupation, and
coverage tier are also not scored. Health declarations and evidence are fictional and redacted
from ordinary logs. New interactive applicants need corresponding verification/claims records;
an unknown ID routes to human review rather than generating evidence. All six dataset applicants
have ordinary IDs and complete verification/claims records. Timeout simulation is injected
in unit tests, never encoded in applicant identity or normal data. The run command selects an
applicant; golden evaluations only check results.

Copy .env.example to .env and fill settings as needed. Never commit credentials. The application reads .env, not .env.example. Host settings default to localhost; Compose overrides service addresses inside the development container.

## Architecture

CLI → Pydantic input validation → LangGraph (intake → enrichment → risk → recommendation) → decision gate → CLI result.

- app/models.py: contracts and shared case state.
- app/adapters.py: Ollama/fake LLM plus mock verification and claims interfaces.
- app/agents.py: independently testable bounded reasoning functions.
- app/workflow.py: graph, routing, durable attempt counts, and explicit resume.
- app/guardrails.py: deterministic policy, evidence and consistency checks.
- app/observability.py: JSONL traces and optional Langfuse generation/span observations.
- app/prompts/: versioned instructions; each call records the actual prompt SHA-256.
- app/cli.py: submit, status, resume, run, and evaluation commands.

PostgreSQL stores graph checkpoints with thread_id = case_id. A per-case PostgreSQL advisory lock prevents concurrent processing through the CLI. Completed cases are returned without replay. Resuming after a policy-version change is rejected for manual handling. Interrupting a running model call may replay that unfinished call; completed checkpoints are preserved.

Transient dependencies/model failures get two retries (three total attempts), exponential backoff, and persisted error context. Malformed model output gets one structured repair attempt. Missing evidence, policy contradictions, and low confidence route to human review. A failed enrichment produces no fabricated risk or recommendation.

Each graph invocation has a 20-step safety ceiling. The default retry policy allows up to
12 node executions across the four stages. Retry configuration is limited to 0–2 retries per
stage, keeping valid configurations within this ceiling.

Use --trace-synthetic on run to record readable fictional input/output payloads locally and in
Langfuse. The application must match its dataset record; modified or user-entered records
cannot opt into full capture. Evaluation overrides remain redacted.

The synthetic health-risk index is illustrative and is not a medical diagnosis, actuarial model,
premium quote, or binding coverage decision. Python calculates the authoritative score. Qwen
reviews every published rule against normalized evidence, and Python derives a second score from
Qwen's selected factors. Exact score/band/factor agreement continues the workflow. A mismatch gets
one targeted correction call; continued disagreement preserves both assessments and requires human
review. Identity and fraud signals are separate operational review flags and do not inflate medical
risk. Risk-review confidence is deterministic completeness/agreement metadata, not a probability.
Recommendation confidence remains model-reported and uncalibrated.

To reproduce a controlled agent failure:

```bash
python -m app.cli failure-demo --json
```

This uses fake responses and deliberately raises three risk-model timeouts. It should finish
with HUMAN_REVIEW_REQUIRED, three recorded errors, and no fabricated risk/recommendation.
The three-case live evaluation uses real Ollama; the failure demo is intentionally deterministic.

See docs/validation-guide.md for the reproducible verification sequence and
docs/performance-analysis.md for actual live measurements.

Langfuse is enabled only when both credentials are configured. Logs omit raw applicant fields and arbitrary model text. Local trace files are ignored by Git. Case state in PostgreSQL contains applicant inputs and must be treated as sensitive.

See docs/underwriting-policy.md for the explicit synthetic policy.

## Test

```bash
python -m pytest -q --cov=app --cov-report=term-missing
python -m ruff check app tests
python -m ruff format --check app tests
```

PostgreSQL integration tests opt in with RUN_POSTGRES_TESTS=1. They create isolated test cases using random identifiers.

```bash
RUN_POSTGRES_TESTS=1 python -m pytest -q -m integration
```

## Cost and latency

evaluate reports wall-clock latency and model token counts per case. Local Ollama has zero direct provider token charges; hardware/runtime costs are not estimated. Fake-mode results validate routing, not model quality. Compare live measurements against your actual response-time target before judging real-time versus batch suitability.

## Limits

This is a single-user CLI demo, not an authenticated multi-user insurance service. Human-review outcomes retain reasons and error history but have no review-resolution UI. Policy and mock records are deliberately illustrative. Process crashes between a provider call and its checkpoint can repeat that call; this does not provide exactly-once external execution.
