# Reproducible project verification

These commands reproduce the implementation evidence used by `docs/submission.md`. Capture the
indicated code, terminal, trace, and CI views in that order when exporting the assignment PDF.

## 1. Design

Read architecture.md for the shared-state decision, rationale, and per-stage ownership table.
Read underwriting-policy.md for the explicitly illustrative scoring rules.

## 2. Four isolated agents

```bash
python -m pytest tests/test_agents.py -v
```

There are separately named intake, enrichment, risk, and recommendation tests. Each calls the
real agent function using controlled input without invoking preceding agents. Unit tests use
FakeLLMAdapter; live evaluations separately establish actual Qwen executions.

## 3. Full local verification

```bash
python -m ruff check app tests
python -m ruff format --check app tests
python -m mypy app --ignore-missing-imports
python -m pytest -q --cov=app --cov-report=term-missing --cov-fail-under=95
```

## 4. Three live application traces

```bash
python -m app.cli evaluate --live --trace-synthetic --output .runs/submission-live-evaluation.json
```

Only unchanged fictional dataset records can opt into full tracing. Ordinary user input stays
redacted. Each case has one Langfuse root and nested agent/generation observations. .runs/
contains complete JSONL execution records. The measurement report distinguishes fake from live.
The final evaluation contains A-1001 and A-1004 approvals plus A-1006 operational human review;
all three execute all four stages. Read performance-analysis.md for measured results and
real-time versus batch reasoning. The optional rule-review benchmark compares first-pass and
post-repair Qwen agreement:

```bash
python -m app.cli evaluate-risk --live --output .runs/risk-evaluation-live-v3.json
```

## 5. Deliberate failure, recovery, and escalation

```bash
python -m app.cli failure-demo --json
python -m pytest tests/test_workflow.py -v
```

The demo injects a risk-model timeout for all three attempts. Expect two retry events,
HUMAN_REVIEW_REQUIRED, three errors, and no invented risk or recommendation. The workflow tests
also cover eventual recovery, interruption after enrichment, unchanged completed-case resume,
case isolation, and durable retry budgets.

## 6. Actual PostgreSQL checks

```bash
docker compose up -d postgres
RUN_POSTGRES_TESTS=1 python -m pytest tests/test_postgres.py -v
python -m app.cli submit --applicant-id A-1001
python -m app.cli status APP-YOUR_CASE_ID --json
```

The integration test creates its own case, pauses after enrichment, closes the checkpointer,
opens a new one, resumes, and verifies that completed agents were not replayed. A competing
execution is rejected by the per-case database lock.

## 7. CI

.github/workflows/ci.yml installs dependencies, checks lint/format/types, enforces 95% coverage,
executes real PostgreSQL integration tests, and runs fake golden evaluations. These steps were
verified locally. A remote GitHub Actions result must be verified after committing and pushing
the reviewed changes; local success is not proof of remote CI.

## Remaining limitations

Risk weights are synthetic, and recommendation confidence is uncalibrated. Rule decisions are
validated, but free-text rationale is not checked claim-by-claim. Human review retains context but
has no resolution UI. New applicant records
remain in case state rather than being inserted into the seed JSON dataset. Model/prompt changes
across resume are not frozen. Local logs/checkpoints do not have automatic retention.
