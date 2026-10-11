# Multi-Agent Health-Insurance Underwriting Pipeline

This document is the ordered source for the assignment PDF. Add the indicated screenshots after
running the commands in `docs/validation-guide.md`, then export this document to PDF.

## 1. State and memory design

The system uses a **shared case state** managed by LangGraph. Each bounded agent reads the fields it
needs and returns a partial state update. `case_id` is also the LangGraph `thread_id`, isolating one
application from another. PostgreSQL checkpoints make state durable across CLI processes and allow
an interrupted case to resume without replaying completed nodes.

State progression:

```text
application
  -> intake
  -> enrichment + enrichment_summary
  -> risk + comparison metadata
  -> recommendation + final status
```

Why shared state: every later stage needs validated evidence from earlier stages, review failures
must retain their context, and durable checkpoints support inspection, human review, and recovery.
A dynamic supervisor was intentionally avoided because underwriting has a fixed, auditable order.

**Screenshot:** `docs/architecture.md` state/memory section and `CaseState` in `app/models.py`.

## 2. Four standalone agents

`app/agents.py` defines four independently callable functions:

1. `intake_agent` validates and summarizes the submitted application.
2. `enrichment_agent` retrieves synthetic eligibility verification and claims evidence, then asks
   the LLM to summarize only that evidence.
3. `risk_scoring_agent` compares deterministic Python policy with Qwen's independent six-rule
   evidence review. One targeted correction is allowed; unresolved disagreement is escalated.
4. `recommendation_agent` lets Python own action/evidence while Qwen writes a grounded reason and
   confidence value.

Each function makes an LLM call through the `LLMAdapter`. Unit tests call the real agent functions
with `FakeLLMAdapter`, which makes tests deterministic and suitable for CI while proving the LLM
boundary was invoked.

**Screenshot:** the four functions in `app/agents.py`, their tests in `tests/test_agents.py`, and
terminal output showing those tests passing.

## 3. Connected pipeline

LangGraph implements the fixed sequence:

```text
START -> intake -> enrichment -> risk -> recommendation -> END
```

Every node validates state with Pydantic, records its attempt counter, and returns only its state
update. Conditional edges retry transient failures or end processing when status becomes
`COMPLETE` or `HUMAN_REVIEW_REQUIRED`. PostgreSQL is used for durable `submit`, `resume`, and
`status`; the `run` command uses an in-memory checkpointer for demonstrations.

**Screenshot:** graph construction and routing in `app/workflow.py`.

## 4. Tracing and observability

`app/observability.py` creates one root case trace with child spans for each agent, external lookup,
and LLM generation. It records stage name, redacted input/output, duration, token usage, model,
prompt hash, retries, repairs, errors, and escalations. Local JSONL traces are always available;
Langfuse is optional remote visualization. Detailed payloads are enabled only for unchanged
fictional data.

**Screenshot:** tracing implementation and one Langfuse tree or local JSONL trace.

## 5. Three final application traces

| Application | Case ID | Result | All four stages |
|---|---|---|---|
| A-1001 | APP-5B76161FE07C | `RECOMMEND_APPROVE` | Yes |
| A-1004 | APP-A2F6F201E2B3 | `RECOMMEND_APPROVE` | Yes |
| A-1006 | APP-0C12E2C6DC50 | `HUMAN_REVIEW` (fraud flag) | Yes |

Evidence files are `.runs/APP-5B76161FE07C.jsonl`, `.runs/APP-A2F6F201E2B3.jsonl`,
`.runs/APP-0C12E2C6DC50.jsonl`, and `.runs/submission-live-evaluation.json`.

**Screenshots:** the complete nested trace for each case, showing all four stages and token/timing
metadata.

## 6. Cost and latency

Across the three final cases, mean wall latency was **21.762 seconds**, total token usage was
**5,031**, mean usage was **1,677 tokens/case**, and direct Ollama provider charge was **$0**.
Hardware and operating costs are excluded. The system misses an assumed five-second interactive
target. At the measured mean, 1,000 sequential cases take approximately 6.05 hours, fitting an
assumed eight-hour overnight window. Full calculations are in `docs/performance-analysis.md`.

**Screenshot:** `docs/performance-analysis.md` tables and conclusion.

## 7. Simulated failure and escalation

`failure-demo` injects three transient failures into the risk LLM call:

```text
risk attempt 1 -> timeout -> retry
risk attempt 2 -> timeout -> retry
risk attempt 3 -> timeout -> HUMAN_REVIEW_REQUIRED
```

The pipeline does not crash and does not fabricate risk or recommendation fields. Intake and
enrichment remain available for investigation. Evidence: `.runs/APP-9992AD3A4E14.jsonl`.

**Screenshot:** failure handling in `app/workflow.py` and terminal output from
`python -m app.cli failure-demo --json`.

## 8. Passing pipeline and CI

Final local verification: Ruff passed, formatting passed, Mypy passed, **92 tests passed**, two
optional live tests were skipped by default, coverage was **96.09%**, the real PostgreSQL
checkpoint/resume test passed, and all three final Qwen evaluations passed.

GitHub Actions repeats lint, formatting, typing, coverage, PostgreSQL integration, and deterministic
pipeline evaluation on Python 3.12, then uploads `.runs` as an artifact.

**Screenshot:** a green GitHub Actions run for `.github/workflows/ci.yml`. Add this only after
committing and pushing the repository.
