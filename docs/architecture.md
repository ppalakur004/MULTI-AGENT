# Architecture

CLI → Pydantic health-application validation → LangGraph → final decision guardrails → CLI output.

## State decision and rationale

We chose shared, typed case state rather than explicit agent-to-agent message passing.
Enrichment and risk need overlapping facts; recommendation needs a validated risk assessment.
A single case record simplifies inspection and checkpoint resume without reassembling handoff
history. The tradeoff is broader node access to state. We constrain it with stage-specific
validated inputs and partial returned updates, and isolate each case using thread_id = case_id.
Qwen has no implicit memory: each call receives only the payload constructed for its task.

| Stage | Reads | Writes |
|---|---|---|
| Intake | application | intake |
| Enrichment | application; intake success is enforced by ordering | enrichment, enrichment_summary |
| Risk | application, enrichment | risk |
| Recommendation | risk | recommendation, terminal status |
| All nodes | attempts, errors | own attempt count, failures, updated_at |

PostgreSQL makes case-scoped state durable; it is not a cross-case knowledge store.
Transient retry/error updates are checkpointed graph steps. Explicit resume continues unfinished
nodes. Interrupted provider calls can repeat; this is not exactly-once execution.

Four standalone agents run in order: intake, enrichment, risk assessment, and recommendation. Each validates its inputs and outputs and returns only its own state updates.

Ollama/Qwen supplies bounded reasoning through the LLM adapter. Eligibility-verification and
claims adapters supply validated fictional facts; unavailable evidence stays missing.

PostgreSQL stores LangGraph checkpoints, with thread_id equal to case_id. Resume is explicit. Transient failures get bounded retries with backoff; unresolved failures route to a persisted human-review outcome.

Prompts live in app/prompts/. Git tracks prompt changes, and traces record the prompt filename and content hash alongside model, tokens, latency, errors, and retries. Applicant data is supplied separately as untrusted data. Logs and Langfuse use redacted summaries.

run --trace-synthetic permits readable I/O only when the supplied application exactly matches
a fictional dataset record. Ordinary submissions remain redacted. Stage and generation I/O are
captured separately. Failed generation attempts retain error metadata; retries are explicit in
local events. No raw credentials enter traces.

Only three adapter interfaces are planned: LLM, eligibility verification, and claims. No queue
or separate worker is included.

The CLI invokes the graph directly. Durable submissions use PostgreSQL; demo and evaluation runs
use in-memory checkpoints and cannot resume across processes. Each transient failure is recorded
as a graph update before a retry, retaining attempt counts across checkpointed interruptions.

The CLI holds a PostgreSQL advisory lock for the case during execution. Explicit resume passes
None to LangGraph, so checkpointed stages are not replayed. A crash during an unfinished provider
call can repeat that call.

The policy is synthetic-health-reviewed-v3, documented in underwriting-policy.md. Python calculates
the authoritative health-risk score. Qwen independently reviews all six rules against normalized,
validated evidence and returns a decision plus evidence path for every rule; it does not calculate
the score. Python derives a reviewer score from Qwen's selected factors and compares the exact score,
band, and factor set. A mismatch gets one targeted correction call; continued disagreement stores
both assessments and routes to human review. Identity and fraud signals are separate operational
review flags and never change the health-risk score. Medium risk also requires human review.
Recommendation guardrails verify evidence, policy action, and minimum confidence. Failed enrichment
never supplies a placeholder score.

Tracing uses case-correlated local JSONL plus optional Langfuse spans and generations.
Each execution has one root span named Underwriting / APP-..., with all agent stages and their
generations/lookups nested beneath it. The root records final status, action, error count, and
workflow duration (including retries and checkpoints). Each invocation uses a new trace ID;
case_id metadata links submissions and later resumes. Status reads do not create run traces.
Telemetry failures do not change underwriting outcomes. Live generation metadata includes token counts;
fake runs report zero tokens and cannot establish real model latency or quality.
