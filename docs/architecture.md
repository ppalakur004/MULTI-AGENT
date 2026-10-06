# Architecture

CLI → Pydantic input validation → LangGraph → final decision guardrails → CLI output.

Four standalone agents run in order: intake, enrichment, risk assessment, and recommendation. Each validates its inputs and outputs and returns only its own state updates.

Ollama/Qwen supplies bounded reasoning through the LLM adapter. Credit and claims adapters supply validated synthetic facts; unavailable evidence stays missing.

PostgreSQL stores LangGraph checkpoints, with thread_id equal to case_id. Resume is explicit. Transient failures get bounded retries with backoff; unresolved failures route to a persisted human-review outcome.

Prompts live in app/prompts/. Git tracks prompt changes, and traces record the prompt filename and content hash alongside model, tokens, latency, errors, and retries. Applicant data is supplied separately as untrusted data. Logs and Langfuse use redacted summaries.

Only three adapter interfaces are planned: LLM, credit, and claims. No queue or separate worker is included.

All application modules, prompts, tests, fixtures, and evals are placeholders.
