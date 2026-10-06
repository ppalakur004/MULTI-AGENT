# Underwriting Pipeline

Project scaffold for the underwriting workflow. Application, prompt, test, and data files are
placeholders; business logic is intentionally not implemented yet.

## Architecture

```text
CLI → Input validation → LangGraph workflow → Final guardrails → CLI result
                        │
                        ├── Credit adapter → mock credit service
                        ├── Claims adapter → mock claims service
                        ├── LLM adapter → Ollama/Qwen
                        └── Checkpointer → PostgreSQL

Cross-cutting: Pydantic validation, structured logs, Langfuse traces, retries, human review
```

The workflow has four fixed stages: intake, enrichment, risk assessment, and recommendation.
LangGraph controls sequencing and stores checkpoints using `case_id` as its `thread_id`.
The CLI invokes the workflow directly. Credit and claims provide facts. Qwen performs
bounded reasoning over validated facts and never invents external evidence.

## Minimal application structure

The application stays flat:

```text
app/
├── cli.py           # input and output only
├── models.py        # Pydantic contracts and workflow state
├── agents.py        # four standalone agent functions
├── adapters.py      # interfaces plus Ollama, credit, and claims adapters
├── workflow.py      # four nodes, routing, retries, safety gates
├── guardrails.py    # evidence and decision consistency checks
├── config.py        # environment settings
├── observability.py # logs, Langfuse, prompt hashes
└── prompts/
    ├── intake.md
    ├── enrichment.md
    ├── risk.md
    └── recommendation.md
```

Only three replaceable adapter boundaries are needed initially:

- `LLMAdapter` for Ollama/Qwen and a deterministic test fake.
- `CreditAdapter` for the mock credit source and a future HTTP implementation.
- `ClaimsAdapter` for the mock claims source and a future HTTP implementation.

PostgreSQL checkpointing should use LangGraph's provided checkpointer directly. It does not need a
custom repository abstraction unless a separate business database is later introduced. Logging and
Langfuse also do not need generic interfaces yet; small wrappers are enough.

## Included setup

- `requirements.txt` and `requirements-dev.txt` — runtime and development dependencies
- `.env.example` — local configuration contract
- `docker-compose.yml` — PostgreSQL, Ollama, and development container
- `.devcontainer/` — Python 3.12 development environment
- `.gitignore` and `.dockerignore` — generated-file exclusions

No webhook system, event bus, supervisor agent, browser frontend, or document-processing stack is
planned for v1.

Tests are grouped by module in `tests/`, synthetic records belong in `tests/fixtures/`, and
evaluation scenarios belong in `evals/cases.jsonl`. See [architecture](docs/architecture.md) and
the pending [underwriting policy](docs/underwriting-policy.md).

## Install dependencies

Use Python 3.12 (Python 3.11 or newer is required):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Development requirements include runtime requirements. For runtime dependencies alone,
use `python -m pip install -r requirements.txt`.
