"""Replaceable LLM, eligibility-verification, and claims adapters."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ValidationError

from app.models import Application, ClaimsHistory, EligibilityVerification

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_application(applicant_id: str, path: Path | None = None) -> Application:
    applications = json.loads((path or DATA_DIR / "applications.json").read_text())
    if not isinstance(applications, list) or any(
        not isinstance(record, dict) for record in applications
    ):
        raise ValueError("application dataset must be a list of objects")
    ids = [record.get("applicant_id") for record in applications]
    if any(not isinstance(value, str) for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("application dataset IDs must be unique strings")
    matches = [record for record in applications if record.get("applicant_id") == applicant_id]
    if len(matches) != 1:
        raise ValueError("applicant ID must identify exactly one dataset record")
    return Application.model_validate(matches[0])


class AdapterError(RuntimeError):
    """Safe error message: never include provider response or applicant text."""


class TransientError(AdapterError):
    pass


class MalformedOutput(AdapterError):
    pass


class MissingEvidence(AdapterError):
    pass


def load_records(path: Path) -> dict[str, Any]:
    try:
        records = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise AdapterError("mock dataset unavailable or invalid") from exc
    if not isinstance(records, dict):
        raise AdapterError("mock dataset must be keyed by applicant ID")
    return records


@dataclass
class ModelResponse:
    value: BaseModel
    input_tokens: int = 0
    output_tokens: int = 0


class LLMAdapter(Protocol):
    model: str

    async def generate(
        self, stage: str, prompt: str, payload: dict[str, Any], schema: type[BaseModel]
    ) -> ModelResponse: ...


class VerificationAdapter(Protocol):
    async def get_verification(self, applicant_id: str) -> EligibilityVerification: ...


class ClaimsAdapter(Protocol):
    async def get_history(self, applicant_id: str) -> ClaimsHistory: ...


class OllamaAdapter:
    def __init__(self, base_url: str, model: str, timeout: float = 30):
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout

    async def generate(self, stage, prompt, payload, schema) -> ModelResponse:
        body = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": schema.model_json_schema(),
            "options": {"temperature": 0, "num_predict": 1024},
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({"untrusted_data": payload})},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.base_url + "/api/chat", json=body)
                response.raise_for_status()
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise TransientError("Ollama unavailable or timed out") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {429, 500, 502, 503, 504}:
                raise TransientError("temporary Ollama HTTP failure") from exc
            raise AdapterError(f"Ollama HTTP {exc.response.status_code}") from exc
        try:
            data = response.json()
            value = schema.model_validate_json(data["message"]["content"])
            return ModelResponse(
                value, int(data.get("prompt_eval_count", 0)), int(data.get("eval_count", 0))
            )
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            raise MalformedOutput("invalid structured model response") from exc


class FakeLLMAdapter:
    model = "deterministic-fake"

    def __init__(self, fail_stage: str | None = None, failures: int = 0):
        self.fail_stage, self.failures = fail_stage, failures
        self.calls: list[str] = []

    async def generate(self, stage, prompt, payload, schema) -> ModelResponse:
        self.calls.append(stage)
        if stage == self.fail_stage and self.failures:
            self.failures -= 1
            raise TransientError("simulated model timeout")
        value: dict[str, Any]
        if stage == "intake":
            value = {"validation_notes": ["application validated"], "missing_fields": []}
        elif stage == "enrichment":
            value = {
                "summary": "Validated synthetic eligibility verification and claims records.",
                "evidence": ["verification", "claims"],
            }
        elif stage == "risk":
            evidence = payload["validated_evidence"]
            decisions = []
            for rule in payload["rules"]:
                section, field = rule["evidence_path"].split(".", 1)
                observed = evidence[section][field]
                triggered = (
                    observed == rule["threshold"]
                    if rule["operator"] == "equals"
                    else observed >= rule["threshold"]
                )
                decisions.append(
                    {
                        "code": rule["code"],
                        "triggered": triggered,
                        "evidence_path": rule["evidence_path"],
                        "explanation": "Compared validated evidence with the stated threshold.",
                    }
                )
            value = {
                "rule_decisions": decisions,
                "data_quality_issues": [],
                "reasoning": "Reviewed every health-risk rule against validated evidence.",
            }
        else:
            value = {
                "reason": "Synthetic policy applied to validated evidence.",
                "confidence": 0.9,
            }
        return ModelResponse(schema.model_validate(value))  # Fake tokens are deliberately zero.


class MockVerificationAdapter:
    def __init__(self, path: Path | None = None):
        self.path = path or DATA_DIR / "eligibility_verifications.json"

    async def get_verification(self, applicant_id: str) -> EligibilityVerification:
        record = load_records(self.path).get(applicant_id)
        if record is None:
            raise MissingEvidence("eligibility verification unavailable")
        if not isinstance(record, dict):
            raise AdapterError("invalid eligibility verification record")
        return EligibilityVerification.model_validate(record)


class MockClaimsAdapter:
    def __init__(self, path: Path | None = None):
        self.path = path or DATA_DIR / "claims_history.json"

    async def get_history(self, applicant_id: str) -> ClaimsHistory:
        record = load_records(self.path).get(applicant_id)
        if record is None:
            raise MissingEvidence("claims record unavailable")
        return ClaimsHistory.model_validate(record)
