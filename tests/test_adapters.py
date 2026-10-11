import json

import httpx
import pytest

from app.adapters import (
    AdapterError,
    MalformedOutput,
    MissingEvidence,
    MockClaimsAdapter,
    MockVerificationAdapter,
    OllamaAdapter,
    TransientError,
    load_application,
)
from app.models import IntakeResult


@pytest.mark.asyncio
async def test_adapters_read_json_records(tmp_path):
    path = tmp_path / "verification.json"
    path.write_text(json.dumps({"CUSTOM": {"identity_verified": True, "fraud_flag": False}}))
    assert (await MockVerificationAdapter(path).get_verification("CUSTOM")).identity_verified
    path.write_text(json.dumps({"CUSTOM": {"identity_verified": False, "fraud_flag": False}}))
    assert not (await MockVerificationAdapter(path).get_verification("CUSTOM")).identity_verified


def test_named_dataset_and_unknown_applicant():
    assert load_application("A-1001").full_name == "Alex Morgan"
    with pytest.raises(ValueError):
        load_application("UNKNOWN")


@pytest.mark.parametrize(
    "content",
    [
        '{"A-1001": {}}',
        "[null]",
        '[{"applicant_id": null}]',
        '[{"applicant_id": "A"}, {"applicant_id": "A"}]',
    ],
)
def test_malformed_application_dataset(tmp_path, content):
    path = tmp_path / "applications.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_application("A", path)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["invalid-json", "[]", '{"CUSTOM": []}'])
async def test_invalid_verification_dataset(tmp_path, content):
    path = tmp_path / "verification.json"
    path.write_text(content)
    with pytest.raises(AdapterError):
        await MockVerificationAdapter(path).get_verification("CUSTOM")


@pytest.mark.asyncio
async def test_mocks_are_deterministic():
    verification, claims = MockVerificationAdapter(), MockClaimsAdapter()
    assert await verification.get_verification("A-1001") == await verification.get_verification(
        "A-1001"
    )
    assert await claims.get_history("A-1001") == await claims.get_history("A-1001")
    assert (await verification.get_verification("A-1001")).synthetic
    with pytest.raises(MissingEvidence):
        await verification.get_verification("NEW")
    with pytest.raises(MissingEvidence):
        await claims.get_history("NEW")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,error", [(429, TransientError), (503, TransientError), (404, AdapterError)]
)
async def test_http_failures(monkeypatch, status, error):
    async def post(self, url, json):
        return httpx.Response(status, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    with pytest.raises(error):
        await OllamaAdapter("http://localhost", "qwen").generate(
            "intake", "prompt", {}, IntakeResult
        )


@pytest.mark.asyncio
async def test_ollama_contract_and_tokens(monkeypatch):
    async def post(self, url, json):
        assert json["think"] is False and json["stream"] is False
        assert "properties" in json["format"]
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "message": {"content": '{"validation_notes": [], "missing_fields": []}'},
                "prompt_eval_count": 12,
                "eval_count": 4,
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    result = await OllamaAdapter("http://localhost", "qwen").generate(
        "intake", "prompt", {}, IntakeResult
    )
    assert result.input_tokens == 12 and result.output_tokens == 4


@pytest.mark.asyncio
async def test_invalid_model_output_and_timeout(monkeypatch):
    async def bad(self, url, json):
        return httpx.Response(
            200, request=httpx.Request("POST", url), json={"message": {"content": "{}"}}
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", bad)
    with pytest.raises(MalformedOutput):
        await OllamaAdapter("http://localhost", "qwen").generate("intake", "", {}, IntakeResult)

    async def timeout(self, url, json):
        raise httpx.ReadTimeout("timeout")

    monkeypatch.setattr(httpx.AsyncClient, "post", timeout)
    with pytest.raises(TransientError):
        await OllamaAdapter("http://localhost", "qwen").generate("intake", "", {}, IntakeResult)
