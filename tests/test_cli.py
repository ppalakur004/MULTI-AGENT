import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typer.testing import CliRunner

from app import cli as module
from app.config import Settings
from app.models import Application

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setattr(
        module,
        "Settings",
        lambda: Settings(
            _env_file=None,
            trace_dir=str(tmp_path),
            retry_backoff_seconds=0,
            langfuse_public_key=None,
            langfuse_secret_key=None,
        ),
    )


@pytest.mark.parametrize(
    "applicant_id,status",
    [
        ("A-1001", "COMPLETE"),
        ("A-1002", "HUMAN_REVIEW_REQUIRED"),
        ("A-1003", "COMPLETE"),
    ],
)
def test_run_applicant(applicant_id, status):
    result = runner.invoke(module.cli, ["run", applicant_id, "--fake"])
    assert result.exit_code == 0, result.output
    assert status in result.stdout


def test_clean_output_and_json_option():
    result = runner.invoke(module.cli, ["run", "A-1001", "--fake"])
    assert "Applicant: Alex Morgan" in result.stdout
    assert "stage_finished" not in result.stderr
    assert result.stderr.count(": complete") == 4
    result = runner.invoke(module.cli, ["run", "A-1001", "--fake", "--json"])
    assert json.loads(result.stdout)["application"]["full_name"] == "Alex Morgan"


def test_verified_synthetic_trace_has_readable_handoffs():
    result = runner.invoke(module.cli, ["run", "A-1001", "--fake", "--trace-synthetic", "--json"])
    assert result.exit_code == 0, result.output
    case_id = json.loads(result.stdout)["case_id"]
    records = [
        json.loads(line)
        for line in (module.Path(module.Settings().trace_dir) / f"{case_id}.jsonl")
        .read_text()
        .splitlines()
    ]
    intake = next(row for row in records if row.get("stage") == "intake_generation")
    assert intake["input"]["full_name"] == "Alex Morgan"
    risk = next(row for row in records if row.get("stage") == "risk_generation")
    assert "policy_result" not in risk["input"]
    assert risk["input"]["policy_version"] == "synthetic-health-reviewed-v3"
    assert len(risk["output"]["rule_decisions"]) == 6
    assert not any(item["triggered"] for item in risk["output"]["rule_decisions"])


def test_failure_demo_escalates_without_fabricated_risk():
    result = runner.invoke(module.cli, ["failure-demo", "--json"])
    assert result.exit_code == 0, result.output
    state = json.loads(result.stdout)
    assert state["status"] == "HUMAN_REVIEW_REQUIRED"
    assert state["attempts"]["risk"] == 3
    assert len(state["errors"]) == 3
    assert "risk" not in state and "recommendation" not in state


def test_interactive_fields_allow_correction(monkeypatch):
    async def capture(application, case_id, **kwargs):
        assert application.age == 22 and application.coverage_tier == "STANDARD"
        return {"status": "HUMAN_REVIEW_REQUIRED"}

    monkeypatch.setattr(module, "run_case", capture)
    result = runner.invoke(
        module.cli,
        ["submit", "--fake"],
        input="2\nNEW/ID\nA-9001\n   \nTest User\n17\n22\nSTANDARD\nfalse\n-1\n0\n-1\n0\nEngineer\n",
    )
    assert result.exit_code == 0, result.output
    assert "Invalid age" in result.stdout
    assert "Invalid declared chronic conditions" in result.stdout


@pytest.mark.asyncio
async def test_detailed_tracing_rejects_user_modified_data():
    application = module.load_application("A-1001").model_copy(
        update={"full_name": "Actual Person"}
    )
    with pytest.raises(ValueError):
        await module.run_case(
            application, "APP-111111111111", backend="fake", memory=True, trace_synthetic=True
        )


def test_bad_arguments():
    for args in [["run", "unknown", "--fake"], ["resume", "../x"], ["status", "bad"]]:
        assert runner.invoke(module.cli, args).exit_code != 0


def test_interactive_menu_recovers_from_invalid_choices(monkeypatch):
    async def capture(application, case_id, **kwargs):
        assert application.applicant_id == "A-1001"
        return {"status": "COMPLETE"}

    monkeypatch.setattr(module, "run_case", capture)
    result = runner.invoke(
        module.cli, ["submit", "--fake"], input="9\n1\nUNKNOWN\n2\nA-1001\n1\nA-1001\n"
    )
    assert result.exit_code == 0, result.output
    assert "Please choose 1 or 2" in result.stdout
    assert "Applicant not found" in result.stdout
    assert "That ID already exists" in result.stdout


def test_submit_and_interactive(monkeypatch, tmp_path):
    saver = InMemorySaver()

    @asynccontextmanager
    async def persistence(*args):
        yield saver

    monkeypatch.setattr(module, "persistence", persistence)
    file = tmp_path / "input.json"
    file.write_text(
        json.dumps(
            dict(
                applicant_id="A-1001",
                full_name="Test Applicant",
                age=35,
                coverage_tier="STANDARD",
                smoker=False,
                declared_chronic_conditions=0,
                hospitalizations_last_2_years=0,
            )
        )
    )
    result = runner.invoke(module.cli, ["submit", "--fake", "--file", str(file)])
    assert result.exit_code == 0, result.output
    assert "RECOMMEND_APPROVE" in result.stdout
    result = runner.invoke(
        module.cli,
        ["submit", "--fake"],
        input="2\nA-9001\nTest Applicant\n35\nSTANDARD\nfalse\n0\n0\nEngineer\n",
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(module.cli, ["submit", "--fake"], input="1\nA-1001\n")
    assert result.exit_code == 0, result.output
    assert "Alex Morgan" in result.stdout
    assert "Full name:" not in result.stdout
    file.write_text("{}")
    assert runner.invoke(module.cli, ["submit", "--file", str(file)]).exit_code == 2


@pytest.mark.asyncio
async def test_status_resume_and_missing(monkeypatch):
    saver = InMemorySaver()

    @asynccontextmanager
    async def persistence(*args):
        yield saver

    monkeypatch.setattr(module, "persistence", persistence)
    application = Application(
        full_name="Test Applicant",
        applicant_id="A-1001",
        age=35,
        coverage_tier="STANDARD",
        smoker=False,
        declared_chronic_conditions=0,
        hospitalizations_last_2_years=0,
    )
    first = await module.run_case(application, "APP-111111111111", backend="fake")
    assert (
        await module.run_case(None, "APP-111111111111", backend="ollama", read_only=True) == first
    )
    assert await module.run_case(None, "APP-111111111111", backend="ollama") == first
    with pytest.raises(ValueError):
        await module.run_case(None, "APP-222222222222", backend="ollama")


def test_command_status_resume_and_error(monkeypatch):
    async def result(*args, **kwargs):
        return {"status": "COMPLETE"}

    monkeypatch.setattr(module, "run_case", result)
    for cmd in ["status", "resume"]:
        assert runner.invoke(module.cli, [cmd, "APP-111111111111"]).exit_code == 0

    async def failure(*args, **kwargs):
        raise RuntimeError("secret")

    monkeypatch.setattr(module, "run_case", failure)
    result = runner.invoke(module.cli, ["run", "A-1001", "--fake"])
    assert result.exit_code == 1 and "secret" not in result.output


def test_evaluation():
    result = runner.invoke(module.cli, ["evaluate"])
    assert result.exit_code == 0, result.output
    assert '"passed": true' in result.stdout


def test_risk_evaluation_uses_twenty_cases():
    result = runner.invoke(module.cli, ["evaluate-risk"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["summary"]["total_cases"] == 20
    assert report["summary"]["exact_agreement_rate"] == 1.0


def test_submit_dataset_record(monkeypatch):
    async def capture(application, case_id, **kwargs):
        assert application.full_name == "Alex Morgan"
        assert application.applicant_id == "A-1001"
        return {"status": "COMPLETE"}

    monkeypatch.setattr(module, "run_case", capture)
    assert (
        runner.invoke(module.cli, ["submit", "--fake", "--applicant-id", "A-1001"]).exit_code == 0
    )
    assert runner.invoke(module.cli, ["submit", "--applicant-id", "UNKNOWN"]).exit_code == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("acquired", [True, False])
async def test_postgres_lock_wiring(monkeypatch, acquired):
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone = AsyncMock(return_value=(acquired,))
    conn.execute = AsyncMock(return_value=cursor)
    conn_context = MagicMock()
    conn_context.__aenter__ = AsyncMock(return_value=conn)
    conn_context.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(module.AsyncConnection, "connect", AsyncMock(return_value=conn_context))
    saver = MagicMock()
    saver.setup = AsyncMock()
    saver_context = MagicMock()
    saver_context.__aenter__ = AsyncMock(return_value=saver)
    saver_context.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(module.AsyncPostgresSaver, "from_conn_string", lambda *args: saver_context)
    if acquired:
        async with module.persistence(Settings(_env_file=None), "APP-111111111111", False) as value:
            assert value is saver
    else:
        with pytest.raises(ValueError):
            async with module.persistence(Settings(_env_file=None), "APP-111111111111", False):
                pass
