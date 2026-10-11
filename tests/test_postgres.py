import os
import uuid

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.adapters import FakeLLMAdapter, MockClaimsAdapter, MockVerificationAdapter
from app.cli import persistence
from app.config import Settings
from app.guardrails import POLICY_VERSION
from app.models import Application
from app.observability import Observer
from app.workflow import build_graph, execute

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_POSTGRES_TESTS") != "1", reason="PostgreSQL opt-in"),
]


@pytest.mark.asyncio
async def test_real_checkpoint_resume_and_lock(tmp_path):
    settings = Settings(_env_file=None, trace_dir=str(tmp_path), retry_backoff_seconds=0)
    case_id = "APP-" + uuid.uuid4().hex[:12].upper()
    llm = FakeLLMAdapter()
    application = Application(
        full_name="Test Applicant",
        applicant_id="A-1001",
        age=35,
        coverage_tier="STANDARD",
        smoker=False,
        declared_chronic_conditions=0,
        hospitalizations_last_2_years=0,
    )
    state = {
        "case_id": case_id,
        "application": application.model_dump(),
        "status": "RUNNING",
        "errors": [],
        "attempts": {},
        "policy_version": POLICY_VERSION,
    }
    async with persistence(settings, case_id, False) as saver:
        graph = build_graph(
            settings,
            llm,
            MockVerificationAdapter(),
            MockClaimsAdapter(),
            Observer(settings, case_id),
            saver,
            interrupt_after=["enrichment"],
        )
        partial = await execute(graph, case_id, state)
        assert "risk" not in partial
        with pytest.raises(ValueError):
            async with persistence(settings, case_id, False):
                pass
    async with AsyncPostgresSaver.from_conn_string(settings.database_url) as saver:
        graph = build_graph(
            settings,
            llm,
            MockVerificationAdapter(),
            MockClaimsAdapter(),
            Observer(settings, case_id),
            saver,
        )
        result = await execute(graph, case_id)
        assert result["status"] == "COMPLETE"
        assert llm.calls == ["intake", "enrichment", "risk", "recommendation"]
