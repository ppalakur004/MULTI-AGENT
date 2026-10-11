import os

import pytest

from app.cli import run_case
from app.models import Application


@pytest.mark.ollama
@pytest.mark.skipif(os.getenv("RUN_OLLAMA_TESTS") != "1", reason="Live Ollama opt-in")
@pytest.mark.asyncio
async def test_live_four_agent_case():
    result = await run_case(
        Application(
            full_name="Test Applicant",
            applicant_id="A-1001",
            age=35,
            coverage_tier="STANDARD",
            smoker=False,
            declared_chronic_conditions=0,
            hospitalizations_last_2_years=0,
        ),
        "APP-012345ABCDEF",
        backend="ollama",
        memory=True,
    )
    assert result["status"] == "COMPLETE"
    assert result["recommendation"]["action"] == "RECOMMEND_APPROVE"
