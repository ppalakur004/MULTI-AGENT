import pytest
from pydantic import ValidationError

from app.config import Settings


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_agent_retries", 3),
        ("ollama_timeout_seconds", float("inf")),
        ("retry_backoff_seconds", float("nan")),
    ],
)
def test_invalid_execution_configuration(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})
