from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.observability import Observer, summarize


def test_redaction():
    result = summarize(
        {
            "applicant_id": "secret",
            "full_name": "secret",
            "occupation": "secret",
            "age": 35,
            "reason": "secret",
            "confidence": 0.9,
            "evidence": ["secret"],
        }
    )
    assert "secret" not in str(result)
    assert result["confidence"] == 0.9
    assert summarize({"smoker": True}) == {}
    assert summarize({"smoker": None}) == {}


@pytest.mark.parametrize("mode", ["quiet", "json", "progress"])
def test_console_mode_preserves_file_traces(tmp_path, capsys, mode):
    observer = Observer(
        Settings(_env_file=None, trace_dir=str(tmp_path), console_logs=mode), "APP-111111111111"
    )
    observer.event("stage_finished", stage="verification_lookup", duration_ms=1, model=None)
    assert observer.path.exists()
    assert '"model": null' not in observer.path.read_text()
    assert bool(capsys.readouterr().err) == (mode == "json")


def test_langfuse_wiring_and_flush(monkeypatch, tmp_path):
    client = MagicMock()
    client.get_current_trace_id.return_value = None

    @contextmanager
    def span(**kwargs):
        assert kwargs["trace_context"]["trace_id"]
        yield MagicMock()

    client.start_as_current_observation.side_effect = span
    monkeypatch.setattr("langfuse.Langfuse", lambda **kwargs: client)
    observer = Observer(
        Settings(
            _env_file=None,
            trace_dir=str(tmp_path),
            langfuse_public_key="test",
            langfuse_secret_key="test",
        ),
        "APP-111111111111",
    )
    with observer.span("generation", model="fake", version="hash") as data:
        data.update(input_tokens=1, output_tokens=2)
    observer.flush()
    client.flush.assert_called_once()
    client.get_current_trace_id.assert_not_called()
    client.flush.side_effect = RuntimeError()
    observer.flush()


def test_nested_observation_inherits_active_parent(tmp_path):
    from opentelemetry.trace import NonRecordingSpan, SpanContext, use_span

    observer = Observer(Settings(_env_file=None, trace_dir=str(tmp_path)), "APP-111111111111")
    observer.client = MagicMock()
    parent = NonRecordingSpan(SpanContext(trace_id=1, span_id=2, is_remote=False))
    with use_span(parent):
        with observer.span("risk_generation", model="fake"):
            pass
    kwargs = observer.client.start_as_current_observation.call_args.kwargs
    assert "trace_context" not in kwargs
    observer.client.get_current_trace_id.assert_not_called()


def test_root_stage_generation_tree(tmp_path):
    from opentelemetry.trace import NonRecordingSpan, SpanContext, get_current_span, use_span

    observer = Observer(Settings(_env_file=None, trace_dir=str(tmp_path)), "APP-111111111111")
    client = MagicMock()
    observations = []
    root_span = MagicMock()

    @contextmanager
    def create(**kwargs):
        parent = get_current_span().get_span_context()
        number = len(observations) + 1
        observations.append((kwargs["name"], parent.span_id))
        trace_id = (
            int(kwargs["trace_context"]["trace_id"], 16)
            if "trace_context" in kwargs
            else parent.trace_id
        )
        with use_span(
            NonRecordingSpan(SpanContext(trace_id=trace_id, span_id=number, is_remote=False))
        ):
            yield root_span if number == 1 else MagicMock()

    client.start_as_current_observation.side_effect = create
    observer.client = client
    with observer.span("Underwriting / APP-111111111111", root=True) as run:
        with observer.span("intake"):
            with observer.span("intake_generation", model="fake"):
                pass
        run["output"] = {"status": "COMPLETE"}
    assert observations == [
        ("Underwriting / APP-111111111111", 0),
        ("intake", 1),
        ("intake_generation", 2),
    ]
    assert root_span.update.call_args.kwargs["output"] == {"status": "COMPLETE"}
    root_span.update_trace.assert_not_called()
    other = Observer(Settings(_env_file=None, trace_dir=str(tmp_path)), observer.case_id)
    assert observer.trace_id != other.trace_id


def test_failed_telemetry_does_not_break_business_work(tmp_path):
    observer = Observer(Settings(_env_file=None, trace_dir=str(tmp_path)), "APP-111111111111")
    client = MagicMock()
    client.start_as_current_observation.side_effect = RuntimeError()
    observer.client = client
    with observer.span("stage"):
        pass
    with pytest.raises(ValueError):
        with observer.span("stage"):
            raise ValueError("private")
    assert "private" not in observer.path.read_text()


def test_unwritable_trace(tmp_path):
    file = tmp_path / "file"
    file.write_text("x")
    observer = Observer(Settings(_env_file=None, trace_dir=str(file)), "APP-111111111111")
    observer.event("test")
