"""Safe local JSONL traces and optional Langfuse observations."""

import json
import logging
import sys
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from opentelemetry.trace import get_current_span

from app.config import Settings
from app.models import now


def summarize(value: Any) -> Any:
    """Keep structure and counts; never export arbitrary strings or applicant values."""
    safe_keys = {"score", "band", "action", "confidence", "synthetic", "status"}
    if isinstance(value, dict):
        return {
            key: item
            if key in safe_keys and isinstance(item, (str, int, float, bool))
            else summarize(item)
            for key, item in value.items()
            if key
            not in {
                "applicant_id",
                "full_name",
                "occupation",
                "age",
                "coverage_tier",
                "smoker",
                "declared_chronic_conditions",
                "hospitalizations_last_2_years",
                "application",
            }
        }
    if isinstance(value, list):
        return {"count": len(value)}
    return None if value is None else "[redacted]"


class Observer:
    def __init__(self, settings: Settings, case_id: str, *, synthetic: bool = False):
        self.case_id = case_id
        self.synthetic = synthetic
        self.trace_id = uuid.uuid4().hex
        self.console_logs = settings.console_logs
        self.path = Path(settings.trace_dir) / f"{case_id}.jsonl"
        self.client: Any = None
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            from langfuse import Langfuse

            try:
                self.client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key.get_secret_value(),
                    base_url=settings.langfuse_base_url,
                )
            except Exception:
                logging.getLogger(__name__).warning("Langfuse initialization failed")

    def capture(self, value: Any) -> Any:
        """Full I/O is opt-in for a verified fictional dataset case."""
        return value if self.synthetic else summarize(value)

    def event(self, event: str, **safe_fields: Any) -> None:
        record = {
            "timestamp": now(),
            "case_id": self.case_id,
            "event": event,
            "data_classification": "synthetic" if self.synthetic else "redacted",
            **{key: value for key, value in safe_fields.items() if value is not None},
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, default=str) + "\n")
        except OSError:
            logging.getLogger(__name__).warning("Local trace file unavailable")
        if self.console_logs == "json":
            print(json.dumps(record, default=str), file=sys.stderr)
        elif self.console_logs == "progress":
            stage = safe_fields.get("stage")
            if event in {"case_started", "case_resumed"}:
                print(
                    f"Case {self.case_id} ({safe_fields.get('backend', 'unknown')})",
                    file=sys.stderr,
                )
            elif event == "stage_finished" and stage in {
                "intake",
                "enrichment",
                "risk",
                "recommendation",
            }:
                outcome = "failed" if "error_class" in safe_fields else "complete"
                print(
                    f"  {stage.capitalize()}: {outcome} ({safe_fields['duration_ms'] / 1000:.2f}s)",
                    file=sys.stderr,
                )
            elif event == "retry_scheduled":
                print(
                    f"  {stage}: retry scheduled after attempt {safe_fields['attempt']}",
                    file=sys.stderr,
                )
            elif event == "human_review_required":
                print(
                    f"  Human review required: {stage} ({safe_fields['error_class']})",
                    file=sys.stderr,
                )

    @contextmanager
    def span(
        self, name: str, *, model: str | None = None, version: str | None = None, root: bool = False
    ):
        start = time.perf_counter()
        metadata: dict[str, Any] = {}
        context = None
        span: Any = None
        if self.client:
            try:
                context = ExitStack()
                observation = self.client.start_as_current_observation(
                    name=name,
                    as_type="generation" if model else "span",
                    **(
                        {"trace_context": {"trace_id": self.trace_id}}
                        if root or not get_current_span().get_span_context().is_valid
                        else {}
                    ),
                    metadata={
                        "case_id": self.case_id,
                        "prompt_hash": version,
                        "data_classification": "synthetic" if self.synthetic else "redacted",
                    },
                    **({"model": model, "version": version} if model else {}),
                )
                span = context.enter_context(observation)
                if root:
                    from langfuse import propagate_attributes

                    context.enter_context(
                        propagate_attributes(
                            trace_name=name,
                            session_id=self.case_id,
                            metadata={
                                "case_id": self.case_id,
                                "data_classification": "synthetic"
                                if self.synthetic
                                else "redacted",
                            },
                        )
                    )
            except Exception:
                if context:
                    context.close()
                context = None
                logging.getLogger(__name__).warning("Langfuse observation unavailable")
        try:
            yield metadata
        except Exception as exc:
            metadata["error_class"] = type(exc).__name__
            self.event("stage_failed", stage=name, error_class=type(exc).__name__)
            raise
        finally:
            if context:
                try:
                    span.update(
                        input=metadata.get("input"),
                        output=metadata.get("output"),
                        **(
                            {
                                "usage_details": {
                                    "input": metadata.get("input_tokens", 0),
                                    "output": metadata.get("output_tokens", 0),
                                },
                                "cost_details": {"total": 0},
                            }
                            if model
                            else {}
                        ),
                        **(
                            {"level": "ERROR", "status_message": metadata["error_class"]}
                            if "error_class" in metadata
                            else {}
                        ),
                    )
                except Exception:
                    logging.getLogger(__name__).warning("Langfuse update failed")
                finally:
                    try:
                        context.__exit__(None, None, None)
                    except Exception:
                        logging.getLogger(__name__).warning("Langfuse observation close failed")
            self.event(
                "stage_finished",
                stage=name,
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
                model=model,
                prompt_hash=version,
                **metadata,
            )

    def flush(self) -> None:
        if self.client:
            try:
                self.client.flush()
            except Exception:
                logging.getLogger(__name__).warning("Langfuse flush failed")
