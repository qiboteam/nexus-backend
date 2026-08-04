"""Futures-style job handle for non-blocking Nexus execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import NexusBackendError
from .translation import TranslationMetadata

#: Statuses after which a Nexus job will never change state again.
TERMINAL_STATUSES = frozenset(
    {"COMPLETED", "ERROR", "CANCELLED", "TERMINATED", "DEPLETED"}
)


def _job_id(ref: Any) -> str:
    value = getattr(ref, "id", None)
    return "unknown" if value is None else str(value)


def _status_name(status: Any) -> str:
    inner = getattr(status, "status", status)
    return str(getattr(inner, "value", inner))


def _best_effort_status(qnx: Any, job_ref: Any) -> Any:
    try:
        return qnx.jobs.status(job_ref)
    except Exception:  # noqa: BLE001 - status retrieval best effort
        return "unknown"


@dataclass(frozen=True)
class JobEntry:
    """Per-circuit context needed to map one Nexus result item to Qibo."""

    circuit: Any
    metadata: TranslationMetadata
    nshots: int


@dataclass(frozen=True)
class JobPart:
    """One underlying Nexus execute job and the circuit entries it covers."""

    ref: Any
    entries: tuple[JobEntry, ...]


def fetch_execution_items(
    qnx: Any, job_ref: Any, *, allow_incomplete: bool, expected: int | None
) -> list[Any]:
    """Fetch result refs for a finished execute job; validate cardinality.

    ``expected=None`` skips the cardinality check (used by the legacy
    ``_execute_programs`` pipeline whose callers validate separately).
    """
    try:
        items = list(qnx.jobs.results(job_ref, allow_incomplete=allow_incomplete))
    except Exception as exc:  # noqa: BLE001
        status = _best_effort_status(qnx, job_ref)
        raise NexusBackendError(
            f"Failed to fetch execute results. job_id={_job_id(job_ref)} "
            f"status={status} reason={exc}"
        ) from exc

    if not items:
        raise NexusBackendError(
            f"Execute job returned no result items. job_id={_job_id(job_ref)}"
        )
    if expected is not None and len(items) != expected:
        raise NexusBackendError(
            f"Result cardinality mismatch: expected {expected}, got "
            f"{len(items)} items. job_id={_job_id(job_ref)}"
        )
    return items
