"""Futures-style job handle for non-blocking Nexus execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

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


class NexusJob:
    """Handle to a submitted Nexus execution (one or more execute jobs).

    Modeled on :class:`concurrent.futures.Future`: submission returns
    immediately and ``result()`` waits, downloads, and maps on demand.
    """

    def __init__(
        self, *, backend: Any, qnx: Any, parts: Sequence[JobPart], single: bool
    ) -> None:
        if not parts:
            raise ValueError("NexusJob requires at least one job part.")
        self._backend = backend
        self._qnx = qnx
        self._parts: tuple[JobPart, ...] = tuple(parts)
        self._single = bool(single)
        self._results: list[Any] | None = None

    def __repr__(self) -> str:
        circuits = sum(len(part.entries) for part in self._parts)
        return (
            f"NexusJob(job_ids={self.job_ids!r}, circuits={circuits}, "
            f"resolved={self._results is not None})"
        )

    def _only_part(self, plural_name: str) -> JobPart:
        if len(self._parts) != 1:
            raise ValueError(
                f"This handle wraps {len(self._parts)} Nexus jobs; "
                f"use {plural_name} instead."
            )
        return self._parts[0]

    @property
    def job_ids(self) -> tuple[str, ...]:
        return tuple(_job_id(part.ref) for part in self._parts)

    @property
    def job_id(self) -> str:
        return _job_id(self._only_part("job_ids").ref)

    @property
    def job_refs(self) -> tuple[Any, ...]:
        return tuple(part.ref for part in self._parts)

    @property
    def job_ref(self) -> Any:
        return self._only_part("job_refs").ref

    def statuses(self) -> list[Any]:
        return [self._qnx.jobs.status(part.ref) for part in self._parts]

    def status(self) -> Any:
        return self._qnx.jobs.status(self._only_part("statuses").ref)

    def done(self) -> bool:
        """Whether every underlying job stopped (successfully or not)."""
        return all(
            _status_name(status) in TERMINAL_STATUSES
            for status in self.statuses()
        )

    def cancel(self) -> None:
        errors: list[str] = []
        for part in self._parts:
            try:
                self._qnx.jobs.cancel(part.ref)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"job_id={_job_id(part.ref)} reason={exc}")
        if errors:
            raise NexusBackendError(
                "Failed to cancel Nexus job(s): " + "; ".join(errors)
            )
