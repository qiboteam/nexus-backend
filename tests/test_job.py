from __future__ import annotations

import types

import pytest

from nexus.errors import NexusBackendError
from nexus.job import (
    JobEntry,
    JobPart,
    TERMINAL_STATUSES,
    _job_id,
    _status_name,
    fetch_execution_items,
)
from nexus.translation import TranslationMetadata


class _Ref:
    def __init__(self, ref_id: str = "job-1") -> None:
        self.id = ref_id


def _metadata() -> TranslationMetadata:
    return TranslationMetadata(measured_qubits=[0], nqubits=1, qasm="q")


def _entry(nshots: int = 10, circuit: object = "circ") -> JobEntry:
    return JobEntry(circuit=circuit, metadata=_metadata(), nshots=nshots)


def _jobs_ns(**overrides) -> types.SimpleNamespace:
    ns = types.SimpleNamespace(
        wait_for=lambda job, timeout=None: job,
        results=lambda job, allow_incomplete=False: ["item-0"],
        status=lambda job: "COMPLETED",
        cancel=lambda job: None,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return types.SimpleNamespace(jobs=ns)


def test_job_id_prefers_id_attribute() -> None:
    assert _job_id(_Ref("abc")) == "abc"
    assert _job_id(object()) == "unknown"


def test_status_name_unwraps_jobstatus_and_enum() -> None:
    enum_like = types.SimpleNamespace(value="RUNNING")
    job_status = types.SimpleNamespace(status=enum_like)
    assert _status_name(job_status) == "RUNNING"
    assert _status_name("COMPLETED") == "COMPLETED"
    assert TERMINAL_STATUSES == {
        "COMPLETED",
        "ERROR",
        "CANCELLED",
        "TERMINATED",
        "DEPLETED",
    }


def test_fetch_execution_items_returns_items_and_forwards_allow_incomplete() -> None:
    captured: dict[str, object] = {}

    def results(job, allow_incomplete=False):
        captured["allow_incomplete"] = allow_incomplete
        return ["item-0", "item-1"]

    qnx = _jobs_ns(results=results)
    items = fetch_execution_items(
        qnx, _Ref(), allow_incomplete=True, expected=2
    )
    assert items == ["item-0", "item-1"]
    assert captured["allow_incomplete"] is True


def test_fetch_execution_items_rejects_empty() -> None:
    qnx = _jobs_ns(results=lambda job, allow_incomplete=False: [])
    with pytest.raises(NexusBackendError, match="no result items"):
        fetch_execution_items(qnx, _Ref(), allow_incomplete=False, expected=None)


def test_fetch_execution_items_enforces_expected_cardinality() -> None:
    qnx = _jobs_ns(results=lambda job, allow_incomplete=False: ["only-one"])
    with pytest.raises(NexusBackendError, match="Result cardinality mismatch"):
        fetch_execution_items(qnx, _Ref(), allow_incomplete=False, expected=2)
    # expected=None skips the cardinality check entirely.
    assert fetch_execution_items(
        qnx, _Ref(), allow_incomplete=False, expected=None
    ) == ["only-one"]


def test_fetch_execution_items_wraps_fetch_failure_with_status() -> None:
    def raising_results(job, allow_incomplete=False):
        raise RuntimeError("boom")

    qnx = _jobs_ns(results=raising_results, status=lambda job: "DEPLETED")
    with pytest.raises(NexusBackendError, match="status=DEPLETED"):
        fetch_execution_items(qnx, _Ref("j-7"), allow_incomplete=False, expected=1)

    qnx.jobs.status = lambda job: (_ for _ in ()).throw(RuntimeError("down"))
    with pytest.raises(NexusBackendError, match="status=unknown"):
        fetch_execution_items(qnx, _Ref("j-7"), allow_incomplete=False, expected=1)


def test_job_part_holds_entries() -> None:
    part = JobPart(ref=_Ref(), entries=(_entry(5),))
    assert part.entries[0].nshots == 5
    assert part.entries[0].metadata.measured_qubits == [0]
