from __future__ import annotations

import types

import pytest

from nexus.errors import NexusBackendError
from nexus.job import (
    JobEntry,
    JobPart,
    NexusJob,
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


class _FakeBackend:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(allow_incomplete=False)
        self.map_calls: list[dict[str, object]] = []

    def _map_execution_result(
        self, *, execution_result_ref, circuit, nshots, metadata
    ):
        self.map_calls.append(
            {
                "ref": execution_result_ref,
                "circuit": circuit,
                "nshots": nshots,
                "metadata": metadata,
            }
        )
        return f"mapped-{execution_result_ref}-{nshots}"


def _single_job(qnx, *, nshots: int = 10, backend: _FakeBackend | None = None):
    backend = backend or _FakeBackend()
    part = JobPart(ref=_Ref("job-1"), entries=(_entry(nshots),))
    return NexusJob(backend=backend, qnx=qnx, parts=[part], single=True), backend


def test_nexus_job_requires_parts() -> None:
    with pytest.raises(ValueError, match="at least one job part"):
        NexusJob(backend=_FakeBackend(), qnx=_jobs_ns(), parts=[], single=True)


def test_single_part_accessors() -> None:
    job, _ = _single_job(_jobs_ns())
    assert job.job_ids == ("job-1",)
    assert job.job_id == "job-1"
    assert job.job_refs[0].id == "job-1"
    assert job.job_ref.id == "job-1"
    assert "job-1" in repr(job)


def test_multi_part_single_accessors_raise() -> None:
    parts = [
        JobPart(ref=_Ref("job-1"), entries=(_entry(),)),
        JobPart(ref=_Ref("job-2"), entries=(_entry(),)),
    ]
    job = NexusJob(backend=_FakeBackend(), qnx=_jobs_ns(), parts=parts, single=False)
    assert job.job_ids == ("job-1", "job-2")
    with pytest.raises(ValueError, match="job_ids"):
        _ = job.job_id
    with pytest.raises(ValueError, match="job_refs"):
        _ = job.job_ref
    with pytest.raises(ValueError, match="statuses"):
        job.status()


def test_statuses_and_done() -> None:
    running = types.SimpleNamespace(
        status=types.SimpleNamespace(value="RUNNING")
    )
    completed = types.SimpleNamespace(
        status=types.SimpleNamespace(value="COMPLETED")
    )
    statuses = {"job-1": completed, "job-2": running}
    qnx = _jobs_ns(status=lambda job: statuses[job.id])
    parts = [
        JobPart(ref=_Ref("job-1"), entries=(_entry(),)),
        JobPart(ref=_Ref("job-2"), entries=(_entry(),)),
    ]
    job = NexusJob(backend=_FakeBackend(), qnx=qnx, parts=parts, single=False)
    assert job.statuses() == [completed, running]
    assert job.done() is False

    statuses["job-2"] = types.SimpleNamespace(
        status=types.SimpleNamespace(value="ERROR")
    )
    assert job.done() is True  # ERROR is terminal; done() means "not running"

    single, _ = _single_job(qnx=_jobs_ns(status=lambda job: completed))
    assert single.status() is completed
    assert single.done() is True


def test_cancel_cancels_every_part_and_aggregates_failures() -> None:
    cancelled: list[str] = []

    def cancel(job):
        if job.id == "job-1":
            raise RuntimeError("already terminal")
        cancelled.append(job.id)

    qnx = _jobs_ns(cancel=cancel)
    parts = [
        JobPart(ref=_Ref("job-1"), entries=(_entry(),)),
        JobPart(ref=_Ref("job-2"), entries=(_entry(),)),
    ]
    job = NexusJob(backend=_FakeBackend(), qnx=qnx, parts=parts, single=False)
    with pytest.raises(NexusBackendError, match="job-1"):
        job.cancel()
    assert cancelled == ["job-2"]  # failure on part 1 did not stop part 2

    ok_job, _ = _single_job(_jobs_ns(cancel=lambda job: cancelled.append(job.id)))
    ok_job.cancel()
    assert cancelled[-1] == "job-1"


def test_result_single_shape_maps_and_caches() -> None:
    calls = {"results": 0}

    def results(job, allow_incomplete=False):
        calls["results"] += 1
        return ["item-0"]

    qnx = _jobs_ns(results=results)
    job, backend = _single_job(qnx, nshots=7)
    assert job.result() == "mapped-item-0-7"
    assert job.result() == "mapped-item-0-7"
    assert calls["results"] == 1  # cached after first success
    assert backend.map_calls[0]["nshots"] == 7
    assert "resolved=True" in repr(job)


def test_result_batch_shape_preserves_order_across_parts() -> None:
    def results(job, allow_incomplete=False):
        return {"job-1": ["a-0", "a-1"], "job-2": ["b-0"]}[job.id]

    qnx = _jobs_ns(results=results)
    parts = [
        JobPart(ref=_Ref("job-1"), entries=(_entry(1), _entry(2))),
        JobPart(ref=_Ref("job-2"), entries=(_entry(3),)),
    ]
    backend = _FakeBackend()
    job = NexusJob(backend=backend, qnx=qnx, parts=parts, single=False)
    assert job.result() == ["mapped-a-0-1", "mapped-a-1-2", "mapped-b-0-3"]


def test_result_forwards_timeout_and_allow_incomplete() -> None:
    captured: dict[str, object] = {}

    def wait_for(job, timeout=None):
        captured["timeout"] = timeout
        return job

    def results(job, allow_incomplete=False):
        captured["allow_incomplete"] = allow_incomplete
        return ["item-0"]

    backend = _FakeBackend()
    backend.config.allow_incomplete = True
    job, _ = _single_job(
        _jobs_ns(wait_for=wait_for, results=results), backend=backend
    )
    job.result(timeout=30.0)
    assert captured["allow_incomplete"] is True
    timeout = captured["timeout"]
    assert isinstance(timeout, float) and 0 < timeout <= 30.0

    job2, _ = _single_job(_jobs_ns(wait_for=wait_for, results=results))
    job2.result()  # no timeout -> wait forever
    assert captured["timeout"] is None


def test_result_timeout_is_reusable() -> None:
    attempts = {"n": 0}

    def wait_for(job, timeout=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("still queued")
        return job

    job, _ = _single_job(_jobs_ns(wait_for=wait_for))
    with pytest.raises(TimeoutError, match="job-1"):
        job.result(timeout=1.0)
    # Handle is still usable; a later call succeeds.
    assert job.result() == "mapped-item-0-10"


def test_result_deadline_spans_parts() -> None:
    import nexus.job as job_mod

    clock = {"now": 100.0}
    fake_time = types.SimpleNamespace(monotonic=lambda: clock["now"])

    def slow_wait(job, timeout=None):
        clock["now"] += 10.0  # first wait consumes the whole budget
        return job

    parts = [
        JobPart(ref=_Ref("job-1"), entries=(_entry(),)),
        JobPart(ref=_Ref("job-2"), entries=(_entry(),)),
    ]
    job = NexusJob(
        backend=_FakeBackend(),
        qnx=_jobs_ns(wait_for=slow_wait),
        parts=parts,
        single=False,
    )
    real_time = job_mod.time
    job_mod.time = fake_time
    try:
        with pytest.raises(TimeoutError, match="job-2"):
            job.result(timeout=5.0)
    finally:
        job_mod.time = real_time


def test_result_wraps_job_failure() -> None:
    def wait_for(job, timeout=None):
        raise RuntimeError("Job errored with detail: quota exceeded")

    qnx = _jobs_ns(wait_for=wait_for, status=lambda job: "ERROR")
    job, _ = _single_job(qnx)
    with pytest.raises(NexusBackendError, match="status=ERROR"):
        job.result()
