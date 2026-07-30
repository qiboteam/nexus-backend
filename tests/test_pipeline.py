"""H-series compile → cost-estimate → execute pipeline tests.

Unlike the contract tests, these mock ONLY the qnx boundary
(``start_compile_job`` / ``start_execute_job`` / ``jobs.*``) so the pipeline
orchestration in ``nexus.backend`` actually executes: job submission, result
extraction, syntax-checker routing for HQC estimation, and every
money-relevant validation guard.
"""

from __future__ import annotations

import sys
import types

import pytest
from qibo import gates
from qibo.models import Circuit

import nexus.backend as backend_mod
from nexus.backend import (
    EstimateItem,
    _ensure_nexus_dependencies,
    _estimate_helios_cost,
    _estimate_helios_costs_batch,
    _estimate_prepared_compilation,
    _execute_programs,
    _expand_n_shots,
    _extract_compiled_program_refs,
    _normalize_nshots,
    _PreparedCompilation,
    _wait_for_job,
    run_compile_execute,
)
from nexus.errors import NexusBackendError
from nexus.translation import TranslationMetadata


class _Job:
    def __init__(self, job_id: str) -> None:
        self.id = job_id


class _CompiledItem:
    def __init__(self, ref: str) -> None:
        self._ref = ref

    def get_output(self) -> str:
        return self._ref


class _QuantinuumConfig:
    def __init__(self, *, device_name: str) -> None:
        self.device_name = device_name


def _install_quantinuum_models(monkeypatch: pytest.MonkeyPatch) -> None:
    models = types.ModuleType("qnexus.models")
    models.QuantinuumConfig = _QuantinuumConfig
    monkeypatch.setitem(sys.modules, "qnexus.models", models)


def _make_qnx(
    calls: dict,
    *,
    compile_outputs: list | None = None,
    cost_items: list | None = None,
    execute_items: list | None = None,
) -> types.SimpleNamespace:
    compile_job = _Job("compile-job-1")
    action_job = _Job("action-job-1")

    def start_compile_job(**kwargs):
        calls["compile_submit"] = kwargs
        return compile_job

    def start_execute_job(**kwargs):
        calls["execute_submit"] = kwargs
        return action_job

    def wait_for(job, timeout):
        calls.setdefault("waited", []).append(job.id)
        return job

    def results(job, allow_incomplete=False):
        if job is compile_job:
            return compile_outputs
        calls["results_allow_incomplete"] = allow_incomplete
        return execute_items

    def cost_confidence(job):
        calls["cost_job"] = job.id
        return cost_items

    return types.SimpleNamespace(
        start_compile_job=start_compile_job,
        start_execute_job=start_execute_job,
        jobs=types.SimpleNamespace(
            wait_for=wait_for,
            results=results,
            cost_confidence=cost_confidence,
            status=lambda job: "ERRORED",
        ),
    )


def _make_h2_backend(
    monkeypatch: pytest.MonkeyPatch, **kwargs
) -> backend_mod.NexusClientBackend:
    monkeypatch.setattr(backend_mod, "_ensure_nexus_dependencies", lambda: None)
    monkeypatch.setattr(backend_mod, "authenticate", lambda **k: None)
    monkeypatch.setattr(
        backend_mod, "ensure_project", lambda project_name: "project-ref"
    )
    monkeypatch.setattr(
        backend_mod,
        "build_nexus_backend_config",
        lambda cfg: types.SimpleNamespace(device_name="H2-1LE"),
    )
    monkeypatch.setattr(
        backend_mod.NexusClientBackend,
        "_upload_translated_program",
        lambda self, circuit, *, parameters=None, sequence_idx=0: (
            f"uploaded-{sequence_idx}",
            TranslationMetadata(measured_qubits=[0], nqubits=1, qasm="q"),
        ),
    )
    return backend_mod.NexusClientBackend(
        platform="hseries:H2-1LE", project="proj", **kwargs
    )


def make_measured_circuit(nqubits: int = 1) -> Circuit:
    circuit = Circuit(nqubits)
    circuit.add(gates.M(*range(nqubits)))
    return circuit


def test_estimate_circuit_routes_cost_through_syntax_checker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cost probe must target the H2-1SC syntax checker (free), never the
    LE emulator target itself, and must price the compiled program."""
    _install_quantinuum_models(monkeypatch)
    calls: dict = {}
    qnx = _make_qnx(
        calls,
        compile_outputs=[_CompiledItem("compiled-0")],
        cost_items=[(1.25, 90.0)],
    )
    backend = _make_h2_backend(monkeypatch, language="LANG-X")
    monkeypatch.setattr(backend_mod, "_import_qnexus", lambda: qnx)

    estimate = backend.estimate_circuit(make_measured_circuit(1), nshots=100)

    assert calls["compile_submit"]["programs"] == ["uploaded-0"]
    assert calls["compile_submit"]["optimisation_level"] == 2
    assert calls["execute_submit"]["backend_config"].device_name == "H2-1SC"
    assert calls["execute_submit"]["programs"] == ["compiled-0"]
    assert calls["execute_submit"]["n_shots"] == 100
    assert estimate.total_hqcs == 1.25
    assert estimate.items == [EstimateItem(0, 100, 1.25, "compile-job-1")]


def test_run_compile_execute_flows_compiled_programs_to_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution must submit the COMPILED program refs (not the uploaded
    sources) and forward shots, language, and the cost cap."""
    calls: dict = {}
    qnx = _make_qnx(
        calls,
        compile_outputs=[_CompiledItem("compiled-0"), _CompiledItem("compiled-1")],
        execute_items=["res-0", "res-1"],
    )
    monkeypatch.setattr(backend_mod, "_import_qnexus", lambda: qnx)

    items = run_compile_execute(
        programs=["uploaded-0", "uploaded-1"],
        backend_config="cfg",
        optimisation_level=1,
        n_shots=[10, 20],
        timeout=5.0,
        allow_incomplete=True,
        language="QIR",
        platform="hseries:H2-1LE",
        job_name_prefix="team",
        project="project-ref",
        max_cost=12.5,
    )

    assert items == ["res-0", "res-1"]
    assert calls["compile_submit"]["programs"] == ["uploaded-0", "uploaded-1"]
    assert calls["execute_submit"]["programs"] == ["compiled-0", "compiled-1"]
    assert calls["execute_submit"]["n_shots"] == [10, 20]
    assert calls["execute_submit"]["language"] == "QIR"
    assert calls["execute_submit"]["max_cost"] == 12.5
    assert calls["results_allow_incomplete"] is True
    assert str(calls["compile_submit"]["name"]).startswith("team-compile-")


def _prepare(qnx: types.SimpleNamespace) -> _PreparedCompilation:
    return backend_mod._prepare_compiled_programs(
        qnx=qnx,
        programs=["p"],
        backend_config="cfg",
        optimisation_level=2,
        n_shots=10,
        timeout=1.0,
        platform="hseries:H2-1LE",
        batch_mode=False,
    )


def test_prepare_wraps_compile_submit_failure() -> None:
    def boom(**kwargs):
        raise RuntimeError("quota exhausted")

    qnx = types.SimpleNamespace(start_compile_job=boom)
    with pytest.raises(NexusBackendError, match="Failed to submit compile job"):
        _prepare(qnx)


def test_prepare_wraps_compile_result_failures() -> None:
    def raising_results(job):
        raise RuntimeError("gone")

    qnx = _make_qnx({}, compile_outputs=[])
    qnx.jobs.results = raising_results
    with pytest.raises(NexusBackendError, match="Failed to retrieve compile output"):
        _prepare(qnx)

    qnx = _make_qnx({}, compile_outputs=[])
    with pytest.raises(NexusBackendError, match="no results"):
        _prepare(qnx)


def _estimate_prepared(qnx, backend_config) -> object:
    prepared = _PreparedCompilation(
        compiled_programs=["compiled-0"],
        submission_n_shots=10,
        shot_values=[10],
        compile_job_id="compile-job-1",
        batch_mode=False,
    )
    return _estimate_prepared_compilation(
        qnx=qnx,
        prepared=prepared,
        backend_config=backend_config,
        project=None,
        platform="hseries:H2-1LE",
        optimisation_level=2,
        timeout=1.0,
    )


@pytest.mark.parametrize(
    "backend_config",
    [
        types.SimpleNamespace(),  # no device_name at all
        types.SimpleNamespace(device_name="aer_simulator"),  # not an H2 device
    ],
)
def test_estimate_rejects_non_h2_targets(backend_config) -> None:
    with pytest.raises(NexusBackendError, match="Quantinuum H2"):
        _estimate_prepared(_make_qnx({}), backend_config)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda qnx: setattr(
                qnx,
                "start_execute_job",
                lambda **kwargs: (_ for _ in ()).throw(RuntimeError("submit")),
            ),
            "Failed to submit cost estimation job",
        ),
        (
            lambda qnx: setattr(
                qnx.jobs,
                "cost_confidence",
                lambda job: (_ for _ in ()).throw(RuntimeError("fetch")),
            ),
            "Failed to fetch batched cost estimation results",
        ),
    ],
)
def test_estimate_wraps_cost_job_failures(mutate, match) -> None:
    h2_config = types.SimpleNamespace(device_name="H2-1LE")
    qnx = _make_qnx({}, cost_items=[(1.0, 90.0)])
    mutate(qnx)
    with pytest.raises(NexusBackendError, match=match):
        _estimate_prepared(qnx, h2_config)


@pytest.mark.parametrize(
    ("cost_items", "match"),
    [
        ([(1.0, 90.0), (2.0, 90.0)], "unexpected number of items"),
        ([(None, 90.0)], "invalid per-item cost data"),
        ([("not-a-number", 90.0)], "invalid per-item cost data"),
    ],
)
def test_estimate_rejects_malformed_cost_results(cost_items, match) -> None:
    h2_config = types.SimpleNamespace(device_name="H2-1LE")
    qnx = _make_qnx({}, cost_items=cost_items)
    with pytest.raises(NexusBackendError, match=match):
        _estimate_prepared(qnx, h2_config)


def _execute(qnx, **overrides):
    kwargs = dict(
        qnx=qnx,
        programs=["p"],
        n_shots=10,
        backend_config="cfg",
        timeout=1.0,
        allow_incomplete=False,
        language=None,
        platform="hseries:H2-1LE",
    )
    kwargs.update(overrides)
    return _execute_programs(**kwargs)


def test_execute_programs_wraps_submit_failure() -> None:
    def boom(**kwargs):
        raise RuntimeError("rejected")

    qnx = types.SimpleNamespace(start_execute_job=boom)
    with pytest.raises(NexusBackendError, match="Failed to submit execute job"):
        _execute(qnx)


def test_execute_programs_reports_status_when_results_fail() -> None:
    qnx = _make_qnx({})

    def raising_results(job, allow_incomplete=False):
        raise RuntimeError("expired")

    qnx.jobs.results = raising_results
    with pytest.raises(NexusBackendError, match="status=ERRORED"):
        _execute(qnx)

    qnx.jobs.status = lambda job: (_ for _ in ()).throw(RuntimeError("also down"))
    with pytest.raises(NexusBackendError, match="status=unknown"):
        _execute(qnx)


def test_execute_programs_rejects_empty_results() -> None:
    qnx = _make_qnx({}, execute_items=[])
    with pytest.raises(NexusBackendError, match="no result items"):
        _execute(qnx)


def test_wait_for_job_reports_job_id_and_status() -> None:
    def timed_out(job, timeout):
        raise TimeoutError("too slow")

    qnx = types.SimpleNamespace(
        jobs=types.SimpleNamespace(wait_for=timed_out, status=lambda job: "DEPLETED")
    )
    with pytest.raises(
        NexusBackendError, match=r"execute job.*job_id=j-7.*status=DEPLETED"
    ):
        _wait_for_job(qnx, _Job("j-7"), timeout=1.0, stage="execute")

    qnx.jobs.status = lambda job: (_ for _ in ()).throw(RuntimeError("no status"))
    with pytest.raises(NexusBackendError, match="status=unknown"):
        _wait_for_job(qnx, _Job("j-7"), timeout=1.0, stage="execute")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda hugr: setattr(
                hugr,
                "cost_confidence",
                lambda **kwargs: (_ for _ in ()).throw(RuntimeError("down")),
            ),
            "Failed to estimate Helios execution cost",
        ),
        (lambda hugr: setattr(hugr, "items", []), "Invalid Helios cost estimate"),
        (
            lambda hugr: setattr(hugr, "items", [(None, 90.0)]),
            "Invalid Helios cost estimate",
        ),
        (
            lambda hugr: setattr(hugr, "items", [("nan-string",)]),
            "Invalid Helios cost estimate",
        ),
    ],
)
def test_estimate_helios_cost_rejects_malformed_results(mutate, match) -> None:
    hugr = types.SimpleNamespace(items=[(1.0, 90.0)])
    hugr.cost_confidence = lambda **kwargs: hugr.items
    mutate(hugr)
    qnx = types.SimpleNamespace(hugr=hugr)
    with pytest.raises(NexusBackendError, match=match):
        _estimate_helios_cost(qnx=qnx, program="p", nshots=10)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda hugr: setattr(
                hugr,
                "cost_confidence",
                lambda **kwargs: (_ for _ in ()).throw(RuntimeError("down")),
            ),
            "Failed to estimate Helios execution costs",
        ),
        (lambda hugr: setattr(hugr, "items", None), "Invalid Helios cost estimate"),
        (
            lambda hugr: setattr(hugr, "items", [(1.0, 90.0)]),
            "returned 1 values for 2 programs",
        ),
        (
            lambda hugr: setattr(hugr, "items", [(1.0, 90.0), (None, 90.0)]),
            "Invalid Helios cost estimate at index 1",
        ),
        (
            lambda hugr: setattr(hugr, "items", [(1.0, 90.0), ("nan-string", 90.0)]),
            "Invalid Helios cost estimate at index 1",
        ),
    ],
)
def test_estimate_helios_batch_costs_rejects_malformed_results(mutate, match) -> None:
    hugr = types.SimpleNamespace(items=[(1.0, 90.0), (2.0, 90.0)])
    hugr.cost_confidence = lambda **kwargs: hugr.items
    mutate(hugr)
    qnx = types.SimpleNamespace(hugr=hugr)
    with pytest.raises(NexusBackendError, match=match):
        _estimate_helios_costs_batch(
            qnx=qnx, programs=["p0", "p1"], n_shots=[10, 20]
        )


def test_normalize_nshots_defaults_none_to_1000() -> None:
    assert _normalize_nshots(None) == 1000


def test_expand_n_shots_rejects_cardinality_mismatch() -> None:
    with pytest.raises(ValueError, match="nshots cardinality mismatch"):
        _expand_n_shots([1, 2], 3)


def test_extract_compiled_program_refs_unwraps_outputs() -> None:
    refs = _extract_compiled_program_refs([_CompiledItem("a"), _CompiledItem("b")])
    assert refs == ["a", "b"]

    with pytest.raises(NexusBackendError, match="no results"):
        _extract_compiled_program_refs([])


def test_ensure_nexus_dependencies_with_fake_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "qnexus", types.ModuleType("qnexus"))
    monkeypatch.setitem(sys.modules, "pytket.qasm", types.ModuleType("pytket.qasm"))
    _ensure_nexus_dependencies()
