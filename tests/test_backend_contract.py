from __future__ import annotations

import types

import pytest
from qibo import gates
from qibo.models import Circuit

import nexus.backend as backend_mod
from nexus.errors import NexusBackendError, UnsupportedExecutionError
from nexus.job import NexusJob
from nexus.translation import TranslationMetadata


def make_measured_circuit(
    nqubits: int = 1, measured_qubits: tuple[int, ...] | None = None
) -> Circuit:
    circuit = Circuit(nqubits)
    targets = measured_qubits if measured_qubits is not None else tuple(range(nqubits))
    circuit.add(gates.M(*targets))
    return circuit


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> backend_mod.NexusClientBackend:
    monkeypatch.setattr(backend_mod, "_ensure_nexus_dependencies", lambda: None)
    monkeypatch.setattr(backend_mod, "authenticate", lambda **kwargs: None)
    monkeypatch.setattr(
        backend_mod, "ensure_project", lambda project_name: "project-ref"
    )
    monkeypatch.setattr(
        backend_mod, "build_nexus_backend_config", lambda cfg: "backend-config"
    )
    monkeypatch.setattr(backend_mod, "_import_qnexus", lambda: types.SimpleNamespace())
    return backend_mod.NexusClientBackend(
        platform="hseries:H2-1LE",
        project="proj",
        job_name_prefix="team-alpha",
    )


def test_execute_circuit_contract_shape(
    backend: backend_mod.NexusClientBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        backend_mod,
        "_import_qnexus",
        lambda: _make_hseries_qnx(calls, execute_items=["execution-item"]),
    )

    upload_calls: dict[str, object] = {}

    def fake_upload(self, circuit, *, parameters=None, sequence_idx=0):
        upload_calls.update({"parameters": parameters, "sequence_idx": sequence_idx})
        return "program-ref", TranslationMetadata(
            measured_qubits=[0, 1],
            nqubits=2,
            qasm="q",
            measurement_registers=["register0", "register1"],
        )

    map_calls: dict[str, object] = {}

    def fake_map(**kwargs):
        map_calls.update(kwargs)
        return {"kind": "MeasurementOutcomes", "nshots": kwargs["nshots"]}

    monkeypatch.setattr(
        backend_mod.NexusClientBackend, "_upload_translated_program", fake_upload
    )
    monkeypatch.setattr(backend_mod, "map_nexus_result_to_qibo", fake_map)

    result = backend.execute_circuit(
        make_measured_circuit(1), nshots=123, parameters=[0.5]
    )

    assert result["kind"] == "MeasurementOutcomes"
    assert result["nshots"] == 123
    assert upload_calls == {"parameters": [0.5], "sequence_idx": 0}
    execute_kwargs = calls["execute"][-1]
    assert execute_kwargs["n_shots"] == 123
    assert str(execute_kwargs["name"]).startswith("team-alpha-execute-")
    assert calls["compile"][-1]["programs"] == ["program-ref"]
    assert map_calls["execution_result_ref"] == "execution-item"
    assert map_calls["measured_qubits"] == [0, 1]
    assert map_calls["register_order"] == ["register0", "register1"]


def test_upload_translated_program_uses_job_name_prefix(
    backend: backend_mod.NexusClientBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        backend_mod,
        "translate_qibo_to_pytket",
        lambda circuit, parameters=None: (
            "pytket-circuit",
            TranslationMetadata(measured_qubits=[0], nqubits=1, qasm="OPENQASM 2.0;"),
        ),
    )

    qnx = types.SimpleNamespace(
        circuits=types.SimpleNamespace(
            upload=lambda *, circuit, name, project: captured.update(
                {"circuit": circuit, "name": name, "project": project}
            )
            or "program-ref"
        )
    )
    monkeypatch.setattr(backend_mod, "_import_qnexus", lambda: qnx)

    program_ref, metadata = backend._upload_translated_program(
        make_measured_circuit(1),
        sequence_idx=7,
    )

    assert program_ref == "program-ref"
    assert metadata.measured_qubits == [0]
    assert captured["circuit"] == "pytket-circuit"
    assert captured["project"] == "project-ref"
    assert str(captured["name"]).startswith("team-alpha-program-7-")


def test_execute_circuits_cardinality_and_order(
    backend: backend_mod.NexusClientBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        backend_mod, "_import_qnexus", lambda: _make_hseries_qnx(calls)
    )

    upload_calls: list[dict[str, object]] = []

    def fake_upload(self, circuit, *, parameters=None, sequence_idx=0):
        upload_calls.append({"parameters": parameters, "sequence_idx": sequence_idx})
        return f"program-ref-{sequence_idx}", TranslationMetadata(
            measured_qubits=[0], nqubits=1, qasm="q"
        )

    map_calls: list[dict[str, object]] = []

    def fake_map(**kwargs):
        map_calls.append(kwargs)
        return f"mapped-{kwargs['execution_result_ref']}"

    monkeypatch.setattr(
        backend_mod.NexusClientBackend, "_upload_translated_program", fake_upload
    )
    monkeypatch.setattr(backend_mod, "map_nexus_result_to_qibo", fake_map)

    circuits = [make_measured_circuit(1), make_measured_circuit(1)]
    result = backend.execute_circuits(
        circuits, nshots=[10, 20], parameters_list=[["a"], ["b"]]
    )

    assert result == ["mapped-execution-item-0", "mapped-execution-item-1"]
    assert calls["compile"][-1]["programs"] == ["program-ref-0", "program-ref-1"]
    assert calls["execute"][-1]["n_shots"] == [10, 20]
    assert upload_calls == [
        {"parameters": ["a"], "sequence_idx": 0},
        {"parameters": ["b"], "sequence_idx": 1},
    ]
    assert [call["nshots"] for call in map_calls] == [10, 20]


def test_execute_circuits_nshots_cardinality_mismatch(
    backend: backend_mod.NexusClientBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_upload(self, circuit, *, parameters=None, sequence_idx=0):
        return f"program-ref-{sequence_idx}", TranslationMetadata(
            measured_qubits=[0], nqubits=1, qasm="q"
        )

    monkeypatch.setattr(
        backend_mod.NexusClientBackend, "_upload_translated_program", fake_upload
    )

    circuits = [make_measured_circuit(1), make_measured_circuit(1)]
    with pytest.raises(ValueError, match="nshots cardinality mismatch"):
        backend.execute_circuits(circuits, nshots=[10], parameters_list=[None, None])

    with pytest.raises(ValueError, match="nshots cardinality mismatch"):
        backend.estimate_circuits(circuits, nshots=[10], parameters_list=[None, None])


def test_estimate_circuit_contract_shape(
    backend: backend_mod.NexusClientBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    def fake_upload(self, circuit, *, parameters=None, sequence_idx=0):
        calls["upload"] = {"parameters": parameters, "sequence_idx": sequence_idx}
        return "program-ref", TranslationMetadata(
            measured_qubits=[0, 1], nqubits=2, qasm="q"
        )

    def fake_prepare(**kwargs):
        calls["prepare"] = kwargs
        return backend_mod._PreparedCompilation(
            compiled_programs=["compiled-program"],
            submission_n_shots=123,
            shot_values=[123],
            compile_job_id="compile-123",
            batch_mode=False,
        )

    def fake_estimate(**kwargs):
        calls["estimate"] = kwargs
        return backend_mod.ExecutionEstimate(
            platform="hseries:H2-1LE",
            optimisation_level=2,
            batch_mode=False,
            total_hqcs=1.75,
            items=[
                backend_mod.EstimateItem(
                    sequence_idx=0,
                    nshots=123,
                    hqcs=1.75,
                    compile_job_id="compile-123",
                )
            ],
        )

    monkeypatch.setattr(
        backend_mod.NexusClientBackend, "_upload_translated_program", fake_upload
    )
    monkeypatch.setattr(backend_mod, "_prepare_compiled_programs", fake_prepare)
    monkeypatch.setattr(backend_mod, "_estimate_prepared_compilation", fake_estimate)
    monkeypatch.setattr(
        backend_mod,
        "_import_qnexus",
        lambda: types.SimpleNamespace(
            circuits=types.SimpleNamespace(cost=lambda *a, **k: None)
        ),
    )

    circuit = make_measured_circuit(1)
    estimate = backend.estimate_circuit(circuit, nshots=123, parameters=[0.5])

    assert estimate.total_hqcs == 1.75
    assert estimate.items[0].nshots == 123
    assert calls["upload"] == {"parameters": [0.5], "sequence_idx": 0}
    assert calls["prepare"]["programs"] == ["program-ref"]
    assert calls["prepare"]["n_shots"] == 123
    assert calls["prepare"]["batch_mode"] is False
    assert calls["estimate"]["prepared"].compile_job_id == "compile-123"


def test_estimate_circuits_batch_contract_shape(
    backend: backend_mod.NexusClientBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_calls: list[dict[str, object]] = []

    def fake_upload(self, circuit, *, parameters=None, sequence_idx=0):
        upload_calls.append({"parameters": parameters, "sequence_idx": sequence_idx})
        return f"program-ref-{sequence_idx}", TranslationMetadata(
            measured_qubits=[0], nqubits=1, qasm="q"
        )

    def fake_prepare(**kwargs):
        assert kwargs["programs"] == ["program-ref-0", "program-ref-1"]
        assert kwargs["n_shots"] == [10, 20]
        assert kwargs["batch_mode"] is True
        return backend_mod._PreparedCompilation(
            compiled_programs=["compiled-0", "compiled-1"],
            submission_n_shots=[10, 20],
            shot_values=[10, 20],
            compile_job_id="compile-456",
            batch_mode=True,
        )

    def fake_estimate(**kwargs):
        prepared = kwargs["prepared"]
        assert prepared.compiled_programs == ["compiled-0", "compiled-1"]
        return backend_mod.ExecutionEstimate(
            platform="hseries:H2-1LE",
            optimisation_level=2,
            batch_mode=True,
            total_hqcs=5.0,
            items=[
                backend_mod.EstimateItem(0, 10, 2.0, "compile-456"),
                backend_mod.EstimateItem(1, 20, 3.0, "compile-456"),
            ],
        )

    monkeypatch.setattr(
        backend_mod.NexusClientBackend, "_upload_translated_program", fake_upload
    )
    monkeypatch.setattr(backend_mod, "_prepare_compiled_programs", fake_prepare)
    monkeypatch.setattr(backend_mod, "_estimate_prepared_compilation", fake_estimate)
    monkeypatch.setattr(
        backend_mod,
        "_import_qnexus",
        lambda: types.SimpleNamespace(
            circuits=types.SimpleNamespace(cost=lambda *a, **k: None)
        ),
    )

    circuits = [make_measured_circuit(1), make_measured_circuit(1)]
    estimate = backend.estimate_circuits(
        circuits, nshots=[10, 20], parameters_list=[["a"], ["b"]]
    )

    assert estimate.total_hqcs == 5.0
    assert [item.hqcs for item in estimate.items] == [2.0, 3.0]
    assert upload_calls == [
        {"parameters": ["a"], "sequence_idx": 0},
        {"parameters": ["b"], "sequence_idx": 1},
    ]


def test_estimate_circuits_non_batch_aggregates_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend_mod, "_ensure_nexus_dependencies", lambda: None)
    monkeypatch.setattr(backend_mod, "authenticate", lambda **kwargs: None)
    monkeypatch.setattr(
        backend_mod, "ensure_project", lambda project_name: "project-ref"
    )
    monkeypatch.setattr(
        backend_mod, "build_nexus_backend_config", lambda cfg: "backend-config"
    )

    backend = backend_mod.NexusClientBackend(
        platform="hseries:H2-1LE",
        project="proj",
        job_name_prefix="team-alpha",
        batch_mode=False,
    )

    def fake_upload(self, circuit, *, parameters=None, sequence_idx=0):
        return f"program-ref-{sequence_idx}", TranslationMetadata(
            measured_qubits=[0], nqubits=1, qasm="q"
        )

    def fake_prepare(**kwargs):
        sequence_idx = int(str(kwargs["programs"][0]).rsplit("-", 1)[-1])
        shots = kwargs["n_shots"]
        return backend_mod._PreparedCompilation(
            compiled_programs=[f"compiled-{sequence_idx}"],
            submission_n_shots=shots,
            shot_values=[shots],
            compile_job_id=f"compile-{sequence_idx}",
            batch_mode=False,
        )

    def fake_estimate(**kwargs):
        prepared = kwargs["prepared"]
        sequence_idx = int(str(prepared.compiled_programs[0]).rsplit("-", 1)[-1])
        shots = prepared.shot_values[0]
        return backend_mod.ExecutionEstimate(
            platform="hseries:H2-1LE",
            optimisation_level=2,
            batch_mode=False,
            total_hqcs=float(sequence_idx + 1),
            items=[
                backend_mod.EstimateItem(
                    sequence_idx=0,
                    nshots=shots,
                    hqcs=float(sequence_idx + 1),
                    compile_job_id=prepared.compile_job_id,
                )
            ],
        )

    monkeypatch.setattr(
        backend_mod.NexusClientBackend, "_upload_translated_program", fake_upload
    )
    monkeypatch.setattr(backend_mod, "_prepare_compiled_programs", fake_prepare)
    monkeypatch.setattr(backend_mod, "_estimate_prepared_compilation", fake_estimate)
    monkeypatch.setattr(
        backend_mod,
        "_import_qnexus",
        lambda: types.SimpleNamespace(
            circuits=types.SimpleNamespace(cost=lambda *a, **k: None)
        ),
    )

    circuits = [make_measured_circuit(1), make_measured_circuit(1)]
    estimate = backend.estimate_circuits(
        circuits, nshots=[10, 20], parameters_list=[["a"], ["b"]]
    )

    assert estimate.batch_mode is False
    assert estimate.total_hqcs == 3.0
    assert [(item.sequence_idx, item.nshots, item.hqcs) for item in estimate.items] == [
        (0, 10, 1.0),
        (1, 20, 2.0),
    ]


def test_unsupported_execution_modes(backend: backend_mod.NexusClientBackend) -> None:
    with pytest.raises(UnsupportedExecutionError, match="execute_circuit_repeated"):
        backend.execute_circuit_repeated(Circuit(1), nshots=10, repetitions=2)

    with pytest.raises(UnsupportedExecutionError, match="Distributed execution"):
        backend.execute_distributed_circuit(Circuit(1))

    with pytest.raises(UnsupportedExecutionError, match="initial_state"):
        backend.execute_circuit(make_measured_circuit(1), initial_state=[1, 0])

    with pytest.raises(UnsupportedExecutionError, match=r"(?i)shot-based"):
        backend.execute_circuit(Circuit(1), nshots=10)

    with pytest.raises(UnsupportedExecutionError, match=r"(?i)shot-based"):
        backend.estimate_circuit(Circuit(1), nshots=10)


def test_execute_forwards_user_max_cost_on_hseries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user-supplied max_cost must cap non-Helios (paid H-series) submissions
    too, not silently apply only to the Helios path."""
    calls: dict[str, object] = {}
    _patch_hseries_env(monkeypatch, _make_hseries_qnx(calls))
    monkeypatch.setattr(
        backend_mod.NexusClientBackend,
        "_upload_translated_program",
        lambda self, circuit, *, parameters=None, sequence_idx=0: (
            "program-ref",
            TranslationMetadata(measured_qubits=[0], nqubits=1, qasm="q"),
        ),
    )
    monkeypatch.setattr(
        backend_mod, "map_nexus_result_to_qibo", lambda **kwargs: "mapped"
    )

    backend = backend_mod.NexusClientBackend(
        platform="hseries:H2-1LE", project="proj", max_cost=10.0
    )
    backend.execute_circuit(make_measured_circuit(1), nshots=10)
    assert calls["execute"][-1]["max_cost"] == 10.0

    backend.execute_circuits(
        [make_measured_circuit(1), make_measured_circuit(1)], nshots=10
    )
    assert calls["execute"][-1]["max_cost"] == 10.0


def test_constructor_is_lazy_and_project_defaults_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"auth": 0, "project": 0, "config": 0}

    monkeypatch.setattr(backend_mod, "_ensure_nexus_dependencies", lambda: None)
    monkeypatch.setattr(
        backend_mod,
        "authenticate",
        lambda **kwargs: calls.__setitem__("auth", calls["auth"] + 1),
    )
    monkeypatch.setattr(
        backend_mod,
        "ensure_project",
        lambda project_name: calls.__setitem__("project", calls["project"] + 1)
        or project_name,
    )
    monkeypatch.setattr(
        backend_mod,
        "build_nexus_backend_config",
        lambda cfg: calls.__setitem__("config", calls["config"] + 1)
        or "backend-config",
    )

    backend = backend_mod.NexusClientBackend(platform="hseries:H2-1LE")

    assert backend.config.project is None
    assert calls == {"auth": 0, "project": 0, "config": 0}


def test_ensure_connected_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"auth": 0, "project": 0, "config": 0}

    monkeypatch.setattr(backend_mod, "_ensure_nexus_dependencies", lambda: None)
    monkeypatch.setattr(
        backend_mod,
        "authenticate",
        lambda **kwargs: calls.__setitem__("auth", calls["auth"] + 1),
    )
    monkeypatch.setattr(
        backend_mod,
        "ensure_project",
        lambda project_name: calls.__setitem__("project", calls["project"] + 1)
        or "project-ref",
    )
    monkeypatch.setattr(
        backend_mod,
        "build_nexus_backend_config",
        lambda cfg: calls.__setitem__("config", calls["config"] + 1)
        or "backend-config",
    )

    backend = backend_mod.NexusClientBackend(platform="hseries:H2-1LE", project="proj")
    backend._ensure_connected()
    backend._ensure_connected()

    assert calls == {"auth": 1, "project": 1, "config": 1}


def test_execute_circuit_helios_uses_hugr_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_mod, "_ensure_nexus_dependencies", lambda: None)
    monkeypatch.setattr(backend_mod, "authenticate", lambda **kwargs: None)
    monkeypatch.setattr(
        backend_mod, "ensure_project", lambda project_name: "project-ref"
    )

    build_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        backend_mod,
        "build_nexus_backend_config",
        lambda cfg, **kwargs: build_calls.append(kwargs) or "helios-backend-config",
    )
    monkeypatch.setattr(
        backend_mod,
        "build_helios_hugr_package",
        lambda circuit, parameters=None, entrypoint_name="helios_entrypoint": (
            "hugr-package",
            TranslationMetadata(measured_qubits=[0, 1], nqubits=2, qasm="OPENQASM"),
        ),
    )

    calls: dict[str, object] = {}

    class JobRef:
        id = "execute-job-1"

    qnx = types.SimpleNamespace(
        hugr=types.SimpleNamespace(
            upload=lambda *, hugr_package, name, project: calls.update(
                {"uploaded": (hugr_package, name, project)}
            )
            or "hugr-ref",
            cost_confidence=lambda *, programs, n_shots, **kw: calls.update(
                {"cost": (programs, n_shots)}
            )
            or [(1.25, 84.0)],
        ),
        circuits=types.SimpleNamespace(
            upload=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("circuits.upload called")
            )
        ),
        start_compile_job=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("start_compile_job called")
        ),
        start_execute_job=lambda **kwargs: calls.update({"execute": kwargs})
        or JobRef(),
        jobs=types.SimpleNamespace(
            wait_for=lambda job, timeout: job,
            results=lambda job, allow_incomplete=False: ["helios-result"],
            status=lambda job: "COMPLETED",
        ),
    )
    monkeypatch.setattr(backend_mod, "_import_qnexus", lambda: qnx)
    monkeypatch.setattr(
        backend_mod,
        "map_helios_result_to_qibo",
        lambda **kwargs: calls.update({"map": kwargs})
        or {"kind": "MeasurementOutcomes"},
    )

    backend = backend_mod.NexusClientBackend(
        platform="helios:Helios-1",
        project="proj",
        emulator=True,
    )
    circuit = make_measured_circuit(2)
    result = backend.execute_circuit(circuit, nshots=64)

    assert result["kind"] == "MeasurementOutcomes"
    assert calls["cost"] == (["hugr-ref"], [64])
    assert calls["execute"]["programs"] == ["hugr-ref"]
    assert "language" not in calls["execute"]
    # HeliosEmulatorConfig.n_qubits is deprecated: emulator sizing goes through
    # the per-item n_qubits kwarg on start_execute_job, never the config build.
    assert all("n_qubits" not in call for call in build_calls)
    assert calls["execute"]["n_qubits"] == 2
    # Estimated 1.25 HQC gets the default 1.2x headroom before max_cost submission.
    assert calls["execute"]["max_cost"] == pytest.approx(1.5)
    assert calls["map"]["execution_result_ref"] == "helios-result"


def test_estimate_circuit_helios_uses_hugr_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend_mod, "_ensure_nexus_dependencies", lambda: None)
    monkeypatch.setattr(backend_mod, "authenticate", lambda **kwargs: None)
    monkeypatch.setattr(
        backend_mod, "ensure_project", lambda project_name: "project-ref"
    )
    monkeypatch.setattr(
        backend_mod, "build_nexus_backend_config", lambda cfg, **kwargs: None
    )
    monkeypatch.setattr(
        backend_mod,
        "build_helios_hugr_package",
        lambda circuit, parameters=None, entrypoint_name="helios_entrypoint": (
            "hugr-package",
            TranslationMetadata(measured_qubits=[0], nqubits=1, qasm="OPENQASM"),
        ),
    )

    calls: dict[str, object] = {}
    qnx = types.SimpleNamespace(
        hugr=types.SimpleNamespace(
            upload=lambda *, hugr_package, name, project: "hugr-ref",
            cost_confidence=lambda *, programs, n_shots, **kw: calls.update(
                {"cost": (programs, n_shots)}
            )
            or [(2.5, 84.0)],
        ),
        circuits=types.SimpleNamespace(
            upload=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("circuits.upload called")
            )
        ),
        start_compile_job=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("start_compile_job called")
        ),
    )
    monkeypatch.setattr(backend_mod, "_import_qnexus", lambda: qnx)

    backend = backend_mod.NexusClientBackend(
        platform="helios:Helios-1",
        project="proj",
        emulator=True,
    )
    estimate = backend.estimate_circuit(make_measured_circuit(1), nshots=11)

    assert calls["cost"] == (["hugr-ref"], [11])
    assert estimate.total_hqcs == 2.5
    assert estimate.items[0].nshots == 11


def test_estimate_circuits_helios_submits_single_batch_cost_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend_mod, "_ensure_nexus_dependencies", lambda: None)
    monkeypatch.setattr(backend_mod, "authenticate", lambda **kwargs: None)
    monkeypatch.setattr(
        backend_mod, "ensure_project", lambda project_name: "project-ref"
    )
    monkeypatch.setattr(
        backend_mod, "build_nexus_backend_config", lambda cfg, **kwargs: None
    )
    monkeypatch.setattr(
        backend_mod,
        "build_helios_hugr_package",
        lambda circuit, parameters=None, entrypoint_name="helios_entrypoint": (
            "hugr-package",
            TranslationMetadata(measured_qubits=[0], nqubits=1, qasm="OPENQASM"),
        ),
    )

    cost_calls: list[dict[str, object]] = []
    qnx = types.SimpleNamespace(
        hugr=types.SimpleNamespace(
            upload=lambda *, hugr_package, name, project: f"hugr-ref-{name}",
            cost_confidence=lambda *, programs, n_shots, **kw: cost_calls.append(
                {"programs": list(programs), "n_shots": list(n_shots)}
            )
            or [(1.5, 84.0), (2.5, 84.0)],
        ),
        circuits=types.SimpleNamespace(
            upload=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("circuits.upload called")
            )
        ),
        start_compile_job=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("start_compile_job called")
        ),
    )
    monkeypatch.setattr(backend_mod, "_import_qnexus", lambda: qnx)

    backend = backend_mod.NexusClientBackend(
        platform="helios:Helios-1",
        project="proj",
        emulator=True,
    )
    circuits = [make_measured_circuit(1), make_measured_circuit(1)]
    estimate = backend.estimate_circuits(circuits, nshots=[10, 20])

    assert len(cost_calls) == 1
    assert len(cost_calls[0]["programs"]) == 2
    assert cost_calls[0]["n_shots"] == [10, 20]
    assert estimate.total_hqcs == 4.0
    assert estimate.items[0].nshots == 10
    assert estimate.items[0].hqcs == 1.5
    assert estimate.items[1].nshots == 20
    assert estimate.items[1].hqcs == 2.5
    assert estimate.batch_mode is False


def test_execute_circuits_helios_emulator_propagates_per_program_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for batched Helios execute_circuits.

    Verifies that the execute path:
      - sizes the emulator state with the maximum metadata.nqubits across circuits
      - passes per-program max_cost (as a list) to qnx.start_execute_job
      - does NOT inject attempt_batching=True into backend_options
        (vendor: batching is unsupported on Helios emulators).
    """
    monkeypatch.setattr(backend_mod, "_ensure_nexus_dependencies", lambda: None)
    monkeypatch.setattr(backend_mod, "authenticate", lambda **kwargs: None)
    monkeypatch.setattr(
        backend_mod, "ensure_project", lambda project_name: "project-ref"
    )

    build_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        backend_mod,
        "build_nexus_backend_config",
        lambda cfg, **kwargs: build_calls.append({"cfg": cfg, **kwargs})
        or "helios-backend-config",
    )

    metadata_by_idx = [
        TranslationMetadata(measured_qubits=[0], nqubits=1, qasm="q1"),
        TranslationMetadata(measured_qubits=[0, 1, 2], nqubits=3, qasm="q3"),
    ]

    def fake_build(circuit, parameters=None, entrypoint_name="helios_entrypoint"):
        # Return a metadata object whose nqubits matches the circuit width.
        idx = circuit.nqubits - 1 if circuit.nqubits == 1 else 1
        return f"hugr-package-{idx}", metadata_by_idx[idx]

    monkeypatch.setattr(backend_mod, "build_helios_hugr_package", fake_build)

    calls: dict[str, object] = {}

    class JobRef:
        id = "execute-job-batch"

    qnx = types.SimpleNamespace(
        hugr=types.SimpleNamespace(
            upload=lambda *, hugr_package, name, project: f"hugr-ref-{hugr_package}",
            cost_confidence=lambda *, programs, n_shots, **kw: calls.update(
                {
                    "cost_programs": list(programs),
                    "cost_n_shots": list(n_shots),
                    "cost_system_name": kw.get("system_name"),
                }
            )
            or [(1.5, 10.0), (4.25, 12.0)],
        ),
        circuits=types.SimpleNamespace(
            upload=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("circuits.upload called")
            )
        ),
        start_compile_job=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("start_compile_job called")
        ),
        start_execute_job=lambda **kwargs: calls.update({"execute": kwargs})
        or JobRef(),
        jobs=types.SimpleNamespace(
            wait_for=lambda job, timeout: job,
            results=lambda job, allow_incomplete=False: ["res-0", "res-1"],
            status=lambda job: "COMPLETED",
        ),
    )
    monkeypatch.setattr(backend_mod, "_import_qnexus", lambda: qnx)
    monkeypatch.setattr(
        backend_mod,
        "map_helios_result_to_qibo",
        lambda **kwargs: f"mapped-{kwargs['execution_result_ref']}",
    )

    backend = backend_mod.NexusClientBackend(
        platform="helios:Helios-1E",
        project="proj",
        emulator=True,
    )
    circuits = [make_measured_circuit(1), make_measured_circuit(3)]
    results = backend.execute_circuits(circuits, nshots=[10, 30])

    assert results == ["mapped-res-0", "mapped-res-1"]

    # Per-program max_cost list and shots flow into start_execute_job (vendor
    # pattern), each estimate padded with the default 1.2x headroom factor.
    assert calls["execute"]["max_cost"] == pytest.approx([1.8, 5.1])
    assert calls["execute"]["n_shots"] == [10, 30]

    # Emulator sizing is per-program via start_execute_job(n_qubits=[...]);
    # the deprecated HeliosEmulatorConfig.n_qubits is no longer auto-injected.
    assert calls["execute"]["n_qubits"] == [1, 3]
    last_build = build_calls[-1]
    assert "n_qubits" not in last_build

    # No batching auto-injection on emulator (vendor: unsupported on Helios emulators).
    cfg = last_build["cfg"]
    assert "attempt_batching" not in cfg.backend_options
    assert "max_batch_cost" not in cfg.backend_options

    # Cost estimation always targets Helios-1 syntax checker — qnexus internally builds
    # `QuantinuumConfig(device_name=f"{system_name}SC")`, and only "Helios-1SC" exists.
    # Even when the user-target platform is Helios-1E, system_name must stay "Helios-1".
    assert calls["cost_system_name"] == "Helios-1"


def _make_helios_qnx(
    calls: dict[str, object], *, cost_items: list[tuple[float, float]]
) -> types.SimpleNamespace:
    class JobRef:
        id = "execute-job-1"

    return types.SimpleNamespace(
        hugr=types.SimpleNamespace(
            upload=lambda *, hugr_package, name, project: "hugr-ref",
            cost_confidence=lambda *, programs, n_shots, **kw: calls.update(
                {"cost": (list(programs), list(n_shots))}
            )
            or cost_items,
        ),
        start_execute_job=lambda **kwargs: calls.update({"execute": kwargs})
        or JobRef(),
        jobs=types.SimpleNamespace(
            wait_for=lambda job, timeout: job,
            results=lambda job, allow_incomplete=False: ["helios-result"],
            status=lambda job: "COMPLETED",
        ),
    )


def _patch_helios_env(monkeypatch: pytest.MonkeyPatch, qnx: types.SimpleNamespace):
    monkeypatch.setattr(backend_mod, "_ensure_nexus_dependencies", lambda: None)
    monkeypatch.setattr(backend_mod, "authenticate", lambda **kwargs: None)
    monkeypatch.setattr(
        backend_mod, "ensure_project", lambda project_name: "project-ref"
    )
    monkeypatch.setattr(
        backend_mod,
        "build_nexus_backend_config",
        lambda cfg, **kwargs: "helios-backend-config",
    )
    monkeypatch.setattr(
        backend_mod,
        "build_helios_hugr_package",
        lambda circuit, parameters=None, entrypoint_name="helios_entrypoint": (
            "hugr-package",
            TranslationMetadata(measured_qubits=[0], nqubits=1, qasm="OPENQASM"),
        ),
    )
    monkeypatch.setattr(backend_mod, "_import_qnexus", lambda: qnx)
    monkeypatch.setattr(
        backend_mod,
        "map_helios_result_to_qibo",
        lambda **kwargs: {"kind": "MeasurementOutcomes"},
    )


def _make_hseries_qnx(
    calls: dict[str, object], *, execute_items: list[object] | None = None
) -> types.SimpleNamespace:
    """qnexus stand-in covering the full hseries compile->execute pipeline."""

    class CompileJobRef:
        id = "compile-job-1"

    class ExecuteJobRef:
        id = "execute-job-1"

    class CompiledItem:
        def get_output(self):
            return "compiled-program"

    compile_ref = CompileJobRef()
    execute_ref = ExecuteJobRef()

    def start_compile_job(**kwargs):
        calls.setdefault("compile", []).append(kwargs)
        return compile_ref

    def start_execute_job(**kwargs):
        calls.setdefault("execute", []).append(kwargs)
        return execute_ref

    def results(job, allow_incomplete=False):
        if isinstance(job, CompileJobRef):
            return [CompiledItem() for _ in calls["compile"][-1]["programs"]]
        if execute_items is not None:
            return list(execute_items)
        return [
            f"execution-item-{i}"
            for i in range(len(calls["execute"][-1]["programs"]))
        ]

    return types.SimpleNamespace(
        start_compile_job=start_compile_job,
        start_execute_job=start_execute_job,
        jobs=types.SimpleNamespace(
            wait_for=lambda job, timeout=None: job,
            results=results,
            status=lambda job: "COMPLETED",
            cancel=lambda job: calls.setdefault("cancel", []).append(job),
            get=lambda **kwargs: calls.update({"get": kwargs}) or execute_ref,
        ),
    )


def _patch_hseries_env(
    monkeypatch: pytest.MonkeyPatch, qnx: types.SimpleNamespace
) -> None:
    monkeypatch.setattr(backend_mod, "_ensure_nexus_dependencies", lambda: None)
    monkeypatch.setattr(backend_mod, "authenticate", lambda **kwargs: None)
    monkeypatch.setattr(
        backend_mod, "ensure_project", lambda project_name: "project-ref"
    )
    monkeypatch.setattr(
        backend_mod, "build_nexus_backend_config", lambda cfg: "backend-config"
    )
    monkeypatch.setattr(backend_mod, "_import_qnexus", lambda: qnx)


def test_execute_circuit_helios_rejects_non_positive_cost_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A (0.0, -1.0) tuple means the server omitted cost data; submitting with
    max_cost=0.0 would instantly deplete the job, so execution must abort."""
    calls: dict[str, object] = {}
    _patch_helios_env(monkeypatch, _make_helios_qnx(calls, cost_items=[(0.0, -1.0)]))

    backend = backend_mod.NexusClientBackend(
        platform="helios:Helios-1", project="proj", emulator=True
    )
    with pytest.raises(NexusBackendError, match="(?i)cost"):
        backend.execute_circuit(make_measured_circuit(1), nshots=10)

    assert "execute" not in calls


def test_execute_circuits_helios_rejects_non_positive_batch_cost_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batch estimator has its own validation loop; a single (0.0, -1.0)
    item anywhere in the batch must abort before any submission."""
    calls: dict[str, object] = {}
    qnx = _make_helios_qnx(calls, cost_items=[(1.5, 84.0), (0.0, -1.0)])
    _patch_helios_env(monkeypatch, qnx)

    backend = backend_mod.NexusClientBackend(
        platform="helios:Helios-1", project="proj", emulator=True
    )
    circuits = [make_measured_circuit(1), make_measured_circuit(1)]
    with pytest.raises(NexusBackendError, match="(?i)cost"):
        backend.execute_circuits(circuits, nshots=[10, 20])

    assert "execute" not in calls


def test_execute_circuit_helios_user_n_qubits_overrides_per_item_sizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user who explicitly over-provisions the emulator (n_qubits in
    backend_options) must not be silently shrunk back to circuit width by the
    per-item sizing hint."""
    calls: dict[str, object] = {}
    _patch_helios_env(monkeypatch, _make_helios_qnx(calls, cost_items=[(1.0, 84.0)]))

    backend = backend_mod.NexusClientBackend(
        platform="helios:Helios-1E",
        project="proj",
        emulator=True,
        n_qubits=40,
    )
    backend.execute_circuit(make_measured_circuit(1), nshots=10)

    assert calls["execute"]["n_qubits"] == 40


def test_execute_circuit_helios_user_max_cost_skips_estimation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    _patch_helios_env(monkeypatch, _make_helios_qnx(calls, cost_items=[(1.0, 84.0)]))

    backend = backend_mod.NexusClientBackend(
        platform="helios:Helios-1",
        project="proj",
        emulator=True,
        max_cost=7.5,
    )
    result = backend.execute_circuit(make_measured_circuit(1), nshots=10)

    assert result["kind"] == "MeasurementOutcomes"
    assert "cost" not in calls
    assert calls["execute"]["max_cost"] == 7.5


def test_execute_circuits_helios_user_max_cost_skips_estimation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    qnx = _make_helios_qnx(calls, cost_items=[(1.0, 84.0), (2.0, 84.0)])
    qnx.jobs.results = lambda job, allow_incomplete=False: ["res-0", "res-1"]
    _patch_helios_env(monkeypatch, qnx)
    monkeypatch.setattr(
        backend_mod,
        "map_helios_result_to_qibo",
        lambda **kwargs: f"mapped-{kwargs['execution_result_ref']}",
    )

    backend = backend_mod.NexusClientBackend(
        platform="helios:Helios-1",
        project="proj",
        emulator=True,
        max_cost=9.0,
    )
    circuits = [make_measured_circuit(1), make_measured_circuit(1)]
    results = backend.execute_circuits(circuits, nshots=[10, 20])

    assert results == ["mapped-res-0", "mapped-res-1"]
    assert "cost" not in calls
    # A scalar user max_cost is forwarded as-is; qnexus broadcasts it per program.
    assert calls["execute"]["max_cost"] == 9.0


def test_execute_circuit_helios_hardware_omits_n_qubits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """n_qubits is an emulator sizing hint; hardware submissions must not
    carry it."""
    calls: dict[str, object] = {}
    _patch_helios_env(monkeypatch, _make_helios_qnx(calls, cost_items=[(1.0, 84.0)]))

    backend = backend_mod.NexusClientBackend(platform="helios:Helios-1", project="proj")
    backend.execute_circuit(make_measured_circuit(1), nshots=10)

    assert "n_qubits" not in calls["execute"]


def test_execute_circuit_helios_max_cost_factor_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    _patch_helios_env(monkeypatch, _make_helios_qnx(calls, cost_items=[(1.25, 84.0)]))

    backend = backend_mod.NexusClientBackend(
        platform="helios:Helios-1",
        project="proj",
        emulator=True,
        max_cost_factor=2.0,
    )
    backend.execute_circuit(make_measured_circuit(1), nshots=10)

    assert calls["execute"]["max_cost"] == pytest.approx(2.5)


def test_execute_and_estimate_circuits_trivial_inputs(
    backend: backend_mod.NexusClientBackend,
) -> None:
    circuits = [make_measured_circuit(1)]
    with pytest.raises(UnsupportedExecutionError, match="initial_states"):
        backend.execute_circuits(circuits, initial_states=[1, 0])
    with pytest.raises(UnsupportedExecutionError, match="initial_states"):
        backend.estimate_circuits(circuits, initial_states=[1, 0])

    assert backend.execute_circuits([]) == []
    empty_estimate = backend.estimate_circuits([])
    assert empty_estimate.total_hqcs == 0.0
    assert empty_estimate.items == []


def test_execute_and_estimate_circuits_parameters_cardinality_mismatch(
    backend: backend_mod.NexusClientBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    circuits = [make_measured_circuit(1), make_measured_circuit(1)]
    with pytest.raises(ValueError, match="parameters_list cardinality mismatch"):
        backend.execute_circuits(circuits, nshots=10, parameters_list=[None])
    with pytest.raises(ValueError, match="parameters_list cardinality mismatch"):
        backend.estimate_circuits(circuits, nshots=10, parameters_list=[None])

    calls: dict[str, object] = {}
    _patch_helios_env(monkeypatch, _make_helios_qnx(calls, cost_items=[(1.0, 84.0)]))
    helios_backend = backend_mod.NexusClientBackend(
        platform="helios:Helios-1", project="proj"
    )
    with pytest.raises(ValueError, match="parameters_list cardinality mismatch"):
        helios_backend.execute_circuits(circuits, nshots=10, parameters_list=[None])


def test_execute_circuits_helios_scalar_nshots_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    qnx = _make_helios_qnx(calls, cost_items=[(1.0, 84.0), (2.0, 84.0)])
    qnx.jobs.results = lambda job, allow_incomplete=False: ["res-0", "res-1"]
    _patch_helios_env(monkeypatch, qnx)

    backend = backend_mod.NexusClientBackend(platform="helios:Helios-1", project="proj")
    circuits = [make_measured_circuit(1), make_measured_circuit(1)]
    results = backend.execute_circuits(circuits, nshots=50)

    assert len(results) == 2
    assert calls["cost"] == (["hugr-ref", "hugr-ref"], [50, 50])
    assert calls["execute"]["n_shots"] == [50, 50]


def test_execute_circuits_helios_result_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch that comes back with fewer results than circuits must fail loudly
    instead of silently mapping results to the wrong circuits."""
    calls: dict[str, object] = {}
    qnx = _make_helios_qnx(calls, cost_items=[(1.0, 84.0), (2.0, 84.0)])
    qnx.jobs.results = lambda job, allow_incomplete=False: ["res-0"]
    _patch_helios_env(monkeypatch, qnx)

    backend = backend_mod.NexusClientBackend(platform="helios:Helios-1", project="proj")
    circuits = [make_measured_circuit(1), make_measured_circuit(1)]
    with pytest.raises(NexusBackendError, match="expected 2, got 1 items"):
        backend.execute_circuits(circuits, nshots=[10, 20])


def test_execute_circuits_non_batch_runs_sequentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    _patch_hseries_env(monkeypatch, _make_hseries_qnx(calls))
    monkeypatch.setattr(
        backend_mod.NexusClientBackend,
        "_upload_translated_program",
        lambda self, circuit, *, parameters=None, sequence_idx=0: (
            f"program-ref-{sequence_idx}",
            TranslationMetadata(measured_qubits=[0], nqubits=1, qasm="q"),
        ),
    )
    monkeypatch.setattr(
        backend_mod,
        "map_nexus_result_to_qibo",
        lambda **kwargs: f"mapped-{kwargs['nshots']}",
    )

    backend = backend_mod.NexusClientBackend(
        platform="hseries:H2-1LE", project="proj", batch_mode=False
    )
    circuits = [make_measured_circuit(1), make_measured_circuit(1)]

    assert backend.execute_circuits(circuits, nshots=7) == ["mapped-7", "mapped-7"]
    assert [k["n_shots"] for k in calls["execute"]] == [7, 7]

    calls["execute"].clear()
    calls["compile"].clear()
    assert backend.execute_circuits(circuits, nshots=[5, 6]) == ["mapped-5", "mapped-6"]
    assert [k["n_shots"] for k in calls["execute"]] == [5, 6]

    with pytest.raises(ValueError, match="nshots cardinality mismatch"):
        backend.execute_circuits(circuits, nshots=[5])
    with pytest.raises(ValueError, match="parameters_list cardinality mismatch"):
        backend.execute_circuits(circuits, nshots=7, parameters_list=[None])


def test_execute_circuits_batch_result_cardinality_mismatch(
    backend: backend_mod.NexusClientBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        backend_mod.NexusClientBackend,
        "_upload_translated_program",
        lambda self, circuit, *, parameters=None, sequence_idx=0: (
            f"program-ref-{sequence_idx}",
            TranslationMetadata(measured_qubits=[0], nqubits=1, qasm="q"),
        ),
    )
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        backend_mod,
        "_import_qnexus",
        lambda: _make_hseries_qnx(calls, execute_items=["only-one-item"]),
    )

    circuits = [make_measured_circuit(1), make_measured_circuit(1)]
    with pytest.raises(NexusBackendError, match="Result cardinality mismatch"):
        backend.execute_circuits(circuits, nshots=[10, 20])


def test_estimate_circuits_helios_scalar_nshots_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    _patch_helios_env(
        monkeypatch, _make_helios_qnx(calls, cost_items=[(1.5, 84.0), (2.5, 84.0)])
    )

    backend = backend_mod.NexusClientBackend(platform="helios:Helios-1", project="proj")
    circuits = [make_measured_circuit(1), make_measured_circuit(1)]
    estimate = backend.estimate_circuits(circuits, nshots=30)

    assert calls["cost"] == (["hugr-ref", "hugr-ref"], [30, 30])
    assert [item.nshots for item in estimate.items] == [30, 30]
    assert estimate.total_hqcs == 4.0


def test_estimate_circuits_non_batch_scalar_and_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend_mod, "_ensure_nexus_dependencies", lambda: None)
    monkeypatch.setattr(backend_mod, "authenticate", lambda **kwargs: None)
    monkeypatch.setattr(
        backend_mod, "ensure_project", lambda project_name: "project-ref"
    )
    monkeypatch.setattr(
        backend_mod, "build_nexus_backend_config", lambda cfg: "backend-config"
    )
    monkeypatch.setattr(backend_mod, "_import_qnexus", lambda: types.SimpleNamespace())
    monkeypatch.setattr(
        backend_mod.NexusClientBackend,
        "_upload_translated_program",
        lambda self, circuit, *, parameters=None, sequence_idx=0: (
            f"program-ref-{sequence_idx}",
            TranslationMetadata(measured_qubits=[0], nqubits=1, qasm="q"),
        ),
    )
    monkeypatch.setattr(
        backend_mod,
        "_prepare_compiled_programs",
        lambda **kwargs: backend_mod._PreparedCompilation(
            compiled_programs=["compiled"],
            submission_n_shots=kwargs["n_shots"],
            shot_values=[kwargs["n_shots"]],
            compile_job_id="compile-1",
            batch_mode=False,
        ),
    )
    monkeypatch.setattr(
        backend_mod,
        "_estimate_prepared_compilation",
        lambda **kwargs: backend_mod.ExecutionEstimate(
            platform="hseries:H2-1LE",
            optimisation_level=2,
            batch_mode=False,
            total_hqcs=1.0,
            items=[
                backend_mod.EstimateItem(
                    sequence_idx=0,
                    nshots=kwargs["prepared"].shot_values[0],
                    hqcs=1.0,
                    compile_job_id="compile-1",
                )
            ],
        ),
    )

    backend = backend_mod.NexusClientBackend(
        platform="hseries:H2-1LE", project="proj", batch_mode=False
    )
    circuits = [make_measured_circuit(1), make_measured_circuit(1)]

    estimate = backend.estimate_circuits(circuits, nshots=15)
    assert [item.nshots for item in estimate.items] == [15, 15]

    with pytest.raises(ValueError, match="nshots cardinality mismatch"):
        backend.estimate_circuits(circuits, nshots=[15])


def test_upload_translated_program_wraps_upload_failures(
    backend: backend_mod.NexusClientBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        backend_mod,
        "translate_qibo_to_pytket",
        lambda circuit, parameters=None: (
            "pytket-circuit",
            TranslationMetadata(measured_qubits=[0], nqubits=1, qasm="q"),
        ),
    )
    failing_qnx = types.SimpleNamespace(
        circuits=types.SimpleNamespace(
            upload=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("full"))
        ),
        hugr=types.SimpleNamespace(
            upload=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("full"))
        ),
    )
    monkeypatch.setattr(backend_mod, "_import_qnexus", lambda: failing_qnx)

    with pytest.raises(NexusBackendError, match="Failed to upload circuit"):
        backend._upload_translated_program(make_measured_circuit(1))

    monkeypatch.setattr(
        backend_mod,
        "build_helios_hugr_package",
        lambda circuit, parameters=None, entrypoint_name="helios_entrypoint": (
            "hugr-package",
            TranslationMetadata(measured_qubits=[0], nqubits=1, qasm="q"),
        ),
    )
    monkeypatch.setattr(backend_mod, "authenticate", lambda **kwargs: None)
    monkeypatch.setattr(
        backend_mod, "ensure_project", lambda project_name: "project-ref"
    )
    monkeypatch.setattr(
        backend_mod, "build_nexus_backend_config", lambda cfg: "helios-config"
    )
    monkeypatch.setattr(backend_mod, "_ensure_nexus_dependencies", lambda: None)
    helios_backend = backend_mod.NexusClientBackend(
        platform="helios:Helios-1", project="proj"
    )
    with pytest.raises(NexusBackendError, match="Failed to upload Helios HUGR"):
        helios_backend._upload_translated_program(make_measured_circuit(1))


def _patch_simple_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        backend_mod.NexusClientBackend,
        "_upload_translated_program",
        lambda self, circuit, *, parameters=None, sequence_idx=0: (
            f"program-ref-{sequence_idx}",
            TranslationMetadata(measured_qubits=[0], nqubits=1, qasm="q"),
        ),
    )


def test_submit_circuit_returns_reusable_handle(
    backend: backend_mod.NexusClientBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        backend_mod,
        "_import_qnexus",
        lambda: _make_hseries_qnx(calls, execute_items=["execution-item"]),
    )
    _patch_simple_upload(monkeypatch)
    monkeypatch.setattr(
        backend_mod, "map_nexus_result_to_qibo", lambda **kwargs: "mapped"
    )

    job = backend.submit_circuit(make_measured_circuit(1), nshots=5)

    assert isinstance(job, NexusJob)
    assert job.job_id == "execute-job-1"
    assert calls["execute"][-1]["n_shots"] == 5
    # Compile stage already ran (block-through-compile semantics).
    assert calls["compile"][-1]["programs"] == ["program-ref-0"]
    assert job.done() is True
    assert job.result() == "mapped"


def test_execute_circuit_blocking_false_returns_handle(
    backend: backend_mod.NexusClientBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        backend_mod,
        "_import_qnexus",
        lambda: _make_hseries_qnx(calls, execute_items=["execution-item"]),
    )
    _patch_simple_upload(monkeypatch)
    monkeypatch.setattr(
        backend_mod, "map_nexus_result_to_qibo", lambda **kwargs: "mapped"
    )

    job = backend.execute_circuit(make_measured_circuit(1), nshots=5, blocking=False)
    assert isinstance(job, NexusJob)
    assert job.result() == "mapped"


def test_backend_level_blocking_false_routes_execute_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    _patch_hseries_env(
        monkeypatch, _make_hseries_qnx(calls, execute_items=["execution-item"])
    )
    _patch_simple_upload(monkeypatch)
    monkeypatch.setattr(
        backend_mod, "map_nexus_result_to_qibo", lambda **kwargs: "mapped"
    )

    nonblocking = backend_mod.NexusClientBackend(
        platform="hseries:H2-1LE", project="proj", blocking=False
    )
    handle = nonblocking.execute_circuit(make_measured_circuit(1), nshots=5)
    assert isinstance(handle, NexusJob)
    # Per-call override still wins over the config default.
    result = nonblocking.execute_circuit(
        make_measured_circuit(1), nshots=5, blocking=True
    )
    assert result == "mapped"


def test_blocking_execute_wraps_timeout_error(
    backend: backend_mod.NexusClientBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}
    qnx = _make_hseries_qnx(calls, execute_items=["execution-item"])

    def flaky_wait(job, timeout=None):
        if getattr(job, "id", "") == "execute-job-1":
            raise TimeoutError("still queued")
        return job

    qnx.jobs.wait_for = flaky_wait
    monkeypatch.setattr(backend_mod, "_import_qnexus", lambda: qnx)
    _patch_simple_upload(monkeypatch)

    with pytest.raises(NexusBackendError, match="timed out/failed while waiting"):
        backend.execute_circuit(make_measured_circuit(1), nshots=5)


def test_blocking_execute_wraps_timeout_error_when_status_lookup_fails(
    backend: backend_mod.NexusClientBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Status retrieval in the timeout handler is best-effort: if it also
    raises, the wrapped NexusBackendError still reports status=unknown
    instead of masking the original timeout with a new exception."""
    calls: dict[str, object] = {}
    qnx = _make_hseries_qnx(calls, execute_items=["execution-item"])

    def flaky_wait(job, timeout=None):
        if getattr(job, "id", "") == "execute-job-1":
            raise TimeoutError("still queued")
        return job

    def failing_status(job):
        raise RuntimeError("status endpoint unavailable")

    qnx.jobs.wait_for = flaky_wait
    qnx.jobs.status = failing_status
    monkeypatch.setattr(backend_mod, "_import_qnexus", lambda: qnx)
    _patch_simple_upload(monkeypatch)

    with pytest.raises(NexusBackendError, match="timed out/failed while waiting"):
        backend.execute_circuit(make_measured_circuit(1), nshots=5)


def test_submit_circuits_batched_single_part(
    backend: backend_mod.NexusClientBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        backend_mod, "_import_qnexus", lambda: _make_hseries_qnx(calls)
    )
    _patch_simple_upload(monkeypatch)
    monkeypatch.setattr(
        backend_mod,
        "map_nexus_result_to_qibo",
        lambda **kwargs: f"mapped-{kwargs['execution_result_ref']}",
    )

    circuits = [make_measured_circuit(1), make_measured_circuit(1)]
    job = backend.submit_circuits(circuits, nshots=9)

    assert isinstance(job, NexusJob)
    assert job.job_ids == ("execute-job-1",)
    assert len(calls["execute"]) == 1
    assert calls["execute"][-1]["n_shots"] == [9, 9]
    assert job.result() == ["mapped-execution-item-0", "mapped-execution-item-1"]


def test_submit_circuits_non_batch_multi_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    _patch_hseries_env(monkeypatch, _make_hseries_qnx(calls))
    _patch_simple_upload(monkeypatch)
    monkeypatch.setattr(
        backend_mod,
        "map_nexus_result_to_qibo",
        lambda **kwargs: f"mapped-{kwargs['nshots']}",
    )

    non_batch = backend_mod.NexusClientBackend(
        platform="hseries:H2-1LE", project="proj", batch_mode=False
    )
    circuits = [make_measured_circuit(1), make_measured_circuit(1)]
    job = non_batch.submit_circuits(circuits, nshots=[5, 6])

    assert len(job.job_ids) == 2
    assert [k["n_shots"] for k in calls["execute"]] == [5, 6]
    with pytest.raises(ValueError, match="job_ids"):
        _ = job.job_id
    assert job.result() == ["mapped-5", "mapped-6"]


def test_submit_circuits_rejects_empty_and_execute_keeps_returning_list(
    backend: backend_mod.NexusClientBackend,
) -> None:
    with pytest.raises(ValueError, match="at least one circuit"):
        backend.submit_circuits([])
    assert backend.execute_circuits([]) == []


def test_submit_circuits_rejects_initial_states(
    backend: backend_mod.NexusClientBackend,
) -> None:
    with pytest.raises(UnsupportedExecutionError, match="initial_states"):
        backend.submit_circuits([make_measured_circuit(1)], initial_states=[1, 0])


def test_execute_circuits_empty_and_non_blocking_raises(
    backend: backend_mod.NexusClientBackend,
) -> None:
    with pytest.raises(ValueError, match="at least one circuit"):
        backend.execute_circuits([], blocking=False)


def test_execute_circuits_blocking_false_returns_handle(
    backend: backend_mod.NexusClientBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        backend_mod, "_import_qnexus", lambda: _make_hseries_qnx(calls)
    )
    _patch_simple_upload(monkeypatch)
    monkeypatch.setattr(
        backend_mod,
        "map_nexus_result_to_qibo",
        lambda **kwargs: f"mapped-{kwargs['execution_result_ref']}",
    )

    circuits = [make_measured_circuit(1), make_measured_circuit(1)]
    job = backend.execute_circuits(circuits, nshots=3, blocking=False)
    assert isinstance(job, NexusJob)
    assert job.result() == ["mapped-execution-item-0", "mapped-execution-item-1"]
