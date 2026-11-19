from __future__ import annotations

import types

from qibo import gates
from qibo.models import Circuit

from nexus.helios import (
    _build_entrypoint_source,
    build_helios_hugr_package,
    map_helios_result_to_qibo,
)
from nexus.translation import TranslationMetadata


def test_build_entrypoint_source_preserves_measurement_order() -> None:
    source = _build_entrypoint_source(
        loaded_name="loaded_pytket",
        entrypoint_name="helios_entrypoint",
        metadata=TranslationMetadata(
            measured_qubits=[2, 0], nqubits=3, qasm="OPENQASM"
        ),
    )

    assert 'result("m[0]", measure(q2).read())' in source
    assert 'result("m[1]", measure(q0).read())' in source
    assert "discard(q1)" in source


def test_build_helios_hugr_package_loads_pytket_with_rebasing(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "nexus.helios.translate_qibo_to_pytket_for_helios",
        lambda circuit, parameters=None: (
            "pytket-circuit",
            TranslationMetadata(measured_qubits=[0], nqubits=1, qasm="OPENQASM"),
        ),
    )

    class DecomposeBoxes:
        def apply(self, circuit):
            calls["decompose"] = circuit

    class AutoRebase:
        def __init__(self, gate_set):
            calls["gate_set"] = gate_set

        def apply(self, circuit):
            calls["rebase"] = circuit

    pytket_circuit_mod = types.ModuleType("pytket.circuit")
    pytket_circuit_mod.OpType = types.SimpleNamespace(CX="CX", H="H", Rz="Rz")
    pytket_passes_mod = types.ModuleType("pytket.passes")
    pytket_passes_mod.DecomposeBoxes = DecomposeBoxes
    pytket_passes_mod.AutoRebase = AutoRebase

    class FakeLoaded:
        def __call__(self, *args):
            return None

    class FakeEntrypoint:
        def compile(self):
            return {"kind": "hugr-package"}

    class FakeGuppy:
        def load_pytket(self, name, circuit, use_arrays=False):
            calls["load_pytket"] = {
                "name": name,
                "circuit": circuit,
                "use_arrays": use_arrays,
            }
            return FakeLoaded()

        def __call__(self, fn):
            calls["decorated_name"] = fn.__name__
            return FakeEntrypoint()

    guppy_mod = types.ModuleType("guppylang")
    guppy_mod.guppy = FakeGuppy()
    builtins_mod = types.ModuleType("guppylang.std.builtins")
    builtins_mod.result = lambda tag, value: (tag, value)
    quantum_mod = types.ModuleType("guppylang.std.quantum")
    quantum_mod.qubit = lambda: object()
    quantum_mod.measure = lambda q: True
    quantum_mod.discard = lambda q: None

    monkeypatch.setitem(__import__("sys").modules, "pytket.circuit", pytket_circuit_mod)
    monkeypatch.setitem(__import__("sys").modules, "pytket.passes", pytket_passes_mod)
    monkeypatch.setitem(__import__("sys").modules, "guppylang", guppy_mod)
    monkeypatch.setitem(
        __import__("sys").modules, "guppylang.std.builtins", builtins_mod
    )
    monkeypatch.setitem(__import__("sys").modules, "guppylang.std.quantum", quantum_mod)

    circuit = Circuit(1)
    circuit.add(gates.M(0))
    package, metadata = build_helios_hugr_package(circuit)

    assert package == {"kind": "hugr-package"}
    assert metadata.measured_qubits == [0]
    assert calls["decompose"] == "pytket-circuit"
    assert calls["rebase"] == "pytket-circuit"
    assert calls["load_pytket"]["use_arrays"] is False


def test_map_helios_result_to_qibo_from_register_bitstrings(monkeypatch) -> None:
    """QsysResult.register_bitstrings() collates the entrypoint's m[idx] tags
    into per-shot bitstrings (index 0 leftmost, i.e. measurement order)."""

    class MeasurementOutcomes:
        def __init__(self, measurements, backend=None, nshots=0, samples=None):
            self.measurements = measurements
            self.backend = backend
            self.nshots = nshots
            self.samples = samples

    qibo_result = types.ModuleType("qibo.result")
    qibo_result.MeasurementOutcomes = MeasurementOutcomes
    monkeypatch.setitem(__import__("sys").modules, "qibo.result", qibo_result)

    class QsysLikeResult:
        def register_bitstrings(self):
            return {"m": ["10", "01", "10"]}

    class ExecutionResultRef:
        def download_result(self):
            return QsysLikeResult()

    circuit = Circuit(2)
    circuit.add(gates.M(1, 0))
    result = map_helios_result_to_qibo(
        execution_result_ref=ExecutionResultRef(),
        circuit=circuit,
        backend=object(),
        nshots=3,
        measured_qubits=[1, 0],
    )

    assert result.nshots == 3
    assert result.samples.shape == (3, 2)
    assert result.measurements[0] is not circuit.measurements[0]
    assert result.measurements[0].init_args == circuit.measurements[0].init_args
    assert result.measurements[0].register_name == circuit.measurements[0].register_name
    rows = [tuple(row) for row in result.samples.tolist()]
    assert rows.count((1, 0)) == 2
    assert rows.count((0, 1)) == 1


import pytest

from nexus.errors import NexusBackendError, NexusResultMappingError
from nexus.helios import _extract_helios_counts


def _install_toolchain(monkeypatch, guppy, *, decompose_raises: bool = False) -> None:
    monkeypatch.setattr(
        "nexus.helios.translate_qibo_to_pytket_for_helios",
        lambda circuit, parameters=None: (
            "pytket-circuit",
            TranslationMetadata(measured_qubits=[0], nqubits=1, qasm="OPENQASM"),
        ),
    )

    class DecomposeBoxes:
        def apply(self, circuit):
            if decompose_raises:
                raise RuntimeError("cannot decompose")

    class AutoRebase:
        def __init__(self, gate_set):
            pass

        def apply(self, circuit):
            pass

    pytket_circuit_mod = types.ModuleType("pytket.circuit")
    pytket_circuit_mod.OpType = types.SimpleNamespace(CX="CX", H="H", Rz="Rz")
    pytket_passes_mod = types.ModuleType("pytket.passes")
    pytket_passes_mod.DecomposeBoxes = DecomposeBoxes
    pytket_passes_mod.AutoRebase = AutoRebase
    guppy_mod = types.ModuleType("guppylang")
    guppy_mod.guppy = guppy
    builtins_mod = types.ModuleType("guppylang.std.builtins")
    builtins_mod.result = lambda tag, value: (tag, value)
    quantum_mod = types.ModuleType("guppylang.std.quantum")
    quantum_mod.qubit = lambda: object()
    quantum_mod.measure = lambda q: True
    quantum_mod.discard = lambda q: None

    sys_modules = __import__("sys").modules
    monkeypatch.setitem(sys_modules, "pytket.circuit", pytket_circuit_mod)
    monkeypatch.setitem(sys_modules, "pytket.passes", pytket_passes_mod)
    monkeypatch.setitem(sys_modules, "guppylang", guppy_mod)
    monkeypatch.setitem(sys_modules, "guppylang.std.builtins", builtins_mod)
    monkeypatch.setitem(sys_modules, "guppylang.std.quantum", quantum_mod)


def _guppy(
    *,
    load_raises: bool = False,
    decorate_raises: bool = False,
    compile_raises: bool = False,
):
    class Loaded:
        def __call__(self, *args):
            return None

    class Entrypoint:
        def compile(self):
            if compile_raises:
                raise RuntimeError("hugr build exploded")
            return {"kind": "hugr-package"}

    class Guppy:
        def load_pytket(self, name, circuit, use_arrays=False):
            if load_raises:
                raise RuntimeError("unsupported gate")
            return Loaded()

        def __call__(self, fn):
            if decorate_raises:
                raise RuntimeError("guppy rejected entrypoint")
            return Entrypoint()

    return Guppy()


def _measured_circuit() -> Circuit:
    circuit = Circuit(1)
    circuit.add(gates.M(0))
    return circuit


@pytest.mark.parametrize(
    ("guppy_kwargs", "match"),
    [
        ({"load_raises": True}, "Failed to load pytket circuit into Guppy"),
        ({"decorate_raises": True}, "Failed to build Helios Guppy entrypoint"),
        ({"compile_raises": True}, "Failed to compile Helios HUGR package"),
    ],
)
def test_build_helios_hugr_package_wraps_toolchain_failures(
    monkeypatch, guppy_kwargs: dict, match: str
) -> None:
    _install_toolchain(monkeypatch, _guppy(**guppy_kwargs))
    with pytest.raises(NexusBackendError, match=match):
        build_helios_hugr_package(_measured_circuit())


def test_build_helios_hugr_package_wraps_rebase_failure(monkeypatch) -> None:
    _install_toolchain(monkeypatch, _guppy(), decompose_raises=True)
    with pytest.raises(NexusBackendError, match="Failed to normalize pytket circuit"):
        build_helios_hugr_package(_measured_circuit())


def test_build_entrypoint_source_zero_qubit_circuit() -> None:
    source = _build_entrypoint_source(
        loaded_name="loaded_pytket",
        entrypoint_name="ep",
        metadata=TranslationMetadata(measured_qubits=[], nqubits=0, qasm="q"),
    )
    assert "loaded_pytket()" in source
    assert "qubit()" not in source


def test_extract_helios_counts_rejects_non_qsys_results() -> None:
    with pytest.raises(NexusResultMappingError, match="backend result type"):
        _extract_helios_counts(object())


def test_extract_helios_counts_requires_measurement_register() -> None:
    class QsysLikeResult:
        def register_bitstrings(self):
            return {"other": ["0"]}

    with pytest.raises(NexusResultMappingError, match="has no 'm' register"):
        _extract_helios_counts(QsysLikeResult())


def test_map_helios_result_rejects_wrong_width_bitstrings(monkeypatch) -> None:
    """A malformed collation (e.g. bool entries collating to 'TrueFalse')
    must fail loudly instead of being padded or truncated into fake counts."""
    qibo_result = types.ModuleType("qibo.result")
    qibo_result.MeasurementOutcomes = object
    monkeypatch.setitem(__import__("sys").modules, "qibo.result", qibo_result)

    class QsysLikeResult:
        def register_bitstrings(self):
            return {"m": ["TrueFalse"]}

    class ExecutionResultRef:
        def download_result(self):
            return QsysLikeResult()

    circuit = Circuit(2)
    circuit.add(gates.M(0, 1))
    with pytest.raises(NexusResultMappingError, match="width mismatch"):
        map_helios_result_to_qibo(
            execution_result_ref=ExecutionResultRef(),
            circuit=circuit,
            backend=object(),
            nshots=1,
            measured_qubits=[0, 1],
        )
