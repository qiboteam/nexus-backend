from __future__ import annotations

import types

import pytest
import sympy
from qibo import gates
from qibo.models import Circuit

from nexus.errors import NexusBackendError
from nexus.translation import (
    prepare_qibo_circuit,
    translate_qibo_to_pytket,
    translate_qibo_to_pytket_for_helios,
)


def test_prepare_qibo_circuit_decomposes_multicontrol() -> None:
    circuit = Circuit(3)
    circuit.add(gates.TOFFOLI(0, 1, 2))
    circuit.add(gates.M(2, register_name="m0"))

    prepared, qasm = prepare_qibo_circuit(circuit)

    assert all(len(g.control_qubits) <= 1 for g in prepared.queue)
    assert "OPENQASM" in qasm


def test_prepare_qibo_circuit_raises_for_unbound_params() -> None:
    theta = sympy.Symbol("theta")
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=theta))
    circuit.add(gates.M(0))

    with pytest.raises(NexusBackendError):
        prepare_qibo_circuit(circuit)


def test_translate_qibo_to_pytket(monkeypatch: pytest.MonkeyPatch) -> None:
    def circuit_from_qasm_str(source: str):
        return {"parsed": source}

    pytket_mod = types.ModuleType("pytket")
    qasm_mod = types.ModuleType("pytket.qasm")
    qasm_mod.circuit_from_qasm_str = circuit_from_qasm_str

    monkeypatch.setitem(__import__("sys").modules, "pytket", pytket_mod)
    monkeypatch.setitem(__import__("sys").modules, "pytket.qasm", qasm_mod)

    circuit = Circuit(3)
    circuit.add(gates.M(2, 0, register_name="m0"))
    parsed, metadata = translate_qibo_to_pytket(circuit)

    assert "OPENQASM" in parsed["parsed"]
    assert metadata.measured_qubits == [2, 0]


def test_translate_qibo_to_pytket_records_register_declaration_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Register names must be recorded in QASM creg declaration order (gate
    order), not sorted — pytket's default bit ordering is lexicographic and
    diverges from it (e.g. m10 < m2)."""

    pytket_mod = types.ModuleType("pytket")
    qasm_mod = types.ModuleType("pytket.qasm")
    qasm_mod.circuit_from_qasm_str = lambda source: {"parsed": source}

    monkeypatch.setitem(__import__("sys").modules, "pytket", pytket_mod)
    monkeypatch.setitem(__import__("sys").modules, "pytket.qasm", qasm_mod)

    circuit = Circuit(2)
    circuit.add(gates.M(0, register_name="zz"))
    circuit.add(gates.M(1, register_name="aa"))

    _, metadata = translate_qibo_to_pytket(circuit)

    assert metadata.measurement_registers == ["zz", "aa"]


def test_translate_qibo_to_pytket_for_helios_strips_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def circuit_from_qasm_str(source: str):
        return {"parsed": source}

    pytket_mod = types.ModuleType("pytket")
    qasm_mod = types.ModuleType("pytket.qasm")
    qasm_mod.circuit_from_qasm_str = circuit_from_qasm_str

    monkeypatch.setitem(__import__("sys").modules, "pytket", pytket_mod)
    monkeypatch.setitem(__import__("sys").modules, "pytket.qasm", qasm_mod)

    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.M(1, 0, register_name="m0"))

    parsed, metadata = translate_qibo_to_pytket_for_helios(circuit)

    assert "OPENQASM" in parsed["parsed"]
    assert "measure" not in parsed["parsed"].lower()
    assert metadata.measured_qubits == [1, 0]


def test_translate_qibo_to_pytket_for_helios_rejects_non_terminal_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytket_mod = types.ModuleType("pytket")
    qasm_mod = types.ModuleType("pytket.qasm")
    qasm_mod.circuit_from_qasm_str = lambda source: {"parsed": source}

    monkeypatch.setitem(__import__("sys").modules, "pytket", pytket_mod)
    monkeypatch.setitem(__import__("sys").modules, "pytket.qasm", qasm_mod)

    circuit = Circuit(2)
    circuit.add(gates.M(0, register_name="m0"))
    circuit.add(gates.H(1))

    with pytest.raises(NexusBackendError, match="terminal"):
        translate_qibo_to_pytket_for_helios(circuit)


def test_prepare_qibo_circuit_binds_parameters() -> None:
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, theta=0.0))
    circuit.add(gates.M(0, register_name="m0"))

    prepared, qasm = prepare_qibo_circuit(circuit, parameters=[1.23])

    assert "1.23" in qasm
    assert "0.0" in circuit.to_qasm()


def test_prepare_qibo_circuit_normalizes_invalid_register_name() -> None:
    circuit = Circuit(1)
    circuit.add(gates.M(0, register_name="1-invalid"))

    prepared, _ = prepare_qibo_circuit(circuit)

    assert prepared.queue[0].register_name == "m0"
    assert circuit.queue[0].register_name == "1-invalid"


from nexus.translation import extract_measurement_qubits


def test_extract_measurement_qubits_defaults_to_all_qubits() -> None:
    assert extract_measurement_qubits(Circuit(2)) == [0, 1]


def test_prepare_qibo_circuit_rewrites_y_basis_measurement_rotations() -> None:
    """qibo expands M(basis=Y) into a Unitary((Y+Z)/sqrt2) + M pair; that
    Unitary is not expressible in OpenQASM 2.0 and must be rewritten to SDG+H
    before export."""
    circuit = Circuit(2)
    circuit.add(gates.H(1))
    circuit.add(gates.M(0, basis=gates.Y))
    circuit.add(gates.M(1))

    prepared, qasm = prepare_qibo_circuit(circuit)

    assert not any(isinstance(g, gates.Unitary) for g in prepared.queue)
    assert "sdg" in qasm.lower()


def test_translate_for_helios_rejects_duplicate_measurement() -> None:
    circuit = Circuit(1)
    circuit.add(gates.M(0))
    circuit.add(gates.M(0))

    with pytest.raises(NexusBackendError, match="measured once"):
        translate_qibo_to_pytket_for_helios(circuit)


def _install_pytket(monkeypatch: pytest.MonkeyPatch, circuit_from_qasm_str) -> None:
    pytket_mod = types.ModuleType("pytket")
    qasm_mod = types.ModuleType("pytket.qasm")
    qasm_mod.circuit_from_qasm_str = circuit_from_qasm_str
    monkeypatch.setitem(__import__("sys").modules, "pytket", pytket_mod)
    monkeypatch.setitem(__import__("sys").modules, "pytket.qasm", qasm_mod)


def test_translate_wraps_pytket_parse_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_parse(source: str):
        raise RuntimeError("unsupported construct")

    _install_pytket(monkeypatch, failing_parse)
    circuit = Circuit(1)
    circuit.add(gates.M(0))

    with pytest.raises(NexusBackendError, match="Failed to parse OpenQASM"):
        translate_qibo_to_pytket(circuit)
    with pytest.raises(NexusBackendError, match="Failed to parse OpenQASM"):
        translate_qibo_to_pytket_for_helios(circuit)



def test_translate_for_helios_wraps_stripped_export_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pytket(monkeypatch, lambda source: {"parsed": source})

    original = Circuit.to_qasm

    def broken_for_stripped(self, *args, **kwargs):
        if not self.measurements:
            raise RuntimeError("cannot serialize")
        return original(self, **kwargs)

    monkeypatch.setattr(Circuit, "to_qasm", broken_for_stripped)

    circuit = Circuit(1)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))

    with pytest.raises(NexusBackendError, match="measurement-free"):
        translate_qibo_to_pytket_for_helios(circuit)
