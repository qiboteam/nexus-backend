"""Result mapping helpers for the Nexus backend."""

from __future__ import annotations

import logging
from collections import Counter
from itertools import repeat
from typing import Any, Iterable

import numpy as np
from qibo.models import Circuit

from .errors import NexusResultMappingError

LOGGER = logging.getLogger(__name__)


def _bits_from_key(key: Any) -> list[int]:
    if isinstance(key, str):
        return [int(ch) for ch in key.strip() if ch in {"0", "1"}]
    if isinstance(key, Iterable):
        return [int(x) for x in key]
    raise NexusResultMappingError(f"Unsupported count key type: {type(key)}")


def normalize_bitstring(
    *,
    key: Any,
    nbits: int,
    measured_qubits: list[int] | None,
    reverse_endianness: bool,
) -> str:
    """Normalize backend count keys into Qibo-style binary strings."""

    bits = _bits_from_key(key)
    if len(bits) != nbits:
        raise NexusResultMappingError(
            f"Count key width mismatch. Expected {nbits}, received {len(bits)} for key={key!r}."
        )

    if reverse_endianness:
        bits = list(reversed(bits))

    # Measurement targets are already serialized in measurement-register order.
    # Preserve that order when constructing Qibo-facing bitstrings.
    _ = measured_qubits

    return "".join(str(bit) for bit in bits)


def _bit_sort_index(bit: Any) -> int:
    return int(bit.index[0])


def _ordered_cbits(backend_result: Any, register_order: list[str] | None) -> Any:
    """Order pytket classical bits by QASM register declaration order.

    pytket's default get_counts() column order is lexicographic by register
    name ("m10" < "m2"), which diverges from declaration (measurement) order
    once auto-numbered register names reach double digits.  Returns None when
    the result does not expose pytket-style c_bits — callers then keep the
    backend default order.  Registers not covered by register_order (e.g.
    scratch registers added during compilation) are excluded from the
    selection with a warning; only if none of the declared registers survive
    does the ordering fall back entirely.
    """
    if not register_order:
        return None
    c_bits = getattr(backend_result, "c_bits", None)
    if not c_bits:
        return None

    rank = {name: pos for pos, name in enumerate(register_order)}
    groups: dict[str, list[Any]] = {}
    for bit in c_bits:
        groups.setdefault(str(bit.reg_name), []).append(bit)

    known = [name for name in groups if name in rank]
    unknown = [name for name in groups if name not in rank]
    if not known:
        LOGGER.warning(
            "None of the result's classical registers %s match the declared "
            "measurement registers %s; falling back to the backend's default "
            "bit order, which may not follow measurement order.",
            sorted(groups),
            register_order,
        )
        return None
    if unknown:
        LOGGER.warning(
            "Excluding unexpected classical registers %s from counts; keeping "
            "declared registers %s in declaration order.",
            sorted(unknown),
            register_order,
        )

    ordered: list[Any] = []
    for name in sorted(known, key=lambda n: rank[n]):
        ordered.extend(sorted(groups[name], key=_bit_sort_index))
    return ordered


def _extract_counts(
    backend_result: Any, register_order: list[str] | None = None
) -> dict[Any, int]:
    cbits = _ordered_cbits(backend_result, register_order)
    if cbits is not None:
        return backend_result.get_counts(cbits=cbits)
    return backend_result.get_counts()


def map_counts_to_qibo_frequencies(
    counts: dict[Any, int],
    *,
    measured_qubits: list[int],
    reverse_endianness: bool = False,
) -> Counter[str]:
    """Map backend counts dictionary to Qibo-compatible binary frequencies."""

    nbits = len(measured_qubits) if measured_qubits else 0
    if nbits == 0 and counts:
        first_key = next(iter(counts))
        nbits = len(_bits_from_key(first_key))

    frequencies: Counter[str] = Counter()
    for key, value in counts.items():
        bitstring = normalize_bitstring(
            key=key,
            nbits=nbits,
            measured_qubits=measured_qubits,
            reverse_endianness=reverse_endianness,
        )
        frequencies[bitstring] += int(value)

    return frequencies


def map_nexus_result_to_qibo(
    *,
    execution_result_ref: Any,
    circuit: Circuit,
    backend: Any,
    nshots: int,
    measured_qubits: list[int],
    reverse_endianness: bool = False,
    register_order: list[str] | None = None,
) -> Any:
    """Download and convert a Nexus execution result to a Qibo result object."""

    backend_result = execution_result_ref.download_result()
    counts = _extract_counts(backend_result, register_order)
    frequencies = map_counts_to_qibo_frequencies(
        counts,
        measured_qubits=measured_qubits,
        reverse_endianness=reverse_endianness,
    )

    try:
        from qibo.result import MeasurementOutcomes
    except Exception as exc:  # pragma: no cover - import environment specific
        raise NexusResultMappingError(
            "qibo is required to build result objects."
        ) from exc

    measurements = list(circuit.measurements)
    total_shots = int(sum(frequencies.values()))
    effective_nshots = total_shots if total_shots > 0 else int(nshots)

    samples = []
    for bitstring, count in frequencies.items():
        sample = [int(b) for b in bitstring]
        samples.extend(repeat(sample, count))

    return MeasurementOutcomes(
        measurements,
        backend=backend,
        nshots=effective_nshots,
        samples=np.array(samples, dtype=int),
    )


__all__ = [
    "normalize_bitstring",
    "map_counts_to_qibo_frequencies",
    "map_nexus_result_to_qibo",
]
