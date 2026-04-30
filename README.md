# nexus-backend

A [Qibo](https://qibo.science) backend for compiling and executing circuits through [Quantinuum Nexus](https://nexus.quantinuum.com).

## Installation

While the package is not yet published to PyPI, install directly from the repo:

```bash
pip install git+https://github.com/qiboteam/nexus-backend
```

For local development:

```bash
git clone https://github.com/qiboteam/nexus-backend
cd nexus-backend
poetry install --with tests
```

## Quickstart

```python
import qibo

# Tell Qibo to route circuits through Nexus.  The string "nexus" is
# mapped to the importable package `nexus` and its top-level
# MetaBackend.load(**kwargs) is invoked under the hood.
qibo.set_backend("nexus", platform="hseries:H2-1LE")

circuit = qibo.models.QFT(5)
circuit.add(qibo.gates.M(0, 2, 4))

result = circuit(nshots=1000)
print(result.frequencies())
```

## Authentication

Importing `nexus` does not contact Quantinuum Nexus. Authentication and project
resolution happen lazily on the first `execute_*` or `estimate_*` call. To pin
the project explicitly, pass `project="my-project"` to `qibo.set_backend(...)`;
otherwise the backend resolves it from your Nexus account on first use.

## Supported platforms

Configure the desired target with the `platform` argument, e.g.
`"hseries:H2-1LE"` or `"helios:Helios-1E"`.

- `hseries:<device-name>` — Quantinuum H-Series (e.g. `hseries:H2-1LE`)
- `helios:<system-name>` — Helios via HUGR (e.g. `helios:Helios-1E`)
- `aer:<simulator-name>` — Qiskit Aer through Nexus

## License

Apache-2.0.
