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

## Non-blocking execution

Hardware queues can hold a job for hours. Instead of blocking, submit and
get a `NexusJob` handle back — modeled on `concurrent.futures.Future`:

```python
backend = qibo.get_backend()          # after qibo.set_backend("nexus", ...)

job = backend.submit_circuit(circuit, nshots=1000)   # returns after submission
job.job_id        # Nexus execute-job id (also visible in the Nexus web UI)
job.status()      # qnexus JobStatus (QUEUED / RUNNING / COMPLETED / ...)
job.done()        # True once the job stopped (successfully or not)
job.result(timeout=60)   # wait up to 60 s; TimeoutError if still queued
job.result()             # wait as long as it takes; result is cached
job.cancel()
```

`result(timeout=...)` bounds only the waiting: on expiry it raises
`TimeoutError`, the remote job keeps running, and the handle stays usable.
Job failures raise `NexusBackendError` instead.

The same handle is available through two other spellings:

```python
job = backend.execute_circuit(circuit, nshots=1000, blocking=False)

# Backend-level default — makes circuit(...) return handles too, since
# Qibo does not forward per-call kwargs to the backend:
qibo.set_backend("nexus", platform="hseries:H2-1LE", blocking=False)
job = circuit(nshots=1000)
```

Batches work the same way: `backend.submit_circuits([c1, c2], nshots=[10, 20])`
returns one handle whose `result()` is a list in submission order.

To pick a job up from a new Python process, re-supply the circuit(s) so the
measurement-mapping metadata can be re-derived locally (nothing is
re-uploaded):

```python
job = backend.get_job("<job-id>", circuit, nshots=1000)
result = job.result()
```

On non-Helios targets, submission blocks through the (fast) remote compile
stage and returns once the execute job is queued. On Helios, submission
includes cost estimation unless `max_cost` is set on the backend.

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
