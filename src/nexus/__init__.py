"""nexus — Qibo backend for Quantinuum Nexus (PyPI distribution: ``nexus-backend``).

Discovery contract for ``qibo.set_backend("nexus", ...)``:
Qibo's :func:`qibo.backends.construct_backend` translates the backend string
``"nexus"`` to the import name ``"nexus"`` (the ``-`` → ``_`` substitution is
a no-op here), imports this package, and looks up :class:`MetaBackend` on the
top-level module to call its :meth:`MetaBackend.load` static method.

The package therefore exposes :class:`MetaBackend` here, alongside the public
:class:`NexusClientBackend` class for direct construction. The PyPI
distribution is named ``nexus-backend`` (mirrors the ``Pillow`` / ``PIL``
dist-vs-import asymmetry); install with ``pip install nexus-backend``.
"""

from __future__ import annotations

import importlib.metadata as _im
from typing import Any

from .backend import (
    EstimateItem,
    ExecutionEstimate,
    NexusClientBackend,
    run_compile_execute,
)
from .config import NexusBackendConfig
from .job import NexusJob

# PyPI distribution name (differs from the import package name "nexus").
_DIST_NAME = "nexus-backend"

try:  # pragma: no cover - importlib.metadata behaviour
    __version__ = _im.version(_DIST_NAME)
except _im.PackageNotFoundError:  # pragma: no cover - editable / source checkout
    __version__ = "0.0.0+unknown"

PLATFORMS = ("hseries", "helios", "aer")
"""Platform families understood by :class:`NexusBackendConfig`."""


class MetaBackend:
    """Loader contract Qibo's :func:`construct_backend` looks for.

    ``qibo.set_backend("nexus", platform="hseries:H2-1LE", project=...)``
    eventually calls ``MetaBackend.load(platform=..., project=...)`` and assigns
    the returned backend instance as the active Qibo backend.
    """

    @staticmethod
    def load(platform: str | None = None, **kwargs: Any) -> NexusClientBackend:
        """Instantiate :class:`NexusClientBackend`.

        Args:
            platform: Optional platform string (e.g. ``"hseries:H2-1LE"``,
                ``"helios:Helios-1E"``, ``"aer:..."``). When ``None``, the
                backend's own default (``"hseries:H2-1LE"``) is used.
            **kwargs: Forwarded to :class:`NexusClientBackend` (``project``,
                ``optimisation_level``, ``timeout``, etc.).
        """
        if platform is not None:
            kwargs["platform"] = platform
        return NexusClientBackend(**kwargs)

    def list_available(self) -> dict[str, bool]:
        """Report supported platform families.

        Consumed by ``qibo.backends.list_available_backends("nexus")``.
        """
        return {family: True for family in PLATFORMS}


__all__ = [
    "EstimateItem",
    "ExecutionEstimate",
    "MetaBackend",
    "NexusBackendConfig",
    "NexusClientBackend",
    "NexusJob",
    "PLATFORMS",
    "__version__",
    "run_compile_execute",
]
