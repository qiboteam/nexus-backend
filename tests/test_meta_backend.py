"""MetaBackend / set_backend integration tests for nexus-backend.

These exercise Qibo's lazy backend-discovery contract:
``qibo.set_backend("nexus-backend", ...)`` must import this package, locate
``MetaBackend.load(**kwargs)``, and yield a :class:`NexusClientBackend`.
"""

from qibo.backends import get_backend, set_backend

import nexus_backend.backend as nexus_mod
from nexus_backend import MetaBackend, NexusClientBackend


def test_meta_backend_load_nexus(monkeypatch):
    monkeypatch.setattr(nexus_mod, "_ensure_nexus_dependencies", lambda: None)
    backend = MetaBackend.load(platform="hseries:H2-1LE")
    assert isinstance(backend, NexusClientBackend)
    assert backend.name == "nexus-backend"


def test_meta_backend_load_nexus_uses_default_platform(monkeypatch):
    """Omitting `platform=` must fall through to NexusClientBackend's default
    (`hseries:H2-1LE`) rather than passing platform=None and breaking parse_platform."""
    monkeypatch.setattr(nexus_mod, "_ensure_nexus_dependencies", lambda: None)
    backend = MetaBackend.load()
    assert isinstance(backend, NexusClientBackend)
    assert backend.config.platform == "hseries:H2-1LE"


def test_set_backend_nexus(monkeypatch):
    monkeypatch.setattr(nexus_mod, "_ensure_nexus_dependencies", lambda: None)
    set_backend("nexus-backend", platform="hseries:H2-1LE")
    assert isinstance(get_backend(), NexusClientBackend)
    assert get_backend().name == "nexus-backend"
