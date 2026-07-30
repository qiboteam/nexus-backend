from __future__ import annotations

import sys
import types

import pytest

from nexus.auth import authenticate, ensure_project
from nexus.errors import NexusAuthError


def _install_fake_qnexus(monkeypatch: pytest.MonkeyPatch, **attrs) -> types.ModuleType:
    fake_qnx = types.ModuleType("qnexus")
    for name, value in attrs.items():
        setattr(fake_qnx, name, value)
    monkeypatch.setitem(sys.modules, "qnexus", fake_qnx)
    return fake_qnx


def test_authenticate_credential_login_calls_qnexus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _install_fake_qnexus(
        monkeypatch, login_with_credentials=lambda: calls.append("credentials")
    )

    authenticate(credential_login=True)

    assert calls == ["credentials"]


def test_authenticate_headless_default_does_not_force_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Token-based default must not trigger any qnexus login call — forcing
    interactive login would hang automated environments."""

    def fail() -> None:
        raise AssertionError("login must not be called")

    _install_fake_qnexus(
        monkeypatch, login_with_credentials=fail, login=fail
    )

    assert authenticate(credential_login=None) is None
    assert authenticate(credential_login=False) is None


def test_authenticate_wraps_login_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise RuntimeError("bad credentials")

    _install_fake_qnexus(monkeypatch, login_with_credentials=boom)

    with pytest.raises(NexusAuthError, match="bad credentials"):
        authenticate(credential_login=True)


def test_ensure_project_none_skips_qnexus() -> None:
    assert ensure_project(None) is None


def test_ensure_project_uses_keyword_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    _install_fake_qnexus(
        monkeypatch,
        projects=types.SimpleNamespace(
            get_or_create=lambda *, name: calls.append({"name": name})
            or f"project-ref-{name}"
        ),
    )

    assert ensure_project("my-project") == "project-ref-my-project"
    assert calls == [{"name": "my-project"}]


def test_ensure_project_wraps_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kwargs) -> None:
        raise RuntimeError("nexus is down")

    _install_fake_qnexus(
        monkeypatch, projects=types.SimpleNamespace(get_or_create=boom)
    )

    with pytest.raises(NexusAuthError, match="my-project"):
        ensure_project("my-project")
