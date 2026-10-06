from __future__ import annotations

import types

import pytest

from nexus.config import (
    NexusBackendConfig,
    _should_use_helios_emulator,
    build_nexus_backend_config,
    helios_emulator_requested,
    parse_platform,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Helios-1SC", False),  # syntax checker must never count as emulator
        ("Helios-1E", True),
        ("helios-emulator", True),
        ("Helios-1", False),  # hardware
    ],
)
def test_helios_emulator_detection(name: str, expected: bool) -> None:
    assert _should_use_helios_emulator(name, None) is expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_cost": 0.0},
        {"max_cost": -1.0},
        {"max_cost": float("nan")},
        {"max_cost_factor": 0.0},
        {"max_cost_factor": -0.5},
    ],
)
def test_config_rejects_non_positive_cost_options(kwargs: dict) -> None:
    """max_cost=0.0 submitted to Helios instantly depletes the job; a zero or
    negative max_cost_factor turns a valid estimate into the same thing."""
    with pytest.raises(ValueError, match="max_cost"):
        NexusBackendConfig(**kwargs)


def test_parse_platform_defaults_to_hseries_when_missing_family() -> None:
    assert parse_platform("H2-1LE") == ("hseries", "H2-1LE")


def test_parse_platform_accepts_aer_family() -> None:
    assert parse_platform("aer:aer_simulator") == ("aer", "aer_simulator")


def test_shot_only_includes_aer_family() -> None:
    cfg = NexusBackendConfig(platform="aer:aer_simulator")
    assert cfg.shot_only is True


def test_parse_platform_rejects_unknown_family() -> None:
    with pytest.raises(ValueError):
        parse_platform("foo:bar")


def test_build_hseries_config(monkeypatch: pytest.MonkeyPatch) -> None:
    class QuantinuumConfig:
        def __init__(self, *, device_name: str, **kwargs):
            self.device_name = device_name
            self.kwargs = kwargs

    fake_qnx = types.SimpleNamespace(QuantinuumConfig=QuantinuumConfig)
    monkeypatch.setitem(__import__("sys").modules, "qnexus", fake_qnx)

    cfg = NexusBackendConfig(platform="hseries:H2-1LE")
    concrete = build_nexus_backend_config(cfg)
    assert concrete.device_name == "H2-1LE"


def _install_helios_models(
    monkeypatch: pytest.MonkeyPatch, *, helios_config, helios_emulator_config
) -> None:
    models_mod = types.ModuleType("qnexus.models")
    models_mod.HeliosConfig = helios_config
    models_mod.HeliosEmulatorConfig = helios_emulator_config
    qnx_mod = types.ModuleType("qnexus")
    qnx_mod.models = models_mod
    sys_modules = __import__("sys").modules
    monkeypatch.setitem(sys_modules, "qnexus", qnx_mod)
    monkeypatch.setitem(sys_modules, "qnexus.models", models_mod)


class _HeliosEmulatorConfig:
    def __init__(self, *, simulator: str = "statevector"):
        self.simulator = simulator


class _HeliosConfig:
    def __init__(self, *, system_name: str, emulator_config=None):
        self.system_name = system_name
        self.emulator_config = emulator_config


def test_build_helios_hardware_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_helios_models(
        monkeypatch,
        helios_config=_HeliosConfig,
        helios_emulator_config=_HeliosEmulatorConfig,
    )

    cfg = NexusBackendConfig(platform="helios:Helios-1")
    concrete = build_nexus_backend_config(cfg)

    assert concrete.system_name == "Helios-1"
    assert concrete.emulator_config is None


def test_build_helios_config_forced_emulator(monkeypatch: pytest.MonkeyPatch) -> None:
    """emulator=True in backend_options must attach an emulator config even for
    a hardware-named target."""
    _install_helios_models(
        monkeypatch,
        helios_config=_HeliosConfig,
        helios_emulator_config=_HeliosEmulatorConfig,
    )

    cfg = NexusBackendConfig(
        platform="helios:Helios-1", backend_options={"emulator": True}
    )
    concrete = build_nexus_backend_config(cfg)

    assert concrete.system_name == "Helios-1"
    assert isinstance(concrete.emulator_config, _HeliosEmulatorConfig)


def test_build_helios_modern_config_uses_emulator_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emulator sizing and per-program max_cost are now passed via
    qnx.start_execute_job (n_qubits/max_cost per-item kwargs), so neither is
    auto-injected into the deprecated HeliosEmulatorConfig fields."""

    _install_helios_models(
        monkeypatch,
        helios_config=_HeliosConfig,
        helios_emulator_config=_HeliosEmulatorConfig,
    )

    cfg = NexusBackendConfig(
        platform="helios:Helios-1E",
        backend_options={"simulator": "statevector", "emulator": True},
    )
    concrete = build_nexus_backend_config(cfg)
    assert concrete.system_name == "Helios-1E"
    assert concrete.emulator_config.simulator == "statevector"
    assert not hasattr(concrete.emulator_config, "n_qubits")


def test_helios_emulator_requested_reflects_platform_and_force_flag() -> None:
    assert helios_emulator_requested(NexusBackendConfig(platform="helios:Helios-1E"))
    assert not helios_emulator_requested(NexusBackendConfig(platform="helios:Helios-1"))
    assert helios_emulator_requested(
        NexusBackendConfig(
            platform="helios:Helios-1", backend_options={"emulator": True}
        )
    )
    assert not helios_emulator_requested(NexusBackendConfig(platform="hseries:H2-1LE"))


def test_build_helios_emulator_config_ignores_helios_config_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """attempt_batching/max_batch_cost must go to HeliosConfig, not HeliosEmulatorConfig."""

    class HeliosEmulatorConfig:
        def __init__(self, *, n_qubits: int):
            self.n_qubits = n_qubits
            # does NOT accept attempt_batching or max_batch_cost

    class HeliosConfig:
        def __init__(
            self,
            *,
            system_name: str,
            emulator_config=None,
            attempt_batching=False,
            max_batch_cost=2000.0,
            **kwargs,
        ):
            self.system_name = system_name
            self.emulator_config = emulator_config
            self.attempt_batching = attempt_batching
            self.max_batch_cost = max_batch_cost

    _install_helios_models(
        monkeypatch,
        helios_config=HeliosConfig,
        helios_emulator_config=HeliosEmulatorConfig,
    )

    cfg = NexusBackendConfig(
        platform="helios:Helios-1E",
        backend_options={
            "emulator": True,
            "n_qubits": 8,
            "attempt_batching": True,
            "max_batch_cost": 99.0,
        },
    )
    concrete = build_nexus_backend_config(cfg)
    assert concrete.attempt_batching is True
    assert concrete.max_batch_cost == 99.0
    assert concrete.emulator_config.n_qubits == 8


def test_build_aer_config(monkeypatch: pytest.MonkeyPatch) -> None:
    class AerConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_qnx = types.SimpleNamespace(
        AerConfig=AerConfig,
        QuantinuumConfig=object,
        HeliosConfig=object,
        HeliosEmulatorConfig=object,
    )
    monkeypatch.setitem(__import__("sys").modules, "qnexus", fake_qnx)

    cfg = NexusBackendConfig(
        platform="aer:aer_simulator",
        backend_options={"seed_simulator": 11, "method": "statevector"},
    )
    concrete = build_nexus_backend_config(cfg)
    assert concrete.kwargs["seed_simulator"] == 11
    assert concrete.kwargs["method"] == "statevector"


def test_platform_name_property() -> None:
    assert NexusBackendConfig(platform="hseries:H2-1LE").platform_name == "H2-1LE"


def test_parse_platform_rejects_blank_family_or_name() -> None:
    with pytest.raises(ValueError, match="Expected '<family>:<name>'"):
        parse_platform("helios: ")
    with pytest.raises(ValueError, match="Expected '<family>:<name>'"):
        parse_platform(" :Helios-1")


def test_blocking_defaults_true_and_is_configurable() -> None:
    from nexus.config import NexusBackendConfig

    assert NexusBackendConfig().blocking is True
    assert NexusBackendConfig(blocking=False).blocking is False
