"""Configuration models and Nexus backend-config construction."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)

_SUPPORTED_PLATFORM_FAMILIES = {"hseries", "helios", "aer"}


@dataclass
class NexusBackendConfig:
    """Runtime configuration for the Qibo <-> Nexus integration backend."""

    platform: str = "hseries:H2-1LE"
    project: str | None = None
    optimisation_level: int = 2
    timeout: float = 1800.0
    allow_incomplete: bool = False
    max_cost: float | None = None
    max_cost_factor: float = 1.2
    language: Any = None
    credential_login: bool | None = None
    batch_mode: bool = True
    reverse_endianness: bool = False
    backend_options: dict[str, Any] = field(default_factory=dict)
    job_name_prefix: str = "qibo-nexus"

    def __post_init__(self) -> None:
        # A max_cost of 0.0 submitted to Helios instantly depletes the job, so
        # neither an explicit non-positive max_cost nor a factor that could
        # produce one may survive construction.  `not (x > 0)` also catches NaN.
        if self.max_cost is not None:
            self.max_cost = float(self.max_cost)
            if not self.max_cost > 0:
                raise ValueError(
                    f"max_cost must be a positive number of HQCs, got {self.max_cost!r}."
                )
        self.max_cost_factor = float(self.max_cost_factor)
        if not self.max_cost_factor > 0:
            raise ValueError(
                f"max_cost_factor must be positive, got {self.max_cost_factor!r}."
            )

    @property
    def platform_family(self) -> str:
        return parse_platform(self.platform)[0]

    @property
    def platform_name(self) -> str:
        return parse_platform(self.platform)[1]

    @property
    def shot_only(self) -> bool:
        return self.platform_family in {"hseries", "helios", "aer"}


def parse_platform(platform: str) -> tuple[str, str]:
    """Parse platform string in the form '<family>:<name>'."""

    if ":" not in platform:
        LOGGER.warning('Platform string missing family prefix. Assuming "hseries".')
        return "hseries", platform

    family, raw_name = platform.split(":", 1)
    family = family.strip().lower()
    name = raw_name.strip()
    if not family or not name:
        raise ValueError(f"Invalid platform '{platform}'. Expected '<family>:<name>'.")
    if family not in _SUPPORTED_PLATFORM_FAMILIES:
        expected = ", ".join(sorted(_SUPPORTED_PLATFORM_FAMILIES))
        raise ValueError(
            f"Unsupported platform family '{family}'. Expected one of: {expected}."
        )
    return family, name


def _should_use_helios_emulator(name: str, forced: Any) -> bool:
    if forced is not None:
        return bool(forced)
    lowered = name.lower()
    return "emulator" in lowered or lowered.endswith("-1e")


def helios_emulator_requested(cfg: NexusBackendConfig) -> bool:
    """Whether the configured Helios target is (or is forced to be) an emulator."""

    family, name = parse_platform(cfg.platform)
    if family != "helios":
        return False
    return _should_use_helios_emulator(name, cfg.backend_options.get("emulator"))


def _supports_parameter(model: Any, parameter: str) -> bool:
    return parameter in inspect.signature(model).parameters


def _build_helios_backend_config(*, name: str, options: dict[str, Any]) -> Any:
    from qnexus.models import HeliosConfig, HeliosEmulatorConfig

    forced_emulator = options.pop("emulator", None)
    if not _should_use_helios_emulator(name, forced_emulator):
        return HeliosConfig(system_name=name, **options)

    # Split options: keys accepted by HeliosEmulatorConfig go on the emulator;
    # any remaining user-supplied options (e.g. attempt_batching for real
    # Helios-1) belong on HeliosConfig.  Emulator sizing is not injected here:
    # n_qubits rides the per-item kwarg on qnx.start_execute_job unless the
    # user set it explicitly in backend_options.
    emulator_options = {
        k: v for k, v in options.items() if _supports_parameter(HeliosEmulatorConfig, k)
    }
    helios_options = {k: v for k, v in options.items() if k not in emulator_options}
    return HeliosConfig(
        system_name=name,
        emulator_config=HeliosEmulatorConfig(**emulator_options),
        **helios_options,
    )


def build_nexus_backend_config(cfg: NexusBackendConfig) -> Any:
    """Build a concrete qnexus backend config object for compile/execute jobs."""

    try:
        import qnexus as qnx
    except Exception as exc:  # pragma: no cover - import environment specific
        raise ImportError("qnexus is required to build Nexus backend configs.") from exc

    family, name = parse_platform(cfg.platform)
    options = dict(cfg.backend_options)

    if family == "aer":
        return qnx.AerConfig(**options)

    if family == "hseries":
        options.pop("emulator", None)
        return qnx.QuantinuumConfig(device_name=name, **options)

    return _build_helios_backend_config(name=name, options=options)
