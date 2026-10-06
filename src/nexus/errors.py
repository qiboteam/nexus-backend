"""Custom exceptions raised by the Nexus backend."""

from __future__ import annotations


class NexusBackendError(RuntimeError):
    """Base class for Nexus backend errors."""


class NexusAuthError(NexusBackendError):
    """Raised when Nexus authentication/context setup fails."""


class NexusResultMappingError(NexusBackendError):
    """Raised when Nexus result payload cannot be converted to Qibo results."""


class UnsupportedExecutionError(NexusBackendError):
    """Raised when user asks for unsupported execution modes."""


class PartialSubmissionError(NexusBackendError):
    """Raised when a multi-job submission fails after some jobs were queued.

    The already-submitted execute jobs are left running; their ids are
    available in ``submitted_job_ids`` so they can be reattached with
    ``get_job`` or cancelled.
    """

    def __init__(self, message: str, submitted_job_ids: tuple[str, ...]) -> None:
        super().__init__(message)
        self.submitted_job_ids = tuple(submitted_job_ids)
