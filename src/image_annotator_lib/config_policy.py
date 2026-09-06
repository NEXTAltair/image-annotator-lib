"""Public configuration I/O policy, usable before importing the package.

Set IMAGE_ANNOTATOR_CONFIG_READ_ONLY=1 before public import to load existing
configuration without automatic creation or persistence. Diagnostic logs are separate.
"""

import os
from pathlib import Path

CONFIG_READ_ONLY_ENV = "IMAGE_ANNOTATOR_CONFIG_READ_ONLY"


class ReadOnlyConfigError(RuntimeError):
    """Required model configuration cannot be loaded or persisted in read-only mode."""

    def __init__(self, path: Path | None, action: str) -> None:
        super().__init__(f"Read-only model configuration requires explicit preparation ({action}): {path}")
        self.details = {"config_path": str(path) if path is not None else None, "action": action}


def config_read_only_enabled() -> bool:
    """Return whether the supported import/persistence policy is enabled."""
    return os.environ.get(CONFIG_READ_ONLY_ENV, "").lower() in {"1", "true", "yes"}


def require_config_write_permission(path: Path | None) -> None:
    if config_read_only_enabled():
        raise ReadOnlyConfigError(path, "write_forbidden")
