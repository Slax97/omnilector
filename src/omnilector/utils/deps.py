# SPDX-License-Identifier: AGPL-3.0-or-later
"""Optional dependencies: check them without importing them, and import them with a clear error."""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

from ..exceptions import DependencyMissingError


def missing_modules(*modules: str, extra: str = "") -> str | None:
    """Readable reason if a module is missing; None if all are there. Does not import them (cheap)."""
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if not missing:
        return None
    hint = f' -> pip install "omnilector[{extra}]"' if extra else ""
    return f"missing {', '.join(missing)}{hint}"


def require(module: str, package: str | None = None, extra: str = "") -> ModuleType:
    """Imports an optional module or raises DependencyMissingError naming the extra that installs it."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise DependencyMissingError(package or module, extra) from e
