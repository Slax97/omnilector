# SPDX-License-Identifier: AGPL-3.0-or-later
"""Library logging: adds no handlers and does not touch the loggers of other libraries.

Without configuration, warnings (WARNING) still reach stderr (logging.lastResort); with the application's
logging configured, they come out once, in its format. log_level (e.g. "DEBUG") sets the level of
"omnilector" and, if the application has configured nothing, adds a handler to show it.
"""

from __future__ import annotations

import logging

_LEVELS = {name: getattr(logging, name) for name in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")}


def get_logger(name: str = "omnilector", level: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if level is None:
        return logger
    value = _LEVELS.get(level.upper())
    if value is None:
        raise ValueError(f"unknown log_level: {level!r} (DEBUG, INFO, WARNING, ERROR)")
    logger.setLevel(value)
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False  # if the application configures logging later, messages do not come out twice
    return logger
