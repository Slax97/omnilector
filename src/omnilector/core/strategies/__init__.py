# SPDX-License-Identifier: AGPL-3.0-or-later
"""omnilector strategies."""

from .base import OCRStrategy
from .strategies import (
    AccurateStrategy,
    BalancedStrategy,
    FastStrategy,
    MaximumRecallStrategy,
    get_strategy,
)

__all__ = [
    "AccurateStrategy",
    "BalancedStrategy",
    "FastStrategy",
    "MaximumRecallStrategy",
    "OCRStrategy",
    "get_strategy",
]
