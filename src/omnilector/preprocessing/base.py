# SPDX-License-Identifier: AGPL-3.0-or-later
"""Preprocessing base."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Preprocessor(Protocol):
    name: str

    def run(self, image: np.ndarray) -> np.ndarray: ...
