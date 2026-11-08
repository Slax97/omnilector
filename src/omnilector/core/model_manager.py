# SPDX-License-Identifier: AGPL-3.0-or-later
"""ModelManager: engine cache of one OCR. Lazy, thread-safe, one engine per (name, script).

With memory_mb, the loaded models do not go over the memory left for them: before loading one, the
least recently used are released (models of other scripts that were tried, EasyOCR after the
consensus...). They are loaded again if needed.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from ..config import OCRConfig
from ..engines import OCREngine, get_engine_class
from ..routing.router import EngineSpec
from ..utils.deps import missing_modules
from ..utils.logging import get_logger
from ..utils.resources import LAYOUT_MB, RUNTIME_MB, release_memory


class ModelManager:
    """One per OCR instance (not a singleton). Engines load their model on the first read, so creating
    an engine is cheap and the global lock does not wait for downloads."""

    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._engines: OrderedDict[EngineSpec, OCREngine] = OrderedDict()  # from least to most recent
        self._runtimes: set[str] = set()  # libraries already loaded (torch is not unloaded)
        self.log = get_logger("omnilector", config.log_level)
        budget = config.models_budget_mb
        if budget is not None and config.layout and missing_modules("rapid_layout", "rapid_table") is None:
            budget -= LAYOUT_MB  # the layout model always stays loaded
        self.budget = budget  # MB for the engines; None = no limit

    def get(self, spec: EngineSpec) -> OCREngine:
        with self._lock:
            engine = self._engines.pop(spec, None)
            if engine is None:
                engine = get_engine_class(spec.name)(self.config, script=spec.script)
            self._engines[spec] = engine
            if not engine.loaded:
                self._make_room(spec, engine)
            return engine

    def resident_mb(self) -> int:
        """Estimated memory of the loaded models and their libraries (MB)."""
        self._runtimes.update(e.info.runtime for e in self._engines.values() if e.loaded and e.info.runtime)
        return sum(e.resident_mb() for e in self._engines.values()) + sum(RUNTIME_MB.get(r, 0) for r in self._runtimes)

    def _make_room(self, spec: EngineSpec, engine: OCREngine) -> None:
        """Releases the least used engines until the one about to be loaded fits."""
        if self.budget is None:
            return
        runtime = engine.info.runtime
        need = engine.info.memory_mb + (RUNTIME_MB.get(runtime, 0) if runtime and runtime not in self._runtimes else 0)
        released = []
        for key in list(self._engines):
            if self.resident_mb() + need <= self.budget:
                break
            if key != spec and self._engines[key].resident_mb():
                released.append(str(key))
                del self._engines[key]
        if released:
            self.log.debug("memory: releasing %s to load %s", ", ".join(released), spec)
            release_memory()

    def release(self) -> None:
        """Frees every model (they are loaded again when used)."""
        with self._lock:
            self._engines.clear()
        release_memory()
