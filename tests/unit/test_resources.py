# SPDX-License-Identifier: AGPL-3.0-or-later
"""Any server: cores and memory of the process (container, systemd, machine) and how the OCR adapts
to them; what is set explicitly always wins."""

from __future__ import annotations

import os

import pytest

from omnilector import OCR
from omnilector.config import OCRConfig
from omnilector.core.model_manager import ModelManager
from omnilector.core.strategies import get_strategy
from omnilector.engines import RapidOCREngine
from omnilector.routing import EngineSpec, PageProfile, Router
from omnilector.utils import resources
from omnilector.utils.resources import (
    available_cpus,
    available_memory_mb,
    cgroup_cpus,
    cgroup_memory_mb,
    cpu_limit,
    image_side_for,
    parse_memory,
)

from .conftest import make_engine, reader, tb, white

GB = 2**30


def cgroup(tmp_path, files: dict[str, str], proc: str = "0::/\n"):
    for name, value in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(value)
    (tmp_path / "proc").write_text(proc)
    return tmp_path, tmp_path / "proc"


def test_container_limits_cgroup_v2(tmp_path):
    root, proc = cgroup(
        tmp_path, {"cpu.max": "200000 100000\n", "memory.max": f"{2 * GB}\n"}
    )  # a quota of 2 cores and 2 GB
    assert cgroup_cpus(root, proc) == 2.0 and cgroup_memory_mb(root, proc) == 2048
    root, proc = cgroup(tmp_path, {"cpu.max": "max 100000\n", "memory.max": "max\n"})
    assert cgroup_cpus(root, proc) is None and cgroup_memory_mb(root, proc) is None


def test_service_limit_in_a_parent_cgroup(tmp_path):
    files = {
        "system.slice/app.service/memory.max": "max\n",
        "system.slice/memory.max": f"{GB}\n",
        "system.slice/app.service/cpu.max": "150000 100000\n",
    }
    root, proc = cgroup(tmp_path, files, proc="0::/system.slice/app.service\n")
    assert cgroup_memory_mb(root, proc) == 1024 and cgroup_cpus(root, proc) == 1.5


def test_container_limits_cgroup_v1(tmp_path):
    files = {
        "cpu/cpu.cfs_quota_us": "150000\n",
        "cpu/cpu.cfs_period_us": "100000\n",
        "memory/memory.limit_in_bytes": f"{GB}\n",
    }
    root, proc = cgroup(tmp_path, files, proc="")
    assert cgroup_cpus(root, proc) == 1.5 and cgroup_memory_mb(root, proc) == 1024
    (tmp_path / "memory/memory.limit_in_bytes").write_text("9223372036854771712\n")  # v1 without a limit
    (tmp_path / "cpu/cpu.cfs_quota_us").write_text("-1\n")
    assert cgroup_memory_mb(root, proc) is None and cgroup_cpus(root, proc) is None


def test_cpus_follow_the_quota(monkeypatch):
    monkeypatch.setattr(os, "process_cpu_count", lambda: 16, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    monkeypatch.setattr(resources, "cgroup_cpus", lambda: 1.5)
    assert available_cpus() == 2 and cpu_limit() == 2  # 1.5 cores -> 2 threads per engine
    monkeypatch.setattr(resources, "cgroup_cpus", lambda: None)
    assert available_cpus() == 16 and cpu_limit() is None  # the whole machine: each engine uses its own


@pytest.mark.parametrize(
    ("text", "mb"), [("2g", 2048), ("1.5G", 1536), ("2048m", 2048), ("2048", 2048), ("512MB", 512), ("2GiB", 2048)]
)
def test_parse_memory(text, mb):
    assert parse_memory(text) == mb


def test_parse_memory_rejects_nonsense():
    with pytest.raises(ValueError):
        parse_memory("two gigs")


def test_image_side_by_memory():
    assert image_side_for(2048) == 2000 and image_side_for(1024) == 1400 and image_side_for(4096) == 3000
    assert image_side_for(32768) > 4000 and image_side_for(256) == 1000


def test_config_adapts_to_the_memory():
    two = OCRConfig(memory_mb=2048)
    assert (two.max_image_side, two.layout, two.models_budget_mb) == (2000, True, 1138)
    assert OCRConfig(memory_mb=8192).max_image_side == 4000  # the strategy's, which fits
    small = OCRConfig(memory_mb=1024)
    assert small.max_image_side == 1400 and small.layout is False  # the layout model does not fit
    assert OCRConfig().max_image_side == 4000 and OCRConfig().memory_mb is None  # without a limit, as always


def test_manual_settings_win():
    cfg = OCRConfig(memory_mb=1024, max_image_side=4000, layout=True, threads=3)
    assert (cfg.max_image_side, cfg.layout, cfg.threads) == (4000, True, 3)
    with pytest.raises(ValueError):
        OCRConfig(memory_mb=100)


def test_environment_and_detection(monkeypatch):
    monkeypatch.setattr(resources, "available_memory_mb", lambda: 1536)
    monkeypatch.setattr(resources, "cpu_limit", lambda: 2)
    assert (OCRConfig().memory_mb, OCRConfig().threads) == (1536, 2)  # the container's
    monkeypatch.setenv("OMNILECTOR_MEMORY_MB", "2g")
    monkeypatch.setenv("OMNILECTOR_THREADS", "4")
    assert (OCRConfig().memory_mb, OCRConfig().threads) == (2048, 4)  # the variable wins over what is detected
    assert OCRConfig(memory_mb=3000, threads=1).memory_mb == 3000  # and explicit settings over the variable


def engines_for_memory():
    page = make_engine("rapidocr", memory_mb=150, mb_per_megapixel=150)
    photo = make_engine("easyocr", cost=3, memory_mb=350, runtime="torch", mb_per_megapixel=770)
    tess = make_engine("tesseract", cost=2, memory_mb=20, mb_per_megapixel=15)
    trocr = make_engine("trocr", level="line", handwriting=True, cost=3, memory_mb=1600, runtime="torch")
    trocr.info = type(trocr.info)(**{**trocr.info.__dict__, "languages": frozenset({"en"})})
    kraken = make_engine("kraken", level="line", handwriting=True, cost=2, memory_mb=380, runtime="torch")
    return page, photo, tess, trocr, kraken


def test_engines_that_do_not_fit_are_not_used(use_engines):
    use_engines(*engines_for_memory())
    profile = PageProfile(script="latin")

    def plan(memory_mb, **options):
        return Router(OCRConfig(memory_mb=memory_mb, layout=False, **options), get_strategy("maximum_recall"))

    two = plan(2048)  # 1138 MB for models: kraken (580) fits; TrOCR (1800) does not
    assert two.line_engine(profile, "en") == "kraken"
    assert two.side("easyocr") == 1000 and not two.fits("easyocr", "rapidocr")  # its detector does not fit at 1500 px
    assert two.fits("tesseract", "rapidocr") and two.plan(profile).ensemble is True  # consensus only with Tesseract
    assert plan(None).line_engine(profile, "en") == "trocr"  # without a limit, the English specialist
    assert plan(16384).side("easyocr") > 2560  # with plenty of memory, EasyOCR at its own size (2560)
    found = plan(1024).plan(profile)  # 270 MB free: no handwriting
    assert found.line_engine is None and any("no consensus with easyocr" in r for r in found.reasons)
    forced = plan(1024, engine="easyocr").plan(profile)
    assert [e.name for e in forced.engines] == ["easyocr"]  # forced: used even if it does not fit


def test_least_used_models_are_released(use_engines):
    loads: list[str] = []

    def engine(name, memory_mb):
        cls = make_engine(name, memory_mb=memory_mb)
        cls._build = lambda self: loads.append(self.info.name) or True
        return cls

    use_engines(engine("a", 100), engine("b", 150), engine("c", 200))
    cfg = OCRConfig(memory_mb=760, max_image_side=1000, layout=False)  # 760 - 250 - 210 = 300 MB for models
    manager = ModelManager(cfg)
    assert manager.budget == 300
    for name in ("a", "b"):
        manager.get(EngineSpec(name)).load()
    manager.get(EngineSpec("a"))  # "a" was just used: "b" is the least used
    manager.get(EngineSpec("c")).load()  # 250 + 200 > 300: "b" is released, not "a"
    assert [s.name for s in manager._engines] == ["a", "c"] and manager.resident_mb() == 300
    manager.get(EngineSpec("b")).load()  # loaded again when needed
    assert loads == ["a", "b", "c", "b"]


def test_torch_counts_once_and_stays(use_engines):
    use_engines(make_engine("rapidocr", memory_mb=100), make_engine("k", memory_mb=250, runtime="torch"))
    manager = ModelManager(OCRConfig(memory_mb=4096, layout=False))
    manager.get(EngineSpec("k")).load()
    assert manager.resident_mb() == 250 + resources.RUNTIME_MB["torch"]
    del manager._engines[EngineSpec("k")]
    assert manager.resident_mb() == resources.RUNTIME_MB["torch"]  # torch is not unloaded


def test_script_probe_loads_only_the_recognizer(monkeypatch):
    engine = RapidOCREngine(OCRConfig(), "cyrillic")
    monkeypatch.setattr(RapidOCREngine, "_build", lambda self: pytest.fail("the whole engine is not needed"))

    class Rec:
        def __call__(self, inputs):
            return type("Out", (), {"txts": ["Москва"] * len(inputs.img), "scores": [0.9] * len(inputs.img)})()

    monkeypatch.setattr(RapidOCREngine, "_build_recognizer", lambda self: Rec())
    found = engine.read_lines(white(200, 60), [(10, 10, 190, 50)])
    assert found[0].text == "Москва" and not engine.loaded and engine.resident_mb() == 50


def test_threads_reach_layout_and_tesseract(monkeypatch):
    from omnilector.core.layout import LayoutAnalyzer
    from omnilector.engines.tesseract import TesseractEngine

    seen = {}

    class Mod:
        class ModelType:
            PP_DOC_LAYOUTV3 = SLANETPLUS = "m"

        def __init__(self, kind):
            self.kind = kind

        def RapidLayoutInput(self, **kw):
            seen["layout"] = kw["engine_cfg"]

        RapidTableInput = RapidLayoutInput

        def RapidLayout(self, cfg):
            return "layout"

        RapidTable = RapidLayout

    analyzer = LayoutAnalyzer(OCRConfig(threads=2))
    monkeypatch.setattr("omnilector.core.layout._model_file", lambda *a: None)
    analyzer._create(Mod("l"), Mod("t"), None, None)
    assert seen["layout"] == {"intra_op_num_threads": 2, "inter_op_num_threads": 1}

    pytest.importorskip("pytesseract")
    monkeypatch.delenv("OMP_THREAD_LIMIT", raising=False)
    monkeypatch.setattr("omnilector.engines.tesseract.find_tesseract", lambda: None)
    TesseractEngine(OCRConfig(threads=2))._build()
    assert os.environ["OMP_THREAD_LIMIT"] == "2"


def test_ocr_reads_within_a_small_memory(use_engines):
    text = reader(("Invoice 2026", (10, 10, 200, 40), 0.99, "rapidocr"), width=300)
    use_engines(make_engine("rapidocr", text, memory_mb=100))
    ocr = OCR(memory_mb=1024, strategy="balanced")
    assert ocr.read(white(300, 60)).text == "Invoice 2026" and ocr.config.max_image_side == 1400


def test_huge_images_are_reduced_when_memory_is_limited(use_engines):
    seen = []

    def read(image):
        seen.append(image.shape[:2])
        return [tb("Invoice 2026", (0, 0, 2000, 50), 0.99)]

    use_engines(make_engine("rapidocr", read))
    huge = white(9000, 100)
    page = OCR(memory_mb=2048, strategy="fast").pipeline.run(huge)  # OCR side 2000 -> comes in at 4000
    assert seen == [(44, 4000)] and page.size == (4000, 44)
    assert page.original_blocks()[0].box == (0, 0, 4500, 113)  # the boxes go back to the received image
    seen.clear()
    OCR(strategy="fast").pipeline.run(huge)  # without a memory limit, as it arrives
    assert seen == [(100, 9000)]


def test_memory_already_in_use_is_not_for_the_ocr(tmp_path, monkeypatch):
    files = {
        "memory.max": f"{2 * GB}\n",
        "memory.current": f"{700 * 2**20}\n",
        "memory.stat": f"anon {600 * 2**20}\nfile {100 * 2**20}\n",
    }
    root, proc = cgroup(tmp_path, files)
    assert resources.cgroup_memory(root, proc) == (2048, 600)  # the file cache does not count: Linux releases it
    monkeypatch.setattr(resources, "machine_memory", lambda: (32768, 9000))
    monkeypatch.setattr(resources, "cgroup_memory", lambda: (2048, 120))
    assert available_memory_mb() == 2048  # a process that just started: the whole limit
    monkeypatch.setattr(resources, "cgroup_memory", lambda: (2048, 750))
    assert available_memory_mb() == 2048 - 600  # your application already uses 600 MB
    monkeypatch.setattr(resources, "cgroup_memory", lambda: None)
    assert available_memory_mb() == 32768 - 8850  # without a container: what is free on the machine
    monkeypatch.setattr(resources, "machine_memory", lambda: (2048, 2040))
    assert available_memory_mb() == 256  # almost no memory: the minimum, without an error
