# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cores and memory the process can use (container, systemd service, taskset) and how much each part of
the OCR takes: with that it adapts on its own to any server. OCRConfig.threads, memory_mb and
max_image_side win if they are set.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import math
import os
import re
import sys
from pathlib import Path

CGROUP = Path("/sys/fs/cgroup")
PROC_CGROUP = Path("/proc/self/cgroup")

# Measured on Linux (CPU, resident memory of the process). See the README, "Any server".
BASE_MB = 250  # Python, numpy, OpenCV, onnxruntime, PyMuPDF and the copies of the page
# Reading an image takes, besides the models, ~150 MB per megapixel with RapidOCR (a square page:
# 0.66 GB at 2000 px, 0.93 GB at 2500 px, 2.2 GB at 4000 px) plus the fixed part of its copies
MB_PER_MEGAPIXEL = 150
IMAGE_FIXED_MB = 60
IMAGE_SHARE = 0.35  # share of the memory to read the image; the rest, for the process and the models
MIN_SIDE = 1000
USEFUL_SIDE = 1500  # a page engine that cannot read at this size is not used in the consensus
LAYOUT_MB = 320  # layout: PP-DocLayoutV3 + SLANet-plus
# What a process with omnilector imported takes (Python, numpy, OpenCV): when looking at how much memory
# is already in use it does not count as somebody else's (it is already in BASE_MB)
BASELINE_MB = 150
PAGE_ENGINE_MB = 150  # RapidOCR: detector, classifier and recognizer (the usual page engine)
# Libraries that are loaded once and not released even if the model that used them is
RUNTIME_MB = {"torch": 200}
# With memory_mb, images with a side larger than this x max_image_side are reduced on entry (JPEGs,
# already while decoding): a 48 Mpx photo at full resolution, with its copies, took the process to the
# 2 GB limit. With twice the side the OCR sees, the crops of each line keep plenty of detail
INPUT_SIDE = 2


def input_side(memory_mb: int | None, max_image_side: int | None) -> int | None:
    """Maximum side of the image that enters the pipeline; None if it is used whole (no memory limit)."""
    return INPUT_SIDE * max_image_side if memory_mb is not None and max_image_side else None


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _v2_dirs(root: Path, proc: Path) -> list[Path]:
    """cgroup v2 folder of the process and those above it (the limit can be in any of them: the container,
    the systemd service, the slice...)."""
    rel = next((line[3:].strip("/") for line in (_read(proc) or "").splitlines() if line.startswith("0::")), "")
    here = root / rel if rel else root
    dirs = [here, *here.parents]
    return [d for d in dirs if d == root or root in d.parents]


def _v2_values(name: str, root: Path, proc: Path) -> list[str]:
    return [v for d in _v2_dirs(root, proc) if (v := _read(d / name)) is not None]


def cgroup_cpus(root: Path = CGROUP, proc: Path = PROC_CGROUP) -> float | None:
    """Cores of the CPU quota of the process's cgroup (cpu.max, or cpu.cfs_quota_us in v1); None if there is no
    quota."""
    quotas = []
    for value in _v2_values("cpu.max", root, proc):  # "200000 100000" or "max 100000"
        quota, _, period = value.partition(" ")
        if quota != "max" and period:
            quotas.append(int(quota) / int(period))
    quota_v1, period_v1 = _read(root / "cpu" / "cpu.cfs_quota_us"), _read(root / "cpu" / "cpu.cfs_period_us")
    if quota_v1 and period_v1 and int(quota_v1) > 0:
        quotas.append(int(quota_v1) / int(period_v1))
    return min(quotas) if quotas else None


def _stat(path: Path, *keys: str) -> int:
    """First value of those keys in a cgroup memory.stat (bytes); 0 if missing."""
    values = dict(line.split(" ", 1) for line in (_read(path) or "").splitlines() if " " in line)
    return next((int(values[k]) for k in keys if values.get(k, "").isdigit()), 0)


def cgroup_memory(root: Path = CGROUP, proc: Path = PROC_CGROUP) -> tuple[int, int] | None:
    """(limit, in use) in MB of the cgroup that leaves the least free (memory.max, or memory.limit_in_bytes in
    v1); in use, without the file cache, which Linux releases if needed. None without a limit."""
    found = []
    for d in _v2_dirs(root, proc):
        limit, current = _read(d / "memory.max"), _read(d / "memory.current")
        if limit and limit.isdigit():
            used = int(current) - _stat(d / "memory.stat", "file") if current and current.isdigit() else 0
            found.append((int(limit) // 2**20, max(0, used) // 2**20))
    v1, usage = _read(root / "memory" / "memory.limit_in_bytes"), _read(root / "memory" / "memory.usage_in_bytes")
    if v1 and v1.isdigit() and int(v1) < 2**60:  # without a limit, cgroup v1 gives a huge number
        used = (
            int(usage) - _stat(root / "memory" / "memory.stat", "total_cache", "cache")
            if usage and usage.isdigit()
            else 0
        )
        found.append((int(v1) // 2**20, max(0, used) // 2**20))
    return min(found, key=lambda f: f[0] - f[1]) if found else None


def cgroup_memory_mb(root: Path = CGROUP, proc: Path = PROC_CGROUP) -> int | None:
    """Memory limit of the container or service; None without a limit."""
    found = cgroup_memory(root, proc)
    return found[0] if found else None


def machine_memory() -> tuple[int, int] | None:
    """(machine RAM, in use) in MB. Outside Linux and Windows, in use is 0 (unknown)."""
    try:
        if sys.platform == "win32":

            class _Status(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong),
                    ("avail", ctypes.c_ulonglong),
                    ("page_total", ctypes.c_ulonglong),
                    ("page_avail", ctypes.c_ulonglong),
                    ("virtual_total", ctypes.c_ulonglong),
                    ("virtual_avail", ctypes.c_ulonglong),
                    ("extended", ctypes.c_ulonglong),
                ]

            status = _Status(length=ctypes.sizeof(_Status))
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return None
            return int(status.total // 2**20), int((status.total - status.avail) // 2**20)
        meminfo = dict(line.split(":", 1) for line in (_read(Path("/proc/meminfo")) or "").splitlines() if ":" in line)
        total, available = meminfo.get("MemTotal", "").split()[:1], meminfo.get("MemAvailable", "").split()[:1]
        if total and available:
            return int(total[0]) // 1024, (int(total[0]) - int(available[0])) // 1024
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**20, 0
    except (ValueError, OSError, AttributeError):
        return None


def available_cpus() -> int:
    """Cores the process can use: the assigned ones (taskset, cpuset) and the container quota."""
    if hasattr(os, "process_cpu_count"):  # Python 3.13
        count = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count()
    count = count or 1
    quota = cgroup_cpus()
    if quota:
        count = min(count, max(1, math.ceil(quota)))  # 1.5 cores -> 2 threads
    return count


def cpu_limit() -> int | None:
    """Threads per engine if the process has fewer cores than the machine (a CPU quota or fewer assigned cores);
    None if it has all of them (each engine uses its own, as always)."""
    count = available_cpus()
    return count if count < (os.cpu_count() or count) else None


def available_memory_mb() -> int | None:
    """Memory for the OCR: the limit of the container or service (or the machine's RAM) minus what is already
    in use: your application, other processes, other workers. What this process takes with omnilector just
    imported does not count (BASELINE_MB). Inside an application already using 600 MB of 2 GB, ~1.5 GB."""
    free = [
        limit - max(0, used - BASELINE_MB)
        for found in (cgroup_memory(), machine_memory())
        if found
        for limit, used in [found]
    ]
    return max(256, min(free)) if free else None


def parse_memory(value: str | int) -> int:
    """'2g', '1.5G', '2048m', '2048' (MB) -> MB."""
    if isinstance(value, int):
        return value
    match = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*([gmk]?)i?b?\s*", value.lower())
    if not match:
        raise ValueError(f"invalid memory: {value!r} (use e.g. 2g, 1536m or 2048)")
    number, unit = float(match.group(1).replace(",", ".")), match.group(2)
    return int(number * {"g": 1024, "m": 1, "k": 1 / 1024, "": 1}[unit])


def side_within(pool_mb: int, mb_per_megapixel: int = MB_PER_MEGAPIXEL) -> int:
    """Maximum image side (square, the worst case) an engine can read with that memory."""
    megapixels = max(0, pool_mb - IMAGE_FIXED_MB) / mb_per_megapixel
    return max(MIN_SIDE, int(math.sqrt(megapixels) * 1000) // 100 * 100)


def image_side_for(memory_mb: int) -> int:
    """Maximum image side for RapidOCR with that memory (2 GB: 2000 px)."""
    return side_within(int(memory_mb * IMAGE_SHARE))


def image_mb(side: int, mb_per_megapixel: int = MB_PER_MEGAPIXEL) -> int:
    """Memory to read an image of that side (square)."""
    return IMAGE_FIXED_MB + math.ceil((side / 1000) ** 2 * mb_per_megapixel)


def engine_side(memory_mb: int | None, mb_per_megapixel: int) -> int | None:
    """Side at which a page engine can read with the share of memory for reading the image; None without a
    memory limit."""
    if memory_mb is None or not mb_per_megapixel:
        return None
    return side_within(int(memory_mb * IMAGE_SHARE), mb_per_megapixel)


def models_budget_mb(memory_mb: int | None, side: int | None) -> int | None:
    """Memory for the loaded models: what is left after the process and the image. None without a limit."""
    if memory_mb is None:
        return None
    return memory_mb - BASE_MB - image_mb(side or 0)


def release_memory() -> None:
    """Gives the memory of released models back to the system (on Linux glibc kept it)."""
    gc.collect()
    if sys.platform.startswith("linux"):
        with contextlib.suppress(OSError, AttributeError):  # without glibc (Alpine, musl) there is no malloc_trim
            ctypes.CDLL("libc.so.6").malloc_trim(0)
