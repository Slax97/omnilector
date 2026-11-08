# SPDX-License-Identifier: AGPL-3.0-or-later
"""Test helpers: fake engines and files generated on the fly (without downloading models)."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image, ImageDraw

from omnilector.engines.base import PAGE_KINDS, EngineInfo, OCREngine
from omnilector.result import TextBlock
from omnilector.routing.languages import SCRIPTS


def make_engine(
    name: str,
    read: Callable | None = None,
    *,
    level: str = "page",
    scripts: frozenset[str] = SCRIPTS,
    kinds: frozenset[str] = PAGE_KINDS,
    handwriting: bool = False,
    reason: str | None = None,
    gpu: str = "optional",
    cost: int = 1,
    read_lines: Callable | None = None,
    max_side: int | None = None,
    flipped: bool | Callable | None = None,
    memory_mb: int = 100,
    runtime: str = "",
    mb_per_megapixel: int = 0,
    print_languages: frozenset[str] = frozenset(),
    weak_languages: dict[str, float] | None = None,
    detect_lines: Callable | None = None,
    extra: str = "",
) -> type[OCREngine]:
    """Fake engine class. `calls` keeps each image read; `flipped`, what it answers (or how it decides,
    from the image and the boxes) about whether the page is upside down; `detect_lines`, the lines it finds if it
    only detects them."""

    class Fake(OCREngine):
        info = EngineInfo(
            name=name,
            level=level,
            scripts=scripts,
            kinds=kinds,
            handwriting=handwriting,
            gpu=gpu,
            cost=cost,
            memory_mb=memory_mb,
            runtime=runtime,
            mb_per_megapixel=mb_per_megapixel,
            print_languages=print_languages,
            weak_languages=weak_languages or {},
            detector=detect_lines is not None,
            extra=extra,
        )
        calls: ClassVar[list] = []

        @classmethod
        def unavailable_reason(cls):
            return reason

        def read(self, image):
            type(self).calls.append(image)
            return read(image) if read else []

        def detect(self, image):
            return detect_lines(image)

        def read_lines(self, image, boxes):
            return read_lines(boxes) if read_lines else [None] * len(boxes)

        def upside_down(self, image, boxes):
            return flipped(image, boxes) if callable(flipped) else flipped

    Fake.max_side = max_side
    Fake.calls = []
    return Fake


def tb(text: str, box: tuple[int, int, int, int], conf: float, engine: str = "rapidocr") -> TextBlock:
    return TextBlock(text=text, box=box, confidence=conf, engine=engine)


def tilted(
    text: str, x: float, y: float, w: float, h: float, angle: float, center: tuple[float, float], conf: float = 0.95
) -> TextBlock:
    """A w x h line at (x, y) of a straight page, rotated `angle` degrees (counterclockwise) around
    `center`: as a page engine reads it in a skewed photo (box and 4 corners)."""
    import math

    a = math.radians(angle)
    cx, cy = center
    corners = []
    for px, py in ((x, y), (x + w, y), (x + w, y + h), (x, y + h)):
        dx, dy = px - cx, py - cy
        corners.append(
            (round(cx + dx * math.cos(a) + dy * math.sin(a)), round(cy - dx * math.sin(a) + dy * math.cos(a)))
        )
    xs, ys = [p[0] for p in corners], [p[1] for p in corners]
    return TextBlock(
        text=text, box=(min(xs), min(ys), max(xs), max(ys)), confidence=conf, engine="rapidocr", quad=tuple(corners)
    )


def reader(*items: tuple[str, tuple[int, int, int, int], float, str], width: int = 300) -> Callable:
    """Fake reading that scales its boxes with the width of the image, like a real OCR (so enlarged passes
    land in the same place when going back to original coordinates)."""

    def read(image):
        s = image.shape[1] / width
        return [tb(text, tuple(round(v * s) for v in box), conf, engine) for text, box, conf, engine in items]

    return read


def white(w: int = 400, h: int = 400) -> np.ndarray:
    return np.full((h, w, 3), 255, np.uint8)


def png_bytes(w: int = 320, h: int = 120, text: str = "Hello 123") -> bytes:
    img = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(img).text((10, h // 3), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def zipcrypto(name: str, content: bytes, password: str) -> bytes:
    """ZIP with one uncompressed file encrypted with ZipCrypto (the classic PKWARE encryption, the default
    of 7-Zip and WinZip): zipfile reads it, but cannot write it."""
    import struct
    import zlib

    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0xEDB88320 if c & 1 else c >> 1
        table.append(c)
    keys = [0x12345678, 0x23456789, 0x34567890]

    def crc(value: int, byte: int) -> int:
        return (value >> 8) ^ table[(value ^ byte) & 0xFF]

    def update(byte: int) -> None:
        keys[0] = crc(keys[0], byte)
        keys[1] = ((keys[1] + (keys[0] & 0xFF)) * 134775813 + 1) & 0xFFFFFFFF
        keys[2] = crc(keys[2], keys[1] >> 24)

    for byte in password.encode():
        update(byte)
    checksum = zlib.crc32(content)
    secret = bytearray()
    for byte in bytes(11) + bytes([checksum >> 24]) + content:  # 12-byte header; the last one checks the key
        k = (keys[2] | 2) & 0xFFFF
        secret.append(byte ^ (((k * (k ^ 1)) >> 8) & 0xFF))
        update(byte)
    encoded = name.encode()
    sizes = (checksum, len(secret), len(content), len(encoded))
    local = struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 1, 0, 0, 0x21, *sizes, 0) + encoded
    central = struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, 1, 0, 0, 0x21, *sizes, 0, 0, 0, 0, 0, 0) + encoded
    end = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, len(central), len(local) + len(secret), 0)
    return bytes(local + secret + central + end)


def make_mixed_pdf(path: Path, image_png: bytes | None = None) -> None:
    """Page 1 with native text; page 2 scanned (image only)."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(6):
        page.insert_text((72, 100 + i * 20), f"Native line number {i} with enough text", fontsize=11)
    scanned = doc.new_page()
    scanned.insert_image(pymupdf.Rect(50, 50, 550, 300), stream=image_png or png_bytes(800, 300, "SCANNED"))
    doc.save(path)
    doc.close()


def make_hybrid_pdf(path: Path) -> None:
    """A page with native text at the top and a large image with text below."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(5):
        page.insert_text((72, 60 + 15 * i), f"Native text of the hybrid page {i}", fontsize=10)
    page.insert_image(pymupdf.Rect(50, 200, 550, 700), stream=png_bytes(600, 600, "PHOTO"))
    doc.save(path)
    doc.close()


def ole_file(streams: dict[str, bytes]) -> bytes:
    """Minimal OLE file (Compound File Binary v3) with those streams ("a/b" = stream b of storage a), like
    .msg files and legacy Office: streams under 4096 bytes in the mini stream and the rest in normal
    sectors, each one with its exact size."""
    import struct

    free, end, fat_mark, none = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFF
    tree: dict[str, dict] = {"": {}}  # storage -> {name: child storage or bytes}
    for path, data in streams.items():
        *parents, leaf = path.split("/")
        node = tree[""]
        for i, part in enumerate(parents):
            key = "/".join(parents[: i + 1])
            tree.setdefault(key, {})
            node = node.setdefault(part, tree[key])
        node[leaf] = data
    entries: list[dict] = [{"name": "Root Entry", "type": 5}]

    def add(children: dict) -> int:
        ids = []
        for name, value in children.items():
            entries.append({"name": name, "type": 1 if isinstance(value, dict) else 2, "value": value})
            ids.append(len(entries) - 1)
        for i, sid in enumerate(ids):  # siblings chained to the right (olefile does not require red-black)
            entries[sid]["right"] = ids[i + 1] if i + 1 < len(ids) else none
            if entries[sid]["type"] == 1:
                entries[sid]["child"] = add(entries[sid]["value"])
        return ids[0] if ids else none

    entries[0]["child"] = add(tree[""])
    # small streams: 64-byte mini sectors inside the mini stream (the Root Entry stream)
    mini, minifat = b"", []
    for e in entries:
        if e["type"] == 2 and len(e["value"]) < 4096:
            count = max(1, -(-len(e["value"]) // 64))
            e["start"] = len(minifat)
            minifat += [*range(len(minifat) + 1, len(minifat) + count), end]
            mini += e["value"].ljust(count * 64, b"\x00")
    big = [e for e in entries if e["type"] == 2 and len(e["value"]) >= 4096]
    minifat_bytes = struct.pack(f"<{len(minifat)}I", *minifat) if minifat else b""
    dir_bytes_len = -(-len(entries) * 128 // 512) * 512
    # in normal sectors, in order: the mini FAT, the mini stream and the large streams
    chains = [(minifat_bytes, "minifat"), (mini, "mini"), *[(e["value"], id(e)) for e in big]]
    data_sectors = sum(-(-len(c) // 512) for c, _ in chains)
    dir_sectors = dir_bytes_len // 512
    fat_sectors = 1
    while fat_sectors * 128 < fat_sectors + dir_sectors + data_sectors:
        fat_sectors += 1
    fat = [fat_mark] * fat_sectors + [*range(fat_sectors + 1, fat_sectors + dir_sectors), end]
    sector, first = fat_sectors + dir_sectors, {}
    for content, owner in chains:
        count = -(-len(content) // 512)
        first[owner] = sector if count else end
        fat += [*range(sector + 1, sector + count), end] if count else []
        sector += count
    fat += [free] * (fat_sectors * 128 - len(fat))
    entries[0]["start"], entries[0]["size"] = first["mini"], len(mini)
    for e in big:
        e["start"] = first[id(e)]
    directory = b""
    for e in entries:
        name = e["name"].encode("utf-16-le")
        size = e.get("size", len(e["value"]) if e["type"] == 2 else 0)
        directory += (
            name.ljust(64, b"\x00")
            + struct.pack("<HBB", len(name) + 2, e["type"], 1)
            + struct.pack("<III", none, e.get("right", none), e.get("child", none))
            + b"\x00" * 36
            + struct.pack("<IQ", e.get("start", end), size)
        )
    minifat_sectors = -(-len(minifat_bytes) // 512)
    header = (
        bytes.fromhex("D0CF11E0A1B11AE1")
        + b"\x00" * 16
        + struct.pack("<HHHHH", 0x3E, 3, 0xFFFE, 9, 6)
        + b"\x00" * 6
        + struct.pack("<IIIII", 0, fat_sectors, fat_sectors, 0, 4096)
        + struct.pack("<IIII", first["minifat"], minifat_sectors, end, 0)
        + struct.pack("<109I", *range(fat_sectors), *([free] * (109 - fat_sectors)))
    )
    body = struct.pack(f"<{len(fat)}I", *fat) + directory.ljust(dir_bytes_len, b"\x00")
    body += b"".join(c.ljust(-(-len(c) // 512) * 512, b"\x00") for c, _ in chains)
    return header + body


def msg_file(
    subject: str,
    body: str,
    attachments: list[tuple[str, bytes, str | None]] = (),
    forwarded: dict | None = None,
    sender: str = "Example Shop",
    email: str = "shop@example.com",
) -> bytes:
    """Minimal Outlook .msg: subject, sender, body, attachments (name, data, content-id if it is a body
    image) and, if given, an attached email {"subject": ..., "body": ...}."""
    import struct

    def text(value: str) -> bytes:
        return value.encode("utf-16-le")

    filetime = (1789123456 + 11644473600) * 10_000_000  # 2026-09-11 as a FILETIME
    streams = {
        "__properties_version1.0": b"\x00" * 32 + struct.pack("<IIQ", 0x00390040, 0, filetime),
        "__substg1.0_0037001F": text(subject),
        "__substg1.0_0C1A001F": text(sender),
        "__substg1.0_0C1F001F": text(email),
        "__substg1.0_0E04001F": text("Customer"),
        "__substg1.0_1000001F": text(body),
    }
    for i, (name, data, cid) in enumerate(attachments):
        folder = f"__attach_version1.0_#{i:08X}"
        streams[f"{folder}/__properties_version1.0"] = b"\x00" * 8
        streams[f"{folder}/__substg1.0_3707001F"] = text(name)
        streams[f"{folder}/__substg1.0_37010102"] = data
        if cid:
            streams[f"{folder}/__substg1.0_3712001F"] = text(cid)
    if forwarded:
        folder = f"__attach_version1.0_#{len(attachments):08X}"
        inner = f"{folder}/__substg1.0_3701000D"
        streams[f"{folder}/__properties_version1.0"] = b"\x00" * 8
        streams[f"{folder}/__substg1.0_3707001F"] = text("forwarded.msg")
        streams[f"{inner}/__properties_version1.0"] = b"\x00" * 24
        streams[f"{inner}/__substg1.0_0037001F"] = text(forwarded["subject"])
        streams[f"{inner}/__substg1.0_1000001F"] = text(forwarded["body"])
    return ole_file(streams)
