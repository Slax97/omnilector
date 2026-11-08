# SPDX-License-Identifier: AGPL-3.0-or-later
"""Email (.eml): headers + body (text or HTML), attachments and body images (pasted screenshots,
multipart/related), each one read with its own loader."""

from __future__ import annotations

from collections.abc import Iterator
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from typing import TYPE_CHECKING

from .base import RawDocument, RawPage, Source, grid_rows, nested_pages
from .text import decode_text, html_cells, text_cells

if TYPE_CHECKING:
    from ..config import OCRConfig

_HEADERS = (("from", "From"), ("to", "To"), ("cc", "CC"), ("date", "Date"), ("subject", "Subject"))


def _body_cells(body: EmailMessage | None, languages: tuple[str, ...]) -> list[list[str]]:
    if body is None:
        return []
    try:
        content = body.get_content()
    except (LookupError, ValueError):  # odd or mislabeled charset: the usual detection
        content = decode_text(body.get_payload(decode=True) or b"", languages)[0]
    if body.get_content_subtype() == "html":
        return html_cells(content)[0]
    return text_cells(str(content))


def _leaves(part: EmailMessage, parents: tuple = ()) -> Iterator[tuple[EmailMessage, tuple]]:
    """Leaf parts of the email with their containers. An attached email is a leaf part: it is read whole,
    separately, as an .eml."""
    if part.is_multipart() and part.get_content_maintype() != "message":
        for sub in part.iter_parts():
            yield from _leaves(sub, (*parents, part))
    else:
        yield part, parents


def _extra_parts(msg: EmailMessage, body: EmailMessage | None) -> list[EmailMessage]:
    """Everything that is not the body: attachments and body images, also those inside multipart/related
    (iter_attachments() only looks at the first level and skips them). The body in another format (the
    HTML of the same text) does not count."""
    leaves = list(_leaves(msg))
    alternative = next(
        (
            p
            for part, parents in leaves
            if part is body
            for p in reversed(parents)
            if p.get_content_type() == "multipart/alternative"
        ),
        None,
    )
    extra = []
    for part, parents in leaves:
        explicit = part.get_content_disposition() == "attachment" or part.get_filename()
        if part is body or (
            not explicit and part.get_content_maintype() == "text" and any(p is alternative for p in parents)
        ):
            continue
        extra.append(part)
    return extra


def _inline(part: EmailMessage) -> bool:
    """Body image (referenced with cid:), not attached."""
    return part.get_content_disposition() != "attachment" and part.get("Content-ID") is not None


def _attachment(part: EmailMessage) -> tuple[str, bytes | None]:
    if part.get_content_maintype() == "message":  # email forwarded as an attachment
        inner = part.get_content()
        data = inner.as_bytes() if isinstance(inner, EmailMessage) else None
        return part.get_filename() or "attachment.eml", data
    name = part.get_filename() or str(part.get("Content-ID", "")).strip("<> ") or "attachment"
    return name, part.get_payload(decode=True)


def load(src: Source, config: OCRConfig) -> RawDocument:
    msg = BytesParser(policy=policy.default).parsebytes(src.read_bytes())
    body = msg.get_body(preferencelist=("plain", "html"))
    metadata = {key: str(msg[key]) for key, _ in _HEADERS if msg[key]}
    cells = [[f"{label}: {msg[key]}"] for key, label in _HEADERS if msg[key]]
    cells += [[""], *_body_cells(body, config.language_hints)]
    parts = [(_inline(part), *_attachment(part)) for part in _extra_parts(msg, body)]
    metadata["attachments"] = [name for inline, name, _ in parts if not inline]
    if any(inline for inline, _, _ in parts):
        metadata["inline_images"] = [name for inline, name, _ in parts if inline]
    rows, size = grid_rows(cells)
    warnings: list[str] = []
    parent = Source(name=src.name, depth=src.depth, budget=src.budget)  # without the bytes of the email

    def pages():
        yield RawPage(1, rows=rows, size=size)
        number = 1
        while parts:
            inline, name, data = parts.pop(0)  # each part is released once done with it
            if not data:
                continue
            prefix = "body image " if inline else "attachment "
            for page in nested_pages(parent, name, data, config, prefix, warnings):
                number += 1
                page.number = number
                yield page

    return RawDocument("email", pages(), metadata=metadata, warnings=warnings)
