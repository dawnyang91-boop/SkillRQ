"""Stable interpretable code IDs."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable


def code_id(level: str, label: object) -> str:
    text = str(label or "unknown").strip() or "unknown"
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not slug:
        slug = "label"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{level}-{slug[:32]}-{digest}"


def semantic_id(codes: Iterable[str]) -> str:
    return "/".join(codes)
