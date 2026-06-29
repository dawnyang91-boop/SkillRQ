"""Stable ID helpers for capability datasets."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable


def slugify(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def stable_id(prefix: str, parts: Iterable[object]) -> str:
    raw = "::".join(str(part or "") for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    readable = "::".join(slugify(part) for part in parts if str(part or "").strip())
    if readable:
        return f"{prefix}::{readable}::{digest}"
    return f"{prefix}::{digest}"

