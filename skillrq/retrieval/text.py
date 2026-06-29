"""Text normalization helpers for retrieval baselines."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if token]


def compact_text(parts: Iterable[Any], max_chars: int = 8000) -> str:
    text = " ".join(_stringify(part) for part in parts if part is not None)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def capability_text(row: Mapping[str, Any]) -> str:
    parameter_text = _parameter_names(row.get("parameters") or [])
    required_text = _parameter_names(row.get("required_parameters") or [])
    return compact_text(
        [
            row.get("name"),
            row.get("description"),
            row.get("category"),
            row.get("domain"),
            row.get("provider"),
            row.get("tool_name"),
            row.get("api_name"),
            row.get("method"),
            row.get("endpoint"),
            parameter_text,
            required_text,
            row.get("input_schema"),
            row.get("output_schema"),
        ]
    )


def skill_text(row: Mapping[str, Any]) -> str:
    return compact_text(
        [
            row.get("name"),
            row.get("namespace"),
            row.get("description"),
            row.get("domain_label"),
            row.get("operation_label"),
            row.get("major"),
            row.get("sub"),
            row.get("primary_action"),
            row.get("primary_object"),
            row.get("body"),
            row.get("skill_md"),
        ]
    )


def skillrouter_skill_text(row: Mapping[str, Any]) -> str:
    return compact_text([row.get("name"), row.get("description"), row.get("body"), row.get("source")])


def _parameter_names(parameters: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for parameter in parameters:
        if not isinstance(parameter, Mapping):
            continue
        parts.extend(
            str(value)
            for value in [
                parameter.get("name"),
                parameter.get("description"),
                parameter.get("type"),
            ]
            if value
        )
    return " ".join(parts)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(f"{key} {_stringify(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_stringify(item) for item in value)
    return str(value)
