"""Helpers for reading candidate pools from M4/M5/M7 predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..utils.io import read_jsonl


def load_prediction_rows(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def flatten_candidate_ids(row: Mapping[str, Any]) -> list[str]:
    if row.get("retrieved_capabilities") is not None:
        return [str(item.get("candidate_id")) for item in row.get("retrieved_capabilities") or []]
    if row.get("residual_code_paths") is not None:
        candidate_ids = []
        for path in row.get("residual_code_paths") or []:
            candidate_ids.extend(str(item.get("candidate_id")) for item in path.get("retrieved_capabilities") or [])
        return candidate_ids
    if row.get("reranked_capabilities") is not None:
        return [str(item.get("candidate_id")) for item in row.get("reranked_capabilities") or []]
    if row.get("predicted_tool_set") is not None:
        return [str(item) for item in row.get("predicted_tool_set") or []]
    return []


def unique_in_order(values: Sequence[str]) -> list[str]:
    seen = set()
    rows = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        rows.append(value)
    return rows


def gold_ids(row: Mapping[str, Any]) -> list[str]:
    return [str(item) for item in row.get("gold_ids") or []]
