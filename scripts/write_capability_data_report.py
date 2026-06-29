#!/usr/bin/env python3
"""Write a Markdown report for canonical capability data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed" / "capability"
REPORT_PATH = ROOT / "docs" / "CAPABILITY_DATA_REPORT.md"


def main() -> int:
    files = [
        ("capabilities.jsonl", "Capability objects: tools, APIs, functions, and runtime actions."),
        ("capability_queries.jsonl", "User instructions, dialogues, tasks, and trajectory-only records."),
        ("capability_qrels.jsonl", "Query-capability relevance labels."),
        ("capability_sequences.jsonl", "Step-wise tool/API calls extracted from trajectories."),
    ]
    stats = _read_json(DATA_ROOT / "capability_stats.json")
    lines = [
        "# Capability Processed Data Report",
        "",
        f"Data directory: `{DATA_ROOT}`",
        "",
        "## Files",
        "",
        "| File | Rows | Fields | Description |",
        "|---|---:|---|---|",
    ]
    for filename, description in files:
        path = DATA_ROOT / filename
        rows = _count_lines(path)
        fields = ", ".join(_collect_fields(path))
        lines.append(f"| `{filename}` | {rows:,} | `{fields}` | {description} |")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for key in [
        "capabilities",
        "queries",
        "qrels",
        "sequences",
        "min_unique_tools_per_query",
        "max_unique_tools_per_query",
        "avg_unique_tools_per_query",
        "min_tool_calls_per_trajectory",
        "max_tool_calls_per_trajectory",
        "avg_tool_calls_per_trajectory",
    ]:
        value = stats.get(key)
        lines.append(f"| `{key}` | {_format_value(value)} |")

    lines.extend(
        [
            "",
            "## Queries By Dataset",
            "",
            "| Dataset | Queries |",
            "|---|---:|",
        ]
    )
    for dataset, count in sorted((stats.get("queries_by_dataset") or {}).items()):
        lines.append(f"| `{dataset}` | {count:,} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `unique_tools_per_query` is the size of the unique gold capability set and is used for recommendation / set selection.",
            "- `tool_calls_per_trajectory` is the length of the extracted execution sequence and is used for execution-order analysis.",
            "- ToolBench instruction records and API-Bank dialogues share the same canonical schema.",
            "- ToolBench answer-tree records are retained as trajectory-only examples when full answer extraction is enabled.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_PATH)
    return 0


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _collect_fields(path: Path) -> list[str]:
    fields: set[str] = set()
    for row in _iter_jsonl(path, limit=1000):
        fields.update(row.keys())
    return sorted(fields)


def _iter_jsonl(path: Path, limit: int | None = None) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if line.strip():
                yield json.loads(line)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())

