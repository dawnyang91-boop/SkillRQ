"""Markdown Code Card generation."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def write_code_cards(
    assignments: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    report_root: Path,
    datasets: Sequence[str] = ("toolbench", "api_bank", "skillret"),
) -> list[Path]:
    output_root = report_root / "code_cards"
    output_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for dataset in datasets:
        rows = [row for row in assignments if row.get("source_dataset") == dataset]
        if not rows:
            continue
        path = output_root / f"{dataset}_code_cards.md"
        path.write_text(_render_card(dataset, rows, quality), encoding="utf-8")
        paths.append(path)
    return paths


def _render_card(dataset: str, rows: Sequence[Mapping[str, Any]], quality: Mapping[str, Any]) -> str:
    metrics = (quality.get("by_dataset") or {}).get(dataset) or {}
    lines = [
        f"# {dataset} Code Cards",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in [
        "assignment_count",
        "unique_semantic_code_paths",
        "unique_l1_codes",
        "code_purity",
        "code_usage_entropy",
        "category_alignment",
        "role_alignment",
        "code_collapse_rate",
    ]:
        lines.append(f"| `{key}` | {_format(metrics.get(key))} |")

    lines.extend(["", "## Top L1 Codes", "", "| Code | Label | Count | Examples |", "|---|---|---:|---|"])
    for code, count in Counter(str(row["l1_code"]) for row in rows).most_common(12):
        label = _first_label(rows, "l1_code", code, "l1_label")
        examples = ", ".join(_examples(rows, "l1_code", code))
        lines.append(f"| `{code}` | {label} | {count} | {examples} |")

    lines.extend(["", "## Top Semantic Paths", "", "| Semantic ID | Count | Examples |", "|---|---:|---|"])
    for semantic_id, count in Counter(str(row["semantic_id"]) for row in rows).most_common(12):
        examples = ", ".join(_examples(rows, "semantic_id", semantic_id))
        lines.append(f"| `{semantic_id}` | {count} | {examples} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- L1 captures domain, scenario, category, or artifact.",
            "- L2 captures the operation or primary capability.",
            "- L3 captures weak execution role evidence.",
            "- L4 captures IO schema, constraints, examples, or validation details.",
            "",
        ]
    )
    return "\n".join(lines)


def _first_label(rows: Sequence[Mapping[str, Any]], key: str, value: str, label_key: str) -> str:
    for row in rows:
        if str(row.get(key)) == value:
            return str(row.get(label_key) or "")
    return ""


def _examples(rows: Sequence[Mapping[str, Any]], key: str, value: str, limit: int = 4) -> list[str]:
    examples = []
    seen = set()
    for row in rows:
        if str(row.get(key)) != value:
            continue
        name = str(row.get("name") or row.get("object_id") or "")
        if name in seen:
            continue
        seen.add(name)
        examples.append(name.replace("|", "/")[:80])
        if len(examples) >= limit:
            break
    return examples


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)
