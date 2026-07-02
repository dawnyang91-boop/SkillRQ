#!/usr/bin/env python3
"""Diagnose nested candidate fields in M5 prediction files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    rows = list(_read_jsonl(args.prediction_path))
    row_count = len(rows)
    total_steps = 0
    total_nested_candidates = 0
    rows_with_candidates = 0
    step_candidate_counts: Counter[int] = Counter()
    step_hit_distribution: Counter[int] = Counter()
    unique_gold_ids: set[str] = set()
    unique_pred_ids: set[str] = set()
    query_hits = 0

    for row in rows:
        gold_ids = {str(item) for item in row.get("gold_ids") or [] if item is not None}
        unique_gold_ids.update(gold_ids)
        query_candidate_ids: list[str] = []
        query_hit_steps = 0
        steps = _iter_prediction_steps(row)
        total_steps += len(steps)

        for step_index, step in enumerate(steps):
            candidates = _candidate_ids(step.get("retrieved_capabilities") or [])
            step_candidate_counts[len(candidates)] += 1
            total_nested_candidates += len(candidates)
            unique_pred_ids.update(candidates)
            query_candidate_ids.extend(candidates)
            if gold_ids and gold_ids.intersection(candidates):
                query_hit_steps += 1
                step_hit_distribution[step_index] += 1

        if query_candidate_ids:
            rows_with_candidates += 1
        if gold_ids and gold_ids.intersection(query_candidate_ids):
            query_hits += 1

    report = {
        "prediction_path": str(args.prediction_path),
        "rows": row_count,
        "total_steps": total_steps,
        "avg_steps": total_steps / max(row_count, 1),
        "total_nested_candidates": total_nested_candidates,
        "avg_candidates_per_query": total_nested_candidates / max(row_count, 1),
        "avg_nested_candidates": total_nested_candidates / max(row_count, 1),
        "rows_with_candidates": rows_with_candidates,
        "rows_with_candidates_ratio": rows_with_candidates / max(row_count, 1),
        "unique_gold_ids": len(unique_gold_ids),
        "unique_pred_ids": len(unique_pred_ids),
        "global_id_intersection": len(unique_gold_ids & unique_pred_ids),
        "query_hit_rate": query_hits / max(sum(1 for row in rows if row.get("gold_ids")), 1),
        "step_candidate_counts": _counter_to_ranges(step_candidate_counts),
        "step_hit_distribution": {str(key): value for key, value in sorted(step_hit_distribution.items())},
    }
    _emit(report, args.output_path)
    return 0


def _iter_prediction_steps(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    steps: list[Mapping[str, Any]] = []
    for key in ("residual_code_paths", "code_plan"):
        for item in row.get(key) or []:
            if isinstance(item, Mapping):
                steps.append(item)
    if row.get("retrieved_capabilities"):
        steps.append({"step_index": None, "retrieved_capabilities": row.get("retrieved_capabilities") or []})
    return steps


def _candidate_ids(candidates: Iterable[Mapping[str, Any]]) -> list[str]:
    ids = []
    for item in candidates:
        candidate_id = item.get("candidate_id") if isinstance(item, Mapping) else None
        if candidate_id is not None:
            ids.append(str(candidate_id))
    return ids


def _counter_to_ranges(counter: Counter[int]) -> Mapping[str, int]:
    ranges = {
        "0": 0,
        "1-5": 0,
        "6-10": 0,
        "11-20": 0,
        "21-50": 0,
        "51-100": 0,
        ">100": 0,
    }
    for value, count in counter.items():
        if value == 0:
            ranges["0"] += count
        elif value <= 5:
            ranges["1-5"] += count
        elif value <= 10:
            ranges["6-10"] += count
        elif value <= 20:
            ranges["11-20"] += count
        elif value <= 50:
            ranges["21-50"] += count
        elif value <= 100:
            ranges["51-100"] += count
        else:
            ranges[">100"] += count
    return ranges


def _read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _emit(report: Mapping[str, Any], output_path: Path | None) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    raise SystemExit(main())
