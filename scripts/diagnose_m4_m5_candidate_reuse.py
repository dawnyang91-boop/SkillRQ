#!/usr/bin/env python3
"""Compare M4 retrieved candidates with M5 nested retrieved candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m4-prediction-path", type=Path, required=True)
    parser.add_argument("--m5-prediction-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    m4_rows = {str(row.get("query_id")): row for row in _read_jsonl(args.m4_prediction_path)}
    m5_rows = {str(row.get("query_id")): row for row in _read_jsonl(args.m5_prediction_path)}
    shared_query_ids = sorted(set(m4_rows) & set(m5_rows))

    query_count_with_gold = 0
    m4_hits = 0
    m5_hits = 0
    m4_hit_m5_miss = 0
    m4_miss_m5_hit = 0
    m4_m5_overlap_sum = 0.0
    m4_candidate_total = 0
    m5_candidate_total = 0
    reused_candidate_total = 0

    for query_id in shared_query_ids:
        m4_row = m4_rows[query_id]
        m5_row = m5_rows[query_id]
        gold_ids = {str(item) for item in (m5_row.get("gold_ids") or m4_row.get("gold_ids") or []) if item is not None}
        if not gold_ids:
            continue
        query_count_with_gold += 1
        m4_candidates = set(_candidate_ids(m4_row.get("retrieved_capabilities") or []))
        m5_candidates = set(_nested_candidate_ids(m5_row))
        m4_candidate_total += len(m4_candidates)
        m5_candidate_total += len(m5_candidates)
        reused_candidate_total += len(m4_candidates & m5_candidates)
        m4_m5_overlap_sum += len(m4_candidates & m5_candidates) / max(len(m4_candidates | m5_candidates), 1)

        m4_hit = bool(gold_ids & m4_candidates)
        m5_hit = bool(gold_ids & m5_candidates)
        m4_hits += int(m4_hit)
        m5_hits += int(m5_hit)
        m4_hit_m5_miss += int(m4_hit and not m5_hit)
        m4_miss_m5_hit += int((not m4_hit) and m5_hit)

    report = {
        "m4_prediction_path": str(args.m4_prediction_path),
        "m5_prediction_path": str(args.m5_prediction_path),
        "m4_queries": len(m4_rows),
        "m5_queries": len(m5_rows),
        "shared_queries": len(shared_query_ids),
        "evaluated_queries": query_count_with_gold,
        "m4_hit_rate": m4_hits / max(query_count_with_gold, 1),
        "m5_hit_rate": m5_hits / max(query_count_with_gold, 1),
        "m4_hit_m5_miss": m4_hit_m5_miss,
        "m4_hit_m5_miss_rate": m4_hit_m5_miss / max(query_count_with_gold, 1),
        "m4_miss_m5_hit": m4_miss_m5_hit,
        "m4_miss_m5_hit_rate": m4_miss_m5_hit / max(query_count_with_gold, 1),
        "avg_m4_candidates_per_query": m4_candidate_total / max(query_count_with_gold, 1),
        "avg_m5_candidates_per_query": m5_candidate_total / max(query_count_with_gold, 1),
        "m4_candidate_reuse_rate": reused_candidate_total / max(m4_candidate_total, 1),
        "avg_m4_m5_candidate_jaccard": m4_m5_overlap_sum / max(query_count_with_gold, 1),
    }
    _emit(report, args.output_path)
    return 0


def _nested_candidate_ids(row: Mapping[str, Any]) -> list[str]:
    ids = []
    ids.extend(_candidate_ids(row.get("retrieved_capabilities") or []))
    for key in ("residual_code_paths", "code_plan"):
        for step in row.get(key) or []:
            if isinstance(step, Mapping):
                ids.extend(_candidate_ids(step.get("retrieved_capabilities") or []))
    return ids


def _candidate_ids(candidates: Iterable[Mapping[str, Any]]) -> list[str]:
    return [str(item.get("candidate_id")) for item in candidates if isinstance(item, Mapping) and item.get("candidate_id") is not None]


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
