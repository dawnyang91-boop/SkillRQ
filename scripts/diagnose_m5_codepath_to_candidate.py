#!/usr/bin/env python3
"""Diagnose the M5 code-path-to-candidate retrieval chain."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping


GENERIC_MARKERS = (
    "get_all",
    "search_for",
    "search_basic",
    "method_unknown",
    "schema_light",
    "toolbench_answer_tree",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-path", type=Path, required=True)
    parser.add_argument("--m4-data-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    candidates = list(_read_jsonl(args.m4_data_root / "candidates.jsonl"))
    query_index = {str(row["query_id"]): row for row in _read_jsonl(args.m4_data_root / "queries.jsonl")}
    bucket_index, candidate_rank_index, candidate_path_index = _build_bucket_indexes(candidates)

    predicted_path_count = 0
    predicted_bucket_hits = 0
    selected_large_50 = 0
    selected_large_100 = 0
    generic_counter: Counter[str] = Counter()
    generic_predicted_paths = 0

    total_gold_paths = 0
    covered_gold_paths = 0
    gold_candidate_checks = 0
    gold_candidate_in_exact_bucket = 0
    gold_candidate_rank_values: list[int] = []
    predicted_candidate_ids: set[str] = set()
    gold_candidate_ids: set[str] = set()
    query_gold_path_coverages: list[float] = []

    for pred_row in _read_jsonl(args.prediction_path):
        query_id = str(pred_row.get("query_id"))
        query = query_index.get(query_id, pred_row)
        predicted_paths = [_path_key(step) for step in _iter_prediction_steps(pred_row) if _path_key(step)]
        predicted_path_set = set(predicted_paths)

        for step in _iter_prediction_steps(pred_row):
            path_key = _path_key(step)
            if not path_key:
                continue
            predicted_path_count += 1
            bucket_size = len(bucket_index.get(path_key, []))
            if bucket_size:
                predicted_bucket_hits += 1
            if bucket_size > 50:
                selected_large_50 += 1
            if bucket_size > 100:
                selected_large_100 += 1
            marker = _generic_marker(path_key)
            if marker:
                generic_predicted_paths += 1
                generic_counter[marker] += 1
            predicted_candidate_ids.update(_candidate_ids(step.get("retrieved_capabilities") or []))

        gold_paths = []
        for gold_path in query.get("gold_code_paths") or []:
            path_key = _path_key(gold_path)
            candidate_id = gold_path.get("candidate_id")
            if path_key:
                gold_paths.append(path_key)
            if candidate_id is not None:
                candidate_id = str(candidate_id)
                gold_candidate_ids.add(candidate_id)
                gold_candidate_checks += 1
                native_path = path_key or candidate_path_index.get(candidate_id)
                rank = candidate_rank_index.get((native_path, candidate_id)) if native_path else None
                if rank is not None:
                    gold_candidate_in_exact_bucket += 1
                    gold_candidate_rank_values.append(rank)
        total_gold_paths += len(gold_paths)
        if gold_paths:
            query_covered = sum(1 for path in gold_paths if path in predicted_path_set)
            covered_gold_paths += query_covered
            query_gold_path_coverages.append(query_covered / len(gold_paths))

    report = {
        "prediction_path": str(args.prediction_path),
        "m4_data_root": str(args.m4_data_root),
        "unique_candidate_code_paths": len(bucket_index),
        "predicted_path_occurrences": predicted_path_count,
        "path_bucket_hit_rate": predicted_bucket_hits / max(predicted_path_count, 1),
        "gold_path_covered_rate": covered_gold_paths / max(total_gold_paths, 1),
        "query_avg_gold_path_covered_rate": mean(query_gold_path_coverages) if query_gold_path_coverages else 0.0,
        "gold_candidate_in_exact_bucket_rate": gold_candidate_in_exact_bucket / max(gold_candidate_checks, 1),
        "gold_candidate_rank_in_exact_bucket": _rank_summary(gold_candidate_rank_values),
        "rank<=20": sum(1 for rank in gold_candidate_rank_values if rank <= 20) / max(len(gold_candidate_rank_values), 1),
        "rank<=100": sum(1 for rank in gold_candidate_rank_values if rank <= 100) / max(len(gold_candidate_rank_values), 1),
        "large_bucket_ratio_gt_50": selected_large_50 / max(predicted_path_count, 1),
        "large_bucket_ratio_gt_100": selected_large_100 / max(predicted_path_count, 1),
        "generic_path_ratio": generic_predicted_paths / max(predicted_path_count, 1),
        "generic_path_frequency": dict(generic_counter.most_common()),
        "unique_gold_ids": len(gold_candidate_ids),
        "unique_pred_ids": len(predicted_candidate_ids),
        "global_id_intersection": len(gold_candidate_ids & predicted_candidate_ids),
        "diagnosis": _diagnosis(
            path_bucket_hit_rate=predicted_bucket_hits / max(predicted_path_count, 1),
            gold_path_covered_rate=covered_gold_paths / max(total_gold_paths, 1),
            rank_le_100=sum(1 for rank in gold_candidate_rank_values if rank <= 100) / max(len(gold_candidate_rank_values), 1),
            large_bucket_ratio=selected_large_100 / max(predicted_path_count, 1),
            global_intersection=len(gold_candidate_ids & predicted_candidate_ids),
        ),
    }
    _emit(report, args.output_path)
    return 0


def _build_bucket_indexes(candidates: Iterable[Mapping[str, Any]]):
    bucket_index: dict[str, list[str]] = defaultdict(list)
    candidate_rank_index: dict[tuple[str, str], int] = {}
    candidate_path_index: dict[str, str] = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id"))
        path_key = _path_key(candidate)
        if not candidate_id or not path_key:
            continue
        bucket_index[path_key].append(candidate_id)
        candidate_path_index[candidate_id] = path_key
        candidate_rank_index[(path_key, candidate_id)] = len(bucket_index[path_key])
    return dict(bucket_index), candidate_rank_index, candidate_path_index


def _iter_prediction_steps(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    steps: list[Mapping[str, Any]] = []
    for key in ("residual_code_paths", "code_plan"):
        for item in row.get(key) or []:
            if isinstance(item, Mapping):
                steps.append(item)
    if row.get("retrieved_capabilities"):
        steps.append({"retrieved_capabilities": row.get("retrieved_capabilities") or []})
    return steps


def _path_key(row: Mapping[str, Any]) -> str | None:
    semantic_id = row.get("semantic_id")
    if semantic_id:
        return str(semantic_id)
    codes = row.get("codes") or row.get("code_path")
    if isinstance(codes, list) and len(codes) == 4:
        return "/".join(str(item) for item in codes)
    return None


def _candidate_ids(candidates: Iterable[Mapping[str, Any]]) -> list[str]:
    return [str(item.get("candidate_id")) for item in candidates if isinstance(item, Mapping) and item.get("candidate_id") is not None]


def _generic_marker(path_key: str) -> str | None:
    lowered = path_key.lower()
    for marker in GENERIC_MARKERS:
        if marker in lowered:
            return marker
    return None


def _rank_summary(values: list[int]) -> Mapping[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "max": max(values),
        "log_mean": mean(math.log1p(value) for value in values),
    }


def _diagnosis(
    path_bucket_hit_rate: float,
    gold_path_covered_rate: float,
    rank_le_100: float,
    large_bucket_ratio: float,
    global_intersection: int,
) -> list[str]:
    findings = []
    if path_bucket_hit_rate < 0.2:
        findings.append("A_or_D: predicted paths rarely exist in candidate buckets; check code path schema or id/path mapping.")
    if gold_path_covered_rate < 0.2:
        findings.append("A: gold code paths are usually not predicted by M5.")
    if gold_path_covered_rate >= 0.2 and rank_le_100 < 0.5:
        findings.append("B: gold paths are sometimes covered, but gold candidates rank poorly inside exact buckets.")
    if large_bucket_ratio > 0.3:
        findings.append("C: many selected paths fall into large generic buckets.")
    if global_intersection == 0:
        findings.append("D_or_E: predicted candidate ids have no global overlap with gold ids; check id mapping or evaluator schema.")
    if not findings:
        findings.append("E: evaluator schema appears readable; remaining issue is likely candidate ranking or downstream truncation.")
    return findings


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
