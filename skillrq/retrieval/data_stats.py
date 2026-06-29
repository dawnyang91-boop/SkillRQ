"""Data statistics reports for M2 datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .datasets import CAPABILITY_DATASETS
from ..config.schema import PathsConfig
from ..utils.io import read_jsonl, write_json


def build_m2_data_stats(paths: PathsConfig, dataset: str) -> Mapping[str, Any]:
    if dataset in CAPABILITY_DATASETS:
        return _capability_stats(paths, dataset)
    if dataset == "skillret":
        return _skillret_stats(paths)
    if dataset == "skillrouter":
        return _skillrouter_stats(paths)
    raise ValueError(f"Unsupported M2 dataset: {dataset}")


def write_m2_data_stats(paths: PathsConfig, dataset: str) -> Mapping[str, Any]:
    stats = build_m2_data_stats(paths, dataset)
    output_path = paths.report_root / "data_stats" / f"{dataset}_stats.json"
    write_json(output_path, stats)
    return stats


def _capability_stats(paths: PathsConfig, dataset: str) -> Mapping[str, Any]:
    query_count = 0
    capability_count = 0
    unique_counts: list[int] = []
    call_counts: list[int] = []
    multi = 0
    ge_3 = 0
    ge_5 = 0
    for row in read_jsonl(paths.capability_processed_root / "capabilities.jsonl"):
        if row.get("source_dataset") == dataset:
            capability_count += 1
    for row in read_jsonl(paths.capability_processed_root / "capability_queries.jsonl"):
        if row.get("source_dataset") != dataset:
            continue
        query_count += 1
        unique_count = len(row.get("gold_capability_ids") or [])
        call_count = len(row.get("tool_call_sequence") or [])
        unique_counts.append(unique_count)
        call_counts.append(call_count)
        multi += int(unique_count > 1)
        ge_3 += int(unique_count >= 3)
        ge_5 += int(unique_count >= 5)
    return _common_stats(dataset, query_count, capability_count, unique_counts, call_counts, multi, ge_3, ge_5)


def _skillret_stats(paths: PathsConfig) -> Mapping[str, Any]:
    query_count = 0
    capability_count = sum(1 for _ in read_jsonl(paths.processed_root / "skills.jsonl"))
    unique_counts: list[int] = []
    multi = 0
    ge_3 = 0
    ge_5 = 0
    for row in read_jsonl(paths.processed_root / "queries.jsonl"):
        query_count += 1
        unique_count = len(row.get("gold_skill_ids") or [])
        unique_counts.append(unique_count)
        multi += int(unique_count > 1)
        ge_3 += int(unique_count >= 3)
        ge_5 += int(unique_count >= 5)
    return _common_stats("skillret", query_count, capability_count, unique_counts, [], multi, ge_3, ge_5)


def _skillrouter_stats(paths: PathsConfig) -> Mapping[str, Any]:
    root = paths.raw_root / "skillrouter" / "eval_core"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    relevance = json.loads((root / "relevance.json").read_text(encoding="utf-8"))
    query_count = 0
    unique_counts: list[int] = []
    multi = 0
    ge_3 = 0
    ge_5 = 0
    for row in read_jsonl(root / "tasks.jsonl"):
        if row.get("excluded"):
            continue
        query_count += 1
        rel = relevance.get(str(row["task_id"])) or {}
        unique_count = len(rel.get("gt_skill_ids") or row.get("skill_names") or [])
        unique_counts.append(unique_count)
        multi += int(unique_count > 1)
        ge_3 += int(unique_count >= 3)
        ge_5 += int(unique_count >= 5)
    capability_count = int((manifest.get("easy") or {}).get("records") or 0) + int(
        (manifest.get("hard") or {}).get("records") or 0
    )
    return _common_stats("skillrouter", query_count, capability_count, unique_counts, [], multi, ge_3, ge_5)


def _common_stats(
    dataset: str,
    query_count: int,
    capability_count: int,
    unique_counts: list[int],
    call_counts: list[int],
    multi: int,
    ge_3: int,
    ge_5: int,
) -> Mapping[str, Any]:
    denominator = query_count or 1
    return {
        "dataset": dataset,
        "query_count": query_count,
        "capability_count": capability_count,
        "avg_unique_capabilities_per_query": (sum(unique_counts) / len(unique_counts)) if unique_counts else 0.0,
        "avg_tool_calls_per_trajectory": (sum(call_counts) / len(call_counts)) if call_counts else 0.0,
        "multi_capability_query_ratio": multi / denominator,
        "unique_capabilities_ge_3_ratio": ge_3 / denominator,
        "unique_capabilities_ge_5_ratio": ge_5 / denominator,
    }
