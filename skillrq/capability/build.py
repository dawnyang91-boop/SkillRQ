"""Build canonical capability recommendation files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from .loaders.api_bank import convert_api_bank
from .loaders.toolbench import convert_toolbench, iter_toolbench_answer_tree_queries, runtime_capability_from_id
from .schema import capability_qrels, capability_sequence_rows
from ..config.schema import PathsConfig
from ..utils.io import append_jsonl, read_jsonl, write_json, write_jsonl


def build_capability_processed_data(
    paths: PathsConfig,
    dataset: str = "all",
    output_root: Path | None = None,
    include_answer_trees: bool = True,
    limit_tools: int | None = None,
    limit_queries: int | None = None,
) -> Mapping[str, Any]:
    if include_answer_trees and dataset in {"toolbench", "all"} and limit_queries is None:
        return _build_capability_processed_data_streaming(
            paths,
            dataset=dataset,
            output_root=output_root,
            limit_tools=limit_tools,
        )

    raw_root = paths.capability_raw_root
    output_root = output_root or paths.capability_processed_root

    chunks: List[Mapping[str, List[Mapping[str, Any]]]] = []
    if dataset in {"toolbench", "all"}:
        chunks.append(
            convert_toolbench(
                raw_root,
                include_answer_trees=include_answer_trees,
                limit_tools=limit_tools,
                limit_queries=limit_queries,
            )
        )
    if dataset in {"api_bank", "all"}:
        chunks.append(convert_api_bank(raw_root))
    if dataset not in {"toolbench", "api_bank", "all"}:
        raise ValueError(f"Unsupported capability dataset: {dataset}")

    capabilities_by_id: Dict[str, Mapping[str, Any]] = {}
    queries: List[Mapping[str, Any]] = []
    for chunk in chunks:
        for capability in chunk["capabilities"]:
            capabilities_by_id[capability["capability_id"]] = capability
        queries.extend(chunk["queries"])

    capabilities = sorted(capabilities_by_id.values(), key=lambda row: row["capability_id"])
    queries = sorted(queries, key=lambda row: row["query_id"])
    qrels = sorted(
        [qrel for query in queries for qrel in capability_qrels(query)],
        key=lambda row: (row["query_id"], row["capability_id"]),
    )
    sequences = sorted(
        [row for query in queries for row in capability_sequence_rows(query)],
        key=lambda row: (row["query_id"], row["step_index"]),
    )
    stats = _build_stats(capabilities, queries, qrels, sequences, dataset)

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "capabilities.jsonl", capabilities)
    write_jsonl(output_root / "capability_queries.jsonl", queries)
    write_jsonl(output_root / "capability_qrels.jsonl", qrels)
    write_jsonl(output_root / "capability_sequences.jsonl", sequences)
    write_json(output_root / "capability_stats.json", stats)
    return stats


def _build_capability_processed_data_streaming(
    paths: PathsConfig,
    dataset: str,
    output_root: Path | None,
    limit_tools: int | None,
) -> Mapping[str, Any]:
    raw_root = paths.capability_raw_root
    output_root = output_root or paths.capability_processed_root
    output_root.mkdir(parents=True, exist_ok=True)

    chunks: List[Mapping[str, List[Mapping[str, Any]]]] = []
    if dataset in {"toolbench", "all"}:
        chunks.append(
            convert_toolbench(
                raw_root,
                include_answer_trees=False,
                limit_tools=limit_tools,
                limit_queries=None,
            )
        )
    if dataset in {"api_bank", "all"}:
        chunks.append(convert_api_bank(raw_root))

    capabilities_by_id: Dict[str, Mapping[str, Any]] = {}
    queries: List[Mapping[str, Any]] = []
    for chunk in chunks:
        for capability in chunk["capabilities"]:
            capabilities_by_id[capability["capability_id"]] = capability
        queries.extend(chunk["queries"])

    queries = sorted(queries, key=lambda row: row["query_id"])
    qrels = sorted(
        [qrel for query in queries for qrel in capability_qrels(query)],
        key=lambda row: (row["query_id"], row["capability_id"]),
    )
    sequences = sorted(
        [row for query in queries for row in capability_sequence_rows(query)],
        key=lambda row: (row["query_id"], row["step_index"]),
    )

    write_jsonl(output_root / "capability_queries.jsonl", queries)
    write_jsonl(output_root / "capability_qrels.jsonl", qrels)
    write_jsonl(output_root / "capability_sequences.jsonl", sequences)

    if dataset in {"toolbench", "all"}:
        for query in iter_toolbench_answer_tree_queries(raw_root):
            for capability_id in query.get("tool_call_sequence") or []:
                if capability_id not in capabilities_by_id:
                    capabilities_by_id[capability_id] = runtime_capability_from_id(capability_id)
            append_jsonl(output_root / "capability_queries.jsonl", [query])
            append_jsonl(output_root / "capability_qrels.jsonl", capability_qrels(query))
            append_jsonl(output_root / "capability_sequences.jsonl", capability_sequence_rows(query))

    capabilities = sorted(capabilities_by_id.values(), key=lambda row: row["capability_id"])
    write_jsonl(output_root / "capabilities.jsonl", capabilities)
    stats = _build_stats_from_output(output_root, dataset=dataset, capabilities_count=len(capabilities))
    write_json(output_root / "capability_stats.json", stats)
    return stats


def _build_stats(
    capabilities: List[Mapping[str, Any]],
    queries: List[Mapping[str, Any]],
    qrels: List[Mapping[str, Any]],
    sequences: List[Mapping[str, Any]],
    dataset: str,
) -> Mapping[str, Any]:
    unique_counts = [len(query.get("gold_capability_ids") or []) for query in queries]
    call_counts = [len(query.get("tool_call_sequence") or []) for query in queries]
    by_dataset: Dict[str, int] = {}
    for query in queries:
        by_dataset[str(query["source_dataset"])] = by_dataset.get(str(query["source_dataset"]), 0) + 1
    return {
        "dataset": dataset,
        "capabilities": len(capabilities),
        "queries": len(queries),
        "qrels": len(qrels),
        "sequences": len(sequences),
        "queries_by_dataset": by_dataset,
        "min_unique_tools_per_query": min(unique_counts) if unique_counts else 0,
        "max_unique_tools_per_query": max(unique_counts) if unique_counts else 0,
        "avg_unique_tools_per_query": (sum(unique_counts) / len(unique_counts)) if unique_counts else 0.0,
        "min_tool_calls_per_trajectory": min(call_counts) if call_counts else 0,
        "max_tool_calls_per_trajectory": max(call_counts) if call_counts else 0,
        "avg_tool_calls_per_trajectory": (sum(call_counts) / len(call_counts)) if call_counts else 0.0,
    }


def _build_stats_from_output(output_root: Path, dataset: str, capabilities_count: int) -> Mapping[str, Any]:
    queries_by_dataset: Dict[str, int] = {}
    unique_counts: List[int] = []
    call_counts: List[int] = []
    for query in read_jsonl(output_root / "capability_queries.jsonl"):
        queries_by_dataset[str(query["source_dataset"])] = queries_by_dataset.get(str(query["source_dataset"]), 0) + 1
        unique_counts.append(len(query.get("gold_capability_ids") or []))
        call_counts.append(len(query.get("tool_call_sequence") or []))
    qrels_count = _count_lines(output_root / "capability_qrels.jsonl")
    sequences_count = _count_lines(output_root / "capability_sequences.jsonl")
    return {
        "dataset": dataset,
        "capabilities": capabilities_count,
        "queries": len(unique_counts),
        "qrels": qrels_count,
        "sequences": sequences_count,
        "queries_by_dataset": queries_by_dataset,
        "min_unique_tools_per_query": min(unique_counts) if unique_counts else 0,
        "max_unique_tools_per_query": max(unique_counts) if unique_counts else 0,
        "avg_unique_tools_per_query": (sum(unique_counts) / len(unique_counts)) if unique_counts else 0.0,
        "min_tool_calls_per_trajectory": min(call_counts) if call_counts else 0,
        "max_tool_calls_per_trajectory": max(call_counts) if call_counts else 0,
        "avg_tool_calls_per_trajectory": (sum(call_counts) / len(call_counts)) if call_counts else 0.0,
    }


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)
