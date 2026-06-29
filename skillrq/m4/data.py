"""Prepare supervised query-to-code data for M4."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from ..config.schema import PathsConfig
from ..retrieval.text import capability_text, skill_text
from ..utils.io import read_jsonl, write_json, write_jsonl


CAPABILITY_TARGET = "capability"
SKILL_TARGET = "skill"


def prepare_m4_data(
    paths: PathsConfig,
    target: str,
    datasets: Sequence[str] | None = None,
    output_root: Path | None = None,
    limit_queries: int | None = None,
) -> Mapping[str, Any]:
    if target == CAPABILITY_TARGET:
        return _prepare_capability(paths, datasets or ("toolbench", "api_bank"), output_root, limit_queries)
    if target == SKILL_TARGET:
        return _prepare_skill(paths, datasets or ("skillret",), output_root, limit_queries)
    raise ValueError(f"Unsupported M4 target: {target}")


def _prepare_capability(
    paths: PathsConfig,
    datasets: Sequence[str],
    output_root: Path | None,
    limit_queries: int | None,
) -> Mapping[str, Any]:
    output_root = output_root or paths.processed_root / "m4" / "capability"
    assignments = _load_assignments(paths.capability_processed_root / "code_assignments.jsonl", datasets)
    capabilities = {
        str(row["capability_id"]): row
        for row in read_jsonl(paths.capability_processed_root / "capabilities.jsonl")
        if row.get("source_dataset") in datasets
    }
    candidates = []
    for candidate_id, assignment in assignments.items():
        capability = capabilities.get(candidate_id)
        if not capability:
            continue
        candidates.append(_candidate_row(candidate_id, assignment, capability_text(capability), capability))

    queries = []
    train_pairs = []
    for row in read_jsonl(paths.capability_processed_root / "capability_queries.jsonl"):
        if row.get("source_dataset") not in datasets:
            continue
        gold_ids = [str(item) for item in row.get("gold_capability_ids") or [] if str(item) in assignments]
        if not gold_ids:
            continue
        query_text = str(row.get("query") or row.get("final_answer") or "")
        if not query_text.strip():
            continue
        query = _query_row(
            query_id=str(row["query_id"]),
            query=query_text,
            source_dataset=str(row["source_dataset"]),
            source_split=str(row.get("source_split") or ""),
            gold_ids=gold_ids,
            assignments=assignments,
            sequence_ids=[str(item) for item in row.get("tool_call_sequence") or []],
        )
        queries.append(query)
        train_pairs.extend(_pair_rows(query, assignments))
        if limit_queries is not None and len(queries) >= limit_queries:
            break

    return _write_m4_data(output_root, "capability", datasets, candidates, queries, train_pairs)


def _prepare_skill(
    paths: PathsConfig,
    datasets: Sequence[str],
    output_root: Path | None,
    limit_queries: int | None,
) -> Mapping[str, Any]:
    output_root = output_root or paths.processed_root / "m4" / "skill"
    assignments = _load_assignments(paths.processed_root / "skill" / "code_assignments.jsonl", datasets)
    skills = {
        str(row["skill_id"]): row
        for row in read_jsonl(paths.processed_root / "skills.jsonl")
        if "skillret" in datasets and row.get("source_dataset") == "skillret"
    }
    candidates = []
    for candidate_id, assignment in assignments.items():
        skill = skills.get(candidate_id)
        if not skill:
            continue
        candidates.append(_candidate_row(candidate_id, assignment, skill_text(skill), skill))

    queries = []
    train_pairs = []
    for row in read_jsonl(paths.processed_root / "queries.jsonl"):
        if row.get("source_dataset") not in datasets:
            continue
        gold_ids = [str(item) for item in row.get("gold_skill_ids") or [] if str(item) in assignments]
        if not gold_ids:
            continue
        query = _query_row(
            query_id=str(row["query_id"]),
            query=str(row.get("query") or ""),
            source_dataset=str(row["source_dataset"]),
            source_split=str(row.get("source_split") or ""),
            gold_ids=gold_ids,
            assignments=assignments,
            sequence_ids=[],
        )
        queries.append(query)
        train_pairs.extend(_pair_rows(query, assignments))
        if limit_queries is not None and len(queries) >= limit_queries:
            break

    return _write_m4_data(output_root, "skill", datasets, candidates, queries, train_pairs)


def _load_assignments(path: Path, datasets: Sequence[str]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(row["object_id"]): row
        for row in read_jsonl(path)
        if row.get("source_dataset") in datasets
    }


def _candidate_row(
    candidate_id: str,
    assignment: Mapping[str, Any],
    text: str,
    raw: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "candidate_id": candidate_id,
        "source_dataset": assignment["source_dataset"],
        "name": assignment.get("name"),
        "text": text,
        "code_path": list(assignment["code_path"]),
        "semantic_id": assignment["semantic_id"],
        "role_hint": assignment.get("l3_label"),
        "code_explanation": assignment.get("code_explanation"),
        "labels": {
            "l1": assignment.get("l1_label"),
            "l2": assignment.get("l2_label"),
            "l3": assignment.get("l3_label"),
            "l4": assignment.get("l4_label"),
        },
        "metadata": {
            "source_dataset": raw.get("source_dataset"),
            "source_id": raw.get("source_capability_id") or raw.get("source_skill_id"),
            "category": raw.get("category") or raw.get("domain_label"),
            "capability_type": raw.get("capability_type"),
        },
    }


def _query_row(
    *,
    query_id: str,
    query: str,
    source_dataset: str,
    source_split: str,
    gold_ids: Sequence[str],
    assignments: Mapping[str, Mapping[str, Any]],
    sequence_ids: Sequence[str],
) -> Mapping[str, Any]:
    gold_code_paths = []
    seen = set()
    for candidate_id in gold_ids:
        assignment = assignments[candidate_id]
        semantic_id = str(assignment["semantic_id"])
        if semantic_id in seen:
            continue
        seen.add(semantic_id)
        gold_code_paths.append(
            {
                "candidate_id": candidate_id,
                "semantic_id": semantic_id,
                "codes": list(assignment["code_path"]),
                "role_hint": assignment.get("l3_label"),
            }
        )
    return {
        "query_id": query_id,
        "query": query,
        "source_dataset": source_dataset,
        "source_split": source_split,
        "split": _canonical_split(source_split),
        "gold_ids": list(dict.fromkeys(gold_ids)),
        "gold_code_paths": gold_code_paths,
        "sequence_ids": list(sequence_ids),
    }


def _pair_rows(query: Mapping[str, Any], assignments: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = []
    for candidate_id in query["gold_ids"]:
        assignment = assignments[candidate_id]
        rows.append(
            {
                "query_id": query["query_id"],
                "query": query["query"],
                "split": query["split"],
                "source_dataset": query["source_dataset"],
                "candidate_id": candidate_id,
                "semantic_id": assignment["semantic_id"],
                "code_path": list(assignment["code_path"]),
                "role_hint": assignment.get("l3_label"),
            }
        )
    return rows


def _canonical_split(source_split: str) -> str:
    text = str(source_split or "").lower()
    if "test" in text:
        return "test"
    if "dev" in text or "valid" in text:
        return "dev"
    return "train"


def _write_m4_data(
    output_root: Path,
    target: str,
    datasets: Sequence[str],
    candidates: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    train_pairs: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = sorted(candidates, key=lambda row: row["candidate_id"])
    queries = sorted(queries, key=lambda row: row["query_id"])
    train_pairs = sorted(train_pairs, key=lambda row: (row["query_id"], row["candidate_id"]))
    stats = {
        "target": target,
        "datasets": list(datasets),
        "candidates": len(candidates),
        "queries": len(queries),
        "train_pairs": len(train_pairs),
        "train_queries": sum(1 for row in queries if row["split"] == "train"),
        "dev_queries": sum(1 for row in queries if row["split"] == "dev"),
        "test_queries": sum(1 for row in queries if row["split"] == "test"),
        "avg_gold_ids_per_query": (
            sum(len(row["gold_ids"]) for row in queries) / len(queries)
            if queries
            else 0.0
        ),
        "avg_gold_code_paths_per_query": (
            sum(len(row["gold_code_paths"]) for row in queries) / len(queries)
            if queries
            else 0.0
        ),
    }
    write_jsonl(output_root / "candidates.jsonl", candidates)
    write_jsonl(output_root / "queries.jsonl", queries)
    write_jsonl(output_root / "train_pairs.jsonl", train_pairs)
    write_json(output_root / "stats.json", stats)
    return stats
