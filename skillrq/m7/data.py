"""Prepare M7 role-aware and sequence-aware reranking data."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .features import build_feature_dict, code_overlap, infer_stage, normalize_role
from ..utils.io import read_jsonl, write_json, write_jsonl


def prepare_m7_data(
    m4_data_root: Path,
    output_root: Path,
    negatives_per_positive: int = 2,
    limit_queries: int | None = None,
) -> Mapping[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = {str(row["candidate_id"]): row for row in read_jsonl(m4_data_root / "candidates.jsonl")}
    l1_index, source_l1_index = _build_indexes(candidates)
    examples = []
    pools = []
    query_count = 0
    positive_count = 0
    negative_count = 0
    sequence_count = 0

    for query in read_jsonl(m4_data_root / "queries.jsonl"):
        gold_ids = [str(item) for item in query.get("gold_ids") or [] if str(item) in candidates]
        if not gold_ids:
            continue
        query_text = str(query.get("query") or "")
        sequence_ids = [str(item) for item in query.get("sequence_ids") or []]
        sequence_positions = {candidate_id: index for index, candidate_id in enumerate(sequence_ids)}
        if sequence_ids:
            sequence_count += 1
        query_examples = []
        pool_ids: list[str] = []
        for candidate_id in gold_ids:
            candidate = candidates[candidate_id]
            position = sequence_positions.get(candidate_id)
            row = _make_example(
                query=query,
                candidate=candidate,
                label=1,
                sequence_position=position,
                sequence_length=len(sequence_ids) if sequence_ids else len(gold_ids),
                query_text=query_text,
                matched_levels=4,
                code_match_score=1.0,
                coverage_gain_score=1.0 / max(len(gold_ids), 1),
            )
            query_examples.append(row)
            pool_ids.append(candidate_id)
            positive_count += 1
            for negative_id in _select_negatives(candidate, query, gold_ids, candidates, l1_index, source_l1_index, negatives_per_positive):
                negative = candidates[negative_id]
                matched_levels = _best_gold_overlap(negative, gold_ids, candidates)
                query_examples.append(
                    _make_example(
                        query=query,
                        candidate=negative,
                        label=0,
                        sequence_position=None,
                        sequence_length=len(sequence_ids) if sequence_ids else len(gold_ids),
                        query_text=query_text,
                        matched_levels=matched_levels,
                        code_match_score=matched_levels / 4.0,
                        coverage_gain_score=0.0,
                    )
                )
                pool_ids.append(negative_id)
                negative_count += 1
        examples.extend(query_examples)
        pools.append(
            {
                "query_id": query["query_id"],
                "query": query_text,
                "source_dataset": query.get("source_dataset"),
                "split": query.get("split"),
                "gold_ids": gold_ids,
                "sequence_ids": sequence_ids,
                "candidate_pool_ids": _dedupe(pool_ids),
            }
        )
        query_count += 1
        if limit_queries is not None and query_count >= limit_queries:
            break

    stats = {
        "m4_data_root": str(m4_data_root),
        "output_root": str(output_root),
        "queries": query_count,
        "examples": len(examples),
        "positives": positive_count,
        "negatives": negative_count,
        "queries_with_sequence": sequence_count,
        "negatives_per_positive": negatives_per_positive,
    }
    write_jsonl(output_root / "rerank_examples.jsonl", examples)
    write_jsonl(output_root / "query_candidate_pools.jsonl", pools)
    write_json(output_root / "stats.json", stats)
    return stats


def _build_indexes(candidates: Mapping[str, Mapping[str, Any]]):
    l1_index: dict[str, list[str]] = defaultdict(list)
    source_l1_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for candidate_id, candidate in candidates.items():
        code_path = list(candidate.get("code_path") or [])
        l1 = str(code_path[0]) if code_path else "UNKNOWN"
        source = str(candidate.get("source_dataset") or "UNKNOWN")
        l1_index[l1].append(candidate_id)
        source_l1_index[(source, l1)].append(candidate_id)
    return l1_index, source_l1_index


def _make_example(
    query: Mapping[str, Any],
    candidate: Mapping[str, Any],
    label: int,
    sequence_position: int | None,
    sequence_length: int,
    query_text: str,
    matched_levels: int,
    code_match_score: float,
    coverage_gain_score: float,
) -> Mapping[str, Any]:
    role = normalize_role(candidate.get("role_hint"))
    stage = infer_stage(role, sequence_position, sequence_length)
    features = build_feature_dict(
        query=query_text,
        candidate=candidate,
        matched_levels=matched_levels,
        code_match_score=code_match_score,
        coverage_gain_score=coverage_gain_score,
        step_index=sequence_position,
    )
    return {
        "query_id": query["query_id"],
        "query": query_text,
        "split": query.get("split"),
        "source_dataset": query.get("source_dataset"),
        "candidate_id": candidate["candidate_id"],
        "candidate_name": candidate.get("name"),
        "candidate_text": candidate.get("text"),
        "code_explanation": candidate.get("code_explanation"),
        "candidate_schema": _candidate_schema_text(candidate),
        "semantic_id": candidate.get("semantic_id"),
        "code_path": candidate.get("code_path") or [],
        "matched_code_path": candidate.get("code_path") or [],
        "role_label": role,
        "stage_label": stage,
        "sequence_position": -1 if sequence_position is None else sequence_position,
        "order_score": 0.0 if sequence_position is None else 1.0 / (sequence_position + 1),
        "label": int(label),
        "features": features,
    }


def _select_negatives(
    positive: Mapping[str, Any],
    query: Mapping[str, Any],
    gold_ids: list[str],
    candidates: Mapping[str, Mapping[str, Any]],
    l1_index: Mapping[str, list[str]],
    source_l1_index: Mapping[tuple[str, str], list[str]],
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []
    code_path = list(positive.get("code_path") or [])
    l1 = str(code_path[0]) if code_path else "UNKNOWN"
    source = str(query.get("source_dataset") or positive.get("source_dataset") or "UNKNOWN")
    pool = [*source_l1_index.get((source, l1), []), *l1_index.get(l1, [])]
    gold = set(gold_ids)
    rows = []
    seen = set()
    for candidate_id in pool:
        if candidate_id in gold or candidate_id in seen:
            continue
        seen.add(candidate_id)
        candidate = candidates[candidate_id]
        rows.append((candidate_id, code_overlap(code_path, list(candidate.get("code_path") or []))))
        if len(rows) >= limit * 8:
            break
    rows.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return [candidate_id for candidate_id, _score in rows[:limit]]


def _best_gold_overlap(
    candidate: Mapping[str, Any],
    gold_ids: list[str],
    candidates: Mapping[str, Mapping[str, Any]],
) -> int:
    code_path = list(candidate.get("code_path") or [])
    return max((code_overlap(code_path, list(candidates[gold_id].get("code_path") or [])) for gold_id in gold_ids), default=0)


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    rows = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        rows.append(value)
    return rows


def _candidate_schema_text(candidate: Mapping[str, Any]) -> str:
    metadata = candidate.get("metadata") or {}
    labels = candidate.get("labels") or {}
    pieces = [
        metadata.get("api_schema"),
        metadata.get("parameters"),
        metadata.get("input_schema"),
        metadata.get("output_schema"),
        labels.get("l4") if isinstance(labels, Mapping) else None,
    ]
    return " ".join(str(item) for item in pieces if item)
