"""Negative sampling diagnostics for M7 reranker data."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from ..utils.io import read_jsonl, write_json, write_jsonl


FEATURE_KEYS = (
    "code_match_score",
    "matched_levels",
    "text_overlap_score",
    "schema_evidence_score",
    "parameter_compatibility_score",
    "role_compatibility_score",
    "coverage_gain_score",
    "first_tool_prior",
    "transition_prior",
    "generic_penalty",
    "constraint_violation_penalty",
)


def diagnose_negative_sampling(
    m7_data_root: Path,
    output_root: Path,
    max_cases: int = 100,
) -> Mapping[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    examples = list(read_jsonl(m7_data_root / "rerank_examples.jsonl"))
    positives = [row for row in examples if int(row.get("label") or 0) == 1]
    negatives = [row for row in examples if int(row.get("label") or 0) == 0]
    per_query = defaultdict(lambda: Counter())
    for row in examples:
        per_query[str(row.get("query_id"))]["positive" if int(row.get("label") or 0) == 1 else "negative"] += 1
    feature_summary = {
        key: {
            "positive_mean": _feature_mean(positives, key),
            "negative_mean": _feature_mean(negatives, key),
            "gap_positive_minus_negative": _feature_mean(positives, key) - _feature_mean(negatives, key),
        }
        for key in FEATURE_KEYS
    }
    negative_overlap_distribution = Counter(
        round(float((row.get("features") or {}).get("matched_levels") or 0.0), 2)
        for row in negatives
    )
    hard_cases = sorted(
        negatives,
        key=lambda row: (
            float((row.get("features") or {}).get("code_match_score") or 0.0),
            float((row.get("features") or {}).get("text_overlap_score") or 0.0),
        ),
        reverse=True,
    )[:max_cases]
    cases = [
        {
            "query_id": row.get("query_id"),
            "query": row.get("query"),
            "candidate_id": row.get("candidate_id"),
            "candidate_name": row.get("candidate_name"),
            "semantic_id": row.get("semantic_id"),
            "role_label": row.get("role_label"),
            "features": row.get("features"),
        }
        for row in hard_cases
    ]
    result = {
        "m7_data_root": str(m7_data_root),
        "examples": len(examples),
        "positives": len(positives),
        "negatives": len(negatives),
        "negative_to_positive_ratio": len(negatives) / max(len(positives), 1),
        "avg_positives_per_query": sum(value["positive"] for value in per_query.values()) / max(len(per_query), 1),
        "avg_negatives_per_query": sum(value["negative"] for value in per_query.values()) / max(len(per_query), 1),
        "feature_summary": feature_summary,
        "negative_matched_levels_distribution": {
            str(key): value for key, value in sorted(negative_overlap_distribution.items())
        },
    }
    write_json(output_root / "negative_sampling_diagnostics.json", result)
    write_jsonl(output_root / "hard_negative_cases.jsonl", cases)
    return result


def _feature_mean(rows: list[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float((row.get("features") or {}).get(key) or 0.0) for row in rows) / len(rows)
