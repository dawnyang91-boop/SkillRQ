"""Codebook quality diagnostics for M3/M4 assignments."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..utils.io import read_jsonl, write_json, write_jsonl


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
LEVELS = ("l1", "l2", "l3", "l4")


def diagnose_codebook(
    m4_data_root: Path,
    output_root: Path,
    max_cases: int = 100,
) -> Mapping[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = {str(row["candidate_id"]): row for row in read_jsonl(m4_data_root / "candidates.jsonl")}
    queries = list(read_jsonl(m4_data_root / "queries.jsonl"))
    query_summary, query_cases = _query_gold_path_diagnostics(queries, candidates, max_cases=max_cases)
    group_summary, group_cases = _semantic_group_diagnostics(candidates, max_cases=max_cases)
    alignment_summary = _query_code_alignment_diagnostics(queries, candidates)
    result = {
        "m4_data_root": str(m4_data_root),
        "query_gold_path_diagnostics": query_summary,
        "semantic_group_diagnostics": group_summary,
        "query_code_alignment_diagnostics": alignment_summary,
    }
    write_json(output_root / "codebook_diagnostics.json", result)
    write_jsonl(output_root / "codebook_query_cases.jsonl", query_cases)
    write_jsonl(output_root / "codebook_mixed_path_cases.jsonl", group_cases)
    return result


def _query_gold_path_diagnostics(
    queries: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    max_cases: int,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    evaluated = 0
    gold_count_sum = 0
    unique_path_sum = 0
    pair_overlap_sum = 0.0
    pair_count = 0
    disjoint_pair_count = 0
    multi_path_queries = 0
    fully_disjoint_queries = 0
    cases = []
    for query in queries:
        gold_ids = [str(item) for item in query.get("gold_ids") or [] if str(item) in candidates]
        if not gold_ids:
            continue
        evaluated += 1
        gold_count_sum += len(gold_ids)
        paths = [list(candidates[candidate_id].get("code_path") or []) for candidate_id in gold_ids]
        semantic_ids = {str(candidates[candidate_id].get("semantic_id")) for candidate_id in gold_ids}
        unique_path_sum += len(semantic_ids)
        if len(semantic_ids) > 1:
            multi_path_queries += 1
        overlaps = []
        for left in range(len(paths)):
            for right in range(left + 1, len(paths)):
                overlap = _code_overlap(paths[left], paths[right])
                overlaps.append(overlap)
                pair_overlap_sum += overlap / 4.0
                pair_count += 1
                if overlap == 0:
                    disjoint_pair_count += 1
        if overlaps and all(overlap == 0 for overlap in overlaps):
            fully_disjoint_queries += 1
        if len(cases) < max_cases and (len(semantic_ids) > 2 or (overlaps and min(overlaps) == 0)):
            cases.append(
                {
                    "query_id": query.get("query_id"),
                    "query": query.get("query"),
                    "gold_ids": gold_ids,
                    "gold_semantic_ids": sorted(semantic_ids),
                    "gold_code_paths": paths,
                    "pair_level_overlaps": overlaps,
                }
            )
    return (
        {
            "evaluated_queries": evaluated,
            "avg_gold_count": gold_count_sum / max(evaluated, 1),
            "avg_unique_gold_semantic_paths": unique_path_sum / max(evaluated, 1),
            "multi_path_query_ratio": multi_path_queries / max(evaluated, 1),
            "avg_pair_code_overlap_ratio": pair_overlap_sum / max(pair_count, 1),
            "disjoint_gold_path_pair_ratio": disjoint_pair_count / max(pair_count, 1),
            "fully_disjoint_multi_gold_query_ratio": fully_disjoint_queries / max(evaluated, 1),
        },
        cases,
    )


def _semantic_group_diagnostics(
    candidates: Mapping[str, Mapping[str, Any]],
    max_cases: int,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates.values():
        groups[str(candidate.get("semantic_id"))].append(candidate)
    group_sizes = []
    category_purities = []
    source_purities = []
    role_purities = []
    mixed_cases = []
    for semantic_id, rows in groups.items():
        group_sizes.append(len(rows))
        category_purities.append(_majority_fraction(_metadata_value(row, "category") for row in rows))
        source_purities.append(_majority_fraction(str(row.get("source_dataset") or "") for row in rows))
        role_purities.append(_majority_fraction(str(row.get("role_hint") or "") for row in rows))
        if len(mixed_cases) < max_cases and len(rows) >= 5:
            category_counts = Counter(_metadata_value(row, "category") for row in rows)
            if _majority_fraction(category_counts.elements()) < 0.7:
                mixed_cases.append(
                    {
                        "semantic_id": semantic_id,
                        "size": len(rows),
                        "category_counts": dict(category_counts.most_common(10)),
                        "source_counts": dict(Counter(str(row.get("source_dataset") or "") for row in rows).most_common(10)),
                        "sample_candidate_ids": [row.get("candidate_id") for row in rows[:10]],
                        "sample_names": [row.get("name") for row in rows[:10]],
                    }
                )
    return (
        {
            "semantic_path_count": len(groups),
            "candidate_count": len(candidates),
            "avg_candidates_per_semantic_path": sum(group_sizes) / max(len(group_sizes), 1),
            "max_candidates_per_semantic_path": max(group_sizes) if group_sizes else 0,
            "avg_category_purity": sum(category_purities) / max(len(category_purities), 1),
            "avg_source_purity": sum(source_purities) / max(len(source_purities), 1),
            "avg_role_purity": sum(role_purities) / max(len(role_purities), 1),
        },
        mixed_cases,
    )


def _query_code_alignment_diagnostics(
    queries: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    level_hits = Counter()
    level_total = Counter()
    candidate_text_overlap_sum = 0.0
    evaluated_pairs = 0
    for query in queries:
        query_tokens = set(_tokens(str(query.get("query") or "")))
        for candidate_id in query.get("gold_ids") or []:
            candidate = candidates.get(str(candidate_id))
            if not candidate:
                continue
            evaluated_pairs += 1
            labels = candidate.get("labels") or {}
            for level in LEVELS:
                level_total[level] += 1
                label_tokens = set(_tokens(str(labels.get(level) or "")))
                if query_tokens & label_tokens:
                    level_hits[level] += 1
            candidate_tokens = set(_tokens(str(candidate.get("text") or "")))
            candidate_text_overlap_sum += len(query_tokens & candidate_tokens) / max(len(query_tokens), 1)
    result: dict[str, Any] = {
        "evaluated_query_candidate_pairs": evaluated_pairs,
        "avg_query_candidate_text_token_recall": candidate_text_overlap_sum / max(evaluated_pairs, 1),
    }
    for level in LEVELS:
        result[f"{level}_label_token_hit_rate"] = level_hits[level] / max(level_total[level], 1)
    return result


def _metadata_value(row: Mapping[str, Any], key: str) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get(key) or row.get(key) or "")


def _majority_fraction(values) -> float:
    rows = [str(value) for value in values if str(value)]
    if not rows:
        return 0.0
    counts = Counter(rows)
    return counts.most_common(1)[0][1] / len(rows)


def _code_overlap(left: Sequence[Any], right: Sequence[Any]) -> int:
    return sum(1 for a, b in zip(left, right) if str(a) == str(b))


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if token]
