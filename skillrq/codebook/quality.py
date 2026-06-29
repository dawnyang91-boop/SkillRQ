"""Code quality metrics for M3 code assignments."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


def build_quality_report(assignments: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in assignments:
        by_dataset[str(row["source_dataset"])].append(row)
    return {
        "overall": _quality_for_rows(assignments),
        "by_dataset": {dataset: _quality_for_rows(rows) for dataset, rows in sorted(by_dataset.items())},
    }


def _quality_for_rows(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    total = len(rows)
    semantic_counts = Counter(str(row["semantic_id"]) for row in rows)
    l1_counts = Counter(str(row["l1_code"]) for row in rows)
    return {
        "assignment_count": total,
        "unique_semantic_code_paths": len(semantic_counts),
        "unique_l1_codes": len(l1_counts),
        "code_purity": _weighted_majority_purity(rows, group_key="l2_code", label_key="category_label"),
        "code_usage_entropy": _normalized_entropy(semantic_counts),
        "category_alignment": _weighted_majority_purity(rows, group_key="l1_code", label_key="category_label"),
        "role_alignment": _weighted_majority_purity(rows, group_key="l3_code", label_key="role_label"),
        "code_collapse_rate": (max(semantic_counts.values()) / total) if total and semantic_counts else 0.0,
    }


def _weighted_majority_purity(
    rows: Sequence[Mapping[str, Any]],
    group_key: str,
    label_key: str,
) -> float:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        group = str(row.get(group_key) or "unknown")
        label = str(row.get(label_key) or "unknown")
        groups[group][label] += 1
    total = sum(sum(counts.values()) for counts in groups.values())
    if not total:
        return 0.0
    return sum(counts.most_common(1)[0][1] for counts in groups.values() if counts) / total


def _normalized_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(len(counts))
