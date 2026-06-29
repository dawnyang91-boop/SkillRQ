"""Multi-positive and sequence supervision diagnostics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..utils.io import read_jsonl, write_json


def diagnose_multi_positive(
    m4_data_root: Path,
    output_root: Path,
) -> Mapping[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    queries = list(read_jsonl(m4_data_root / "queries.jsonl"))
    gold_counts = Counter()
    path_counts = Counter()
    sequence_counts = Counter()
    source_counts = Counter()
    split_counts = Counter()
    total = 0
    multi_gold = 0
    multi_path = 0
    with_sequence = 0
    sequence_has_all_gold = 0
    for query in queries:
        total += 1
        gold_ids = [str(item) for item in query.get("gold_ids") or []]
        semantic_ids = {
            str(item.get("semantic_id"))
            for item in query.get("gold_code_paths") or []
            if item.get("semantic_id")
        }
        sequence_ids = [str(item) for item in query.get("sequence_ids") or []]
        gold_counts[len(gold_ids)] += 1
        path_counts[len(semantic_ids)] += 1
        sequence_counts[len(sequence_ids)] += 1
        source_counts[str(query.get("source_dataset") or "")] += 1
        split_counts[str(query.get("split") or "")] += 1
        multi_gold += int(len(gold_ids) > 1)
        multi_path += int(len(semantic_ids) > 1)
        with_sequence += int(bool(sequence_ids))
        if sequence_ids and set(gold_ids).issubset(set(sequence_ids)):
            sequence_has_all_gold += 1
    result = {
        "m4_data_root": str(m4_data_root),
        "queries": total,
        "multi_gold_query_ratio": multi_gold / max(total, 1),
        "multi_code_path_query_ratio": multi_path / max(total, 1),
        "sequence_query_ratio": with_sequence / max(total, 1),
        "sequence_contains_all_gold_ratio": sequence_has_all_gold / max(with_sequence, 1),
        "gold_count_distribution": _counter_to_dict(gold_counts),
        "gold_semantic_path_count_distribution": _counter_to_dict(path_counts),
        "sequence_length_distribution": _counter_to_dict(sequence_counts),
        "source_distribution": dict(source_counts.most_common()),
        "split_distribution": dict(split_counts.most_common()),
    }
    write_json(output_root / "multi_positive_diagnostics.json", result)
    return result


def _counter_to_dict(counter: Counter[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(counter.items())}
