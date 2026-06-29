"""Build a sequence-aware evaluation view from M4 data."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

from ..utils.io import read_jsonl, write_json, write_jsonl


def build_sequence_eval_view(
    m4_data_root: Path,
    output_root: Path,
    sequence_dev_size: int = 2000,
    sequence_test_size: int = 5000,
    seed: int = 13,
) -> Mapping[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = list(read_jsonl(m4_data_root / "candidates.jsonl"))
    queries = [dict(row) for row in read_jsonl(m4_data_root / "queries.jsonl")]
    train_pairs = [dict(row) for row in read_jsonl(m4_data_root / "train_pairs.jsonl")]
    sequence_train_indices = [
        index
        for index, row in enumerate(queries)
        if row.get("split") == "train" and row.get("sequence_ids")
    ]
    rng = random.Random(seed)
    rng.shuffle(sequence_train_indices)
    sequence_dev = set(sequence_train_indices[:sequence_dev_size])
    sequence_test = set(sequence_train_indices[sequence_dev_size : sequence_dev_size + sequence_test_size])
    for index, row in enumerate(queries):
        if index in sequence_dev:
            row["split"] = "sequence_dev"
            row["source_split"] = f"{row.get('source_split') or 'train'}::sequence_dev"
        elif index in sequence_test:
            row["split"] = "sequence_test"
            row["source_split"] = f"{row.get('source_split') or 'train'}::sequence_test"
    split_by_query_id = {str(row["query_id"]): row["split"] for row in queries}
    for row in train_pairs:
        query_id = str(row.get("query_id"))
        if query_id in split_by_query_id:
            row["split"] = split_by_query_id[query_id]
    stats = {
        "m4_data_root": str(m4_data_root),
        "output_root": str(output_root),
        "seed": seed,
        "candidates": len(candidates),
        "queries": len(queries),
        "train_pairs": len(train_pairs),
        "sequence_train_queries_before_split": len(sequence_train_indices),
        "sequence_train_candidates": len(sequence_train_indices),
        "sequence_dev_queries": len(sequence_dev),
        "sequence_test_queries": len(sequence_test),
        "train_queries": sum(1 for row in queries if row.get("split") == "train"),
        "dev_queries": sum(1 for row in queries if row.get("split") == "dev"),
        "test_queries": sum(1 for row in queries if row.get("split") == "test"),
        "sequence_dev_size_requested": sequence_dev_size,
        "sequence_test_size_requested": sequence_test_size,
    }
    write_jsonl(output_root / "candidates.jsonl", candidates)
    write_jsonl(output_root / "queries.jsonl", queries)
    write_jsonl(output_root / "train_pairs.jsonl", train_pairs)
    write_json(output_root / "stats.json", stats)
    return stats
