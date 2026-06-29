"""Evaluate M4 code-retrieval predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..utils.io import read_jsonl, write_json


def evaluate_m4_predictions(
    prediction_path: Path,
    output_path: Path,
    top_ks: Sequence[int] = (1, 5, 10, 20, 50, 100),
    set_metric_name: str = "tool_set_recall",
) -> Mapping[str, Any]:
    rows = list(read_jsonl(prediction_path))
    sums: dict[str, float] = {}
    evaluated = 0
    candidate_pool_sizes = []
    for row in rows:
        gold = set(str(item) for item in row.get("gold_ids") or [])
        if not gold:
            continue
        evaluated += 1
        retrieved = [str(item.get("candidate_id")) for item in row.get("retrieved_capabilities") or []]
        candidate_pool_sizes.append(len(retrieved))
        for k in top_ks:
            prefix = set(retrieved[:k])
            recall = len(prefix & gold) / len(gold)
            sums[f"recall@{k}"] = sums.get(f"recall@{k}", 0.0) + recall
            sums[f"completeness@{k}"] = sums.get(f"completeness@{k}", 0.0) + float(gold.issubset(prefix))
            sums[f"{set_metric_name}@{k}"] = sums.get(f"{set_metric_name}@{k}", 0.0) + recall
            sums[f"recall_under_same_candidate_budget@{k}"] = sums.get(
                f"recall_under_same_candidate_budget@{k}", 0.0
            ) + recall
    metrics = {
        "prediction_path": str(prediction_path),
        "evaluated_queries": evaluated,
        "avg_candidate_pool_size": (
            sum(candidate_pool_sizes) / len(candidate_pool_sizes) if candidate_pool_sizes else 0.0
        ),
    }
    denominator = evaluated or 1
    for key, value in sorted(sums.items()):
        metrics[key] = value / denominator
    write_json(output_path, metrics)
    return metrics
