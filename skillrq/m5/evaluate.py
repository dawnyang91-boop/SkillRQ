"""Evaluate M5 residual coverage predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..utils.io import read_jsonl, write_json


def evaluate_m5_predictions(
    prediction_path: Path,
    output_path: Path,
    top_ks: Sequence[int] = (5, 10, 20, 50, 100),
    set_metric_name: str = "tool_set_recall",
) -> Mapping[str, Any]:
    rows = list(read_jsonl(prediction_path))
    sums: dict[str, float] = {}
    evaluated = 0
    total_redundant_paths = 0
    total_paths = 0
    total_candidate_duplicates = 0
    total_candidates = 0
    step_gain_sums: dict[int, float] = {}
    step_counts: dict[int, int] = {}
    for row in rows:
        gold = set(str(item) for item in row.get("gold_ids") or [])
        if not gold:
            continue
        evaluated += 1
        residual_paths = list(row.get("residual_code_paths") or [])
        seen_path_ids = set()
        retrieved_ids = []
        covered = set()
        for path in residual_paths:
            path_key = str(path.get("semantic_id") or path.get("codes"))
            if path_key in seen_path_ids:
                total_redundant_paths += 1
            seen_path_ids.add(path_key)
            total_paths += 1
            candidates = [str(item.get("candidate_id")) for item in path.get("retrieved_capabilities") or []]
            new_hits = (set(candidates) & gold) - covered
            covered.update(set(candidates) & gold)
            step = int(path.get("step_index") or 0)
            step_gain_sums[step] = step_gain_sums.get(step, 0.0) + len(new_hits) / max(len(gold), 1)
            step_counts[step] = step_counts.get(step, 0) + 1
            retrieved_ids.extend(candidates)
        total_candidates += len(retrieved_ids)
        total_candidate_duplicates += len(retrieved_ids) - len(set(retrieved_ids))
        for k in top_ks:
            prefix = set(retrieved_ids[:k])
            recall = len(prefix & gold) / len(gold)
            sums[f"recall@{k}"] = sums.get(f"recall@{k}", 0.0) + recall
            sums[f"completeness@{k}"] = sums.get(f"completeness@{k}", 0.0) + float(gold.issubset(prefix))
            sums[f"{set_metric_name}@{k}"] = sums.get(f"{set_metric_name}@{k}", 0.0) + recall

    metrics: dict[str, Any] = {
        "prediction_path": str(prediction_path),
        "evaluated_queries": evaluated,
        "redundant_code_path_ratio": total_redundant_paths / max(total_paths, 1),
        "candidate_redundancy_ratio": total_candidate_duplicates / max(total_candidates, 1),
    }
    for step, gain_sum in sorted(step_gain_sums.items()):
        metrics[f"step_{step}_coverage_gain"] = gain_sum / max(step_counts.get(step, 0), 1)
    for key, value in sorted(sums.items()):
        metrics[key] = value / max(evaluated, 1)
    write_json(output_path, metrics)
    return metrics
