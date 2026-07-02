"""Evaluate simulated LLM tool-call plans."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..utils.io import read_jsonl, write_json


def evaluate_tool_call_plans(
    plan_path: Path,
    output_path: Path,
    top_ks: Sequence[int] = (1, 3, 5, 10),
) -> Mapping[str, Any]:
    rows = list(read_jsonl(plan_path))
    evaluated = 0
    sequence_evaluated = 0
    sums: dict[str, float] = {}
    invalid_tool_count = 0
    total_tool_calls = 0
    grounded_tool_calls = 0
    first_tool_hits = 0
    transition_sum = 0.0
    for row in rows:
        gold_ids = [str(item) for item in row.get("gold_ids") or []]
        if not gold_ids:
            continue
        evaluated += 1
        allowed = {str(item) for item in row.get("allowed_candidate_ids") or []}
        predicted = [str(call.get("candidate_id")) for call in row.get("tool_calls") or [] if call.get("candidate_id")]
        total_tool_calls += len(predicted)
        invalid_tool_count += sum(1 for candidate_id in predicted if candidate_id not in allowed)
        grounded_tool_calls += sum(1 for candidate_id in predicted if candidate_id in allowed)
        gold = set(gold_ids)
        for k in top_ks:
            prefix = set(predicted[:k])
            recall = len(prefix & gold) / len(gold)
            sums[f"tool_set_recall@{k}"] = sums.get(f"tool_set_recall@{k}", 0.0) + recall
            sums[f"completeness@{k}"] = sums.get(f"completeness@{k}", 0.0) + float(gold.issubset(prefix))
        sequence_ids = [str(item) for item in row.get("sequence_ids") or []]
        if sequence_ids:
            sequence_evaluated += 1
            first_tool_hits += int(bool(predicted) and predicted[0] == sequence_ids[0])
            transition_sum += _transition_accuracy(predicted, sequence_ids)

    metrics: dict[str, Any] = {
        "plan_path": str(plan_path),
        "evaluated_queries": evaluated,
        "sequence_evaluated_queries": sequence_evaluated,
        "invalid_tool_rate": invalid_tool_count / max(total_tool_calls, 1),
        "prompt_grounding_rate": grounded_tool_calls / max(total_tool_calls, 1),
        "avg_tool_calls": total_tool_calls / max(evaluated, 1),
        "first_tool_accuracy": first_tool_hits / max(sequence_evaluated, 1),
        "transition_accuracy": transition_sum / max(sequence_evaluated, 1),
    }
    for key, value in sorted(sums.items()):
        metrics[key] = value / max(evaluated, 1)
    write_json(output_path, metrics)
    return metrics


def _transition_accuracy(predicted: Sequence[str], gold_sequence: Sequence[str]) -> float:
    if len(gold_sequence) < 2:
        return 1.0
    gold_edges = {(gold_sequence[index], gold_sequence[index + 1]) for index in range(len(gold_sequence) - 1)}
    predicted_edges = {(predicted[index], predicted[index + 1]) for index in range(len(predicted) - 1)}
    return len(gold_edges & predicted_edges) / len(gold_edges)
