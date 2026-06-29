"""Evaluate M7 reranked recommendations and sequence predictions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..utils.io import read_jsonl, write_json


def evaluate_reranked_predictions(
    prediction_path: Path,
    output_path: Path,
    top_ks: Sequence[int] = (5, 10, 20, 50, 100),
    set_metric_name: str = "tool_set_recall",
) -> Mapping[str, Any]:
    rows = list(read_jsonl(prediction_path))
    sums: dict[str, float] = {}
    evaluated = 0
    sequence_evaluated = 0
    first_tool_hits = 0
    transition_sum = 0.0
    kendall_sum = 0.0
    kendall_count = 0
    for row in rows:
        gold = [str(item) for item in row.get("gold_ids") or []]
        if not gold:
            continue
        evaluated += 1
        ranked = [str(item.get("candidate_id")) for item in row.get("reranked_capabilities") or []]
        gold_set = set(gold)
        for k in top_ks:
            top = ranked[:k]
            top_set = set(top)
            recall = len(top_set & gold_set) / len(gold_set)
            sums[f"recall@{k}"] = sums.get(f"recall@{k}", 0.0) + recall
            sums[f"{set_metric_name}@{k}"] = sums.get(f"{set_metric_name}@{k}", 0.0) + recall
            sums[f"completeness@{k}"] = sums.get(f"completeness@{k}", 0.0) + float(gold_set.issubset(top_set))
            sums[f"ndcg@{k}"] = sums.get(f"ndcg@{k}", 0.0) + _ndcg(top, gold_set, k)
            sums[f"mrr@{k}"] = sums.get(f"mrr@{k}", 0.0) + _mrr(top, gold_set)
        sequence = [str(item) for item in row.get("sequence_ids") or []]
        predicted_order = [str(item) for item in row.get("predicted_tool_order") or []]
        if sequence:
            sequence_evaluated += 1
            if predicted_order and predicted_order[0] == sequence[0]:
                first_tool_hits += 1
            transition_sum += _transition_accuracy(predicted_order, sequence)
            tau = _kendall_tau(predicted_order, sequence)
            if tau is not None:
                kendall_sum += tau
                kendall_count += 1

    metrics: dict[str, Any] = {
        "prediction_path": str(prediction_path),
        "evaluated_queries": evaluated,
        "sequence_evaluated_queries": sequence_evaluated,
        "first_tool_accuracy": first_tool_hits / max(sequence_evaluated, 1),
        "transition_accuracy": transition_sum / max(sequence_evaluated, 1),
        "kendall_tau": kendall_sum / max(kendall_count, 1),
    }
    for key, value in sorted(sums.items()):
        metrics[key] = value / max(evaluated, 1)
    write_json(output_path, metrics)
    return metrics


def _ndcg(ranked: Sequence[str], gold: set[str], k: int) -> float:
    dcg = 0.0
    for index, candidate_id in enumerate(ranked[:k], start=1):
        if candidate_id in gold:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _mrr(ranked: Sequence[str], gold: set[str]) -> float:
    for index, candidate_id in enumerate(ranked, start=1):
        if candidate_id in gold:
            return 1.0 / index
    return 0.0


def _transition_accuracy(predicted: Sequence[str], gold_sequence: Sequence[str]) -> float:
    gold_pairs = list(zip(gold_sequence, gold_sequence[1:]))
    if not gold_pairs:
        return 0.0
    predicted_pairs = set(zip(predicted, predicted[1:]))
    return sum(1 for pair in gold_pairs if pair in predicted_pairs) / len(gold_pairs)


def _kendall_tau(predicted: Sequence[str], gold_sequence: Sequence[str]) -> float | None:
    gold_positions = {candidate_id: index for index, candidate_id in enumerate(gold_sequence)}
    predicted_positions = {
        candidate_id: index
        for index, candidate_id in enumerate(predicted)
        if candidate_id in gold_positions
    }
    common = [candidate_id for candidate_id in gold_sequence if candidate_id in predicted_positions]
    if len(common) < 2:
        return None
    concordant = 0
    discordant = 0
    for left_index in range(len(common)):
        for right_index in range(left_index + 1, len(common)):
            left = common[left_index]
            right = common[right_index]
            gold_order = gold_positions[left] - gold_positions[right]
            predicted_order = predicted_positions[left] - predicted_positions[right]
            if gold_order * predicted_order > 0:
                concordant += 1
            elif gold_order * predicted_order < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0
