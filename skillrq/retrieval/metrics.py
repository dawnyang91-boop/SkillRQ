"""Unified retrieval metrics for tool/API and skill recommendation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .types import Query


def evaluate_predictions(
    queries: Sequence[Query],
    predictions: Mapping[str, Sequence[str]],
    top_ks: Sequence[int],
    task_type: str,
) -> dict[str, Any]:
    top_ks = sorted(set(int(k) for k in top_ks if int(k) > 0))
    metrics: dict[str, Any] = {
        "task_type": task_type,
        "query_count": len(queries),
        "evaluated_query_count": 0,
        "sequence_query_count": 0,
    }
    sums: dict[str, float] = {}
    sequence_sums: dict[str, float] = {}

    for query in queries:
        gold = list(dict.fromkeys(str(item) for item in query.gold_ids))
        if not gold:
            continue
        metrics["evaluated_query_count"] += 1
        ranked = [str(item) for item in predictions.get(query.query_id, [])]
        gold_set = set(gold)
        for k in top_ks:
            prefix = ranked[:k]
            hits = [candidate_id for candidate_id in prefix if candidate_id in gold_set]
            sums[f"recall@{k}"] = sums.get(f"recall@{k}", 0.0) + (len(set(hits)) / len(gold_set))
            sums[f"ndcg@{k}"] = sums.get(f"ndcg@{k}", 0.0) + _ndcg(prefix, gold_set, k)
            sums[f"mrr@{k}"] = sums.get(f"mrr@{k}", 0.0) + _mrr(prefix, gold_set)
            sums[f"completeness@{k}"] = sums.get(f"completeness@{k}", 0.0) + float(gold_set.issubset(set(prefix)))
            set_key = "tool_set_recall" if task_type == "tool" else "skill_set_recall"
            sums[f"{set_key}@{k}"] = sums.get(f"{set_key}@{k}", 0.0) + (len(set(hits)) / len(gold_set))

        sequence = [item for item in query.sequence_ids if item]
        if sequence:
            metrics["sequence_query_count"] += 1
            sequence_sums["first_tool_accuracy"] = sequence_sums.get("first_tool_accuracy", 0.0) + float(
                bool(ranked) and ranked[0] == sequence[0]
            )
            sequence_sums["transition_accuracy"] = sequence_sums.get("transition_accuracy", 0.0) + _transition_accuracy(
                ranked, sequence
            )
            sequence_sums["kendall_tau"] = sequence_sums.get("kendall_tau", 0.0) + _kendall_tau(ranked, sequence)

    denominator = metrics["evaluated_query_count"] or 1
    for key, value in sorted(sums.items()):
        metrics[key] = value / denominator

    sequence_denominator = metrics["sequence_query_count"] or 0
    if sequence_denominator:
        for key, value in sorted(sequence_sums.items()):
            metrics[key] = value / sequence_denominator
    else:
        metrics["first_tool_accuracy"] = None
        metrics["transition_accuracy"] = None
        metrics["kendall_tau"] = None
    return metrics


def _ndcg(prefix: Sequence[str], gold_set: set[str], k: int) -> float:
    dcg = 0.0
    for rank, candidate_id in enumerate(prefix[:k], start=1):
        if candidate_id in gold_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gold_set), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def _mrr(prefix: Sequence[str], gold_set: set[str]) -> float:
    for rank, candidate_id in enumerate(prefix, start=1):
        if candidate_id in gold_set:
            return 1.0 / rank
    return 0.0


def _transition_accuracy(ranked: Sequence[str], sequence: Sequence[str]) -> float:
    if len(sequence) < 2:
        return 1.0
    ranks = _rank_map(ranked)
    correct = 0
    total = 0
    missing_rank = len(ranked) + len(sequence) + 1
    for left, right in zip(sequence, sequence[1:]):
        if left == right:
            continue
        total += 1
        if ranks.get(left, missing_rank) < ranks.get(right, missing_rank):
            correct += 1
    return correct / total if total else 1.0


def _kendall_tau(ranked: Sequence[str], sequence: Sequence[str]) -> float:
    ordered_gold = list(dict.fromkeys(sequence))
    if len(ordered_gold) < 2:
        return 1.0
    ranks = _rank_map(ranked)
    missing_rank = len(ranked) + len(ordered_gold) + 1
    concordant = 0
    discordant = 0
    for left_index, left in enumerate(ordered_gold):
        for right in ordered_gold[left_index + 1 :]:
            if ranks.get(left, missing_rank) <= ranks.get(right, missing_rank):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return ((concordant - discordant) / total) if total else 1.0


def _rank_map(ranked: Sequence[str]) -> dict[str, int]:
    return {candidate_id: rank for rank, candidate_id in enumerate(ranked)}
