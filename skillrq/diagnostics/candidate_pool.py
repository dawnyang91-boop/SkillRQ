"""Candidate-pool upper-bound diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .pools import flatten_candidate_ids, gold_ids, load_prediction_rows, unique_in_order
from ..utils.io import write_json, write_jsonl


def diagnose_candidate_pool(
    prediction_paths: Mapping[str, Path],
    output_root: Path,
    top_ks: Sequence[int] = (5, 10, 20, 50, 100),
    max_cases: int = 50,
) -> Mapping[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    all_cases = []
    for name, path in prediction_paths.items():
        rows = load_prediction_rows(path)
        if not rows:
            summaries[name] = {"prediction_path": str(path), "available": False}
            continue
        summary, cases = _diagnose_rows(name, path, rows, top_ks=top_ks, max_cases=max_cases)
        summaries[name] = summary
        all_cases.extend(cases)
    result = {"candidate_pool_upper_bounds": summaries}
    write_json(output_root / "candidate_pool_upper_bounds.json", result)
    write_jsonl(output_root / "candidate_pool_failure_cases.jsonl", all_cases)
    return result


def _diagnose_rows(
    name: str,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    top_ks: Sequence[int],
    max_cases: int,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    sums: dict[str, float] = {}
    evaluated = 0
    pool_sizes = []
    unique_pool_sizes = []
    full_pool_recall_sum = 0.0
    full_pool_completeness_sum = 0.0
    cases = []
    for row in rows:
        gold = set(gold_ids(row))
        if not gold:
            continue
        evaluated += 1
        ranked = flatten_candidate_ids(row)
        unique_pool = unique_in_order(ranked)
        pool_sizes.append(len(ranked))
        unique_pool_sizes.append(len(unique_pool))
        covered = set(unique_pool) & gold
        full_pool_recall = len(covered) / len(gold)
        full_pool_recall_sum += full_pool_recall
        full_pool_completeness_sum += float(gold.issubset(set(unique_pool)))
        for k in top_ks:
            observed_top = set(ranked[:k])
            observed_recall = len(observed_top & gold) / len(gold)
            oracle_hits = min(len(covered), k)
            oracle_recall = oracle_hits / len(gold)
            oracle_completeness = float(len(covered) == len(gold) and len(gold) <= k)
            sums[f"observed_recall@{k}"] = sums.get(f"observed_recall@{k}", 0.0) + observed_recall
            sums[f"observed_completeness@{k}"] = sums.get(f"observed_completeness@{k}", 0.0) + float(gold.issubset(observed_top))
            sums[f"oracle_recall@{k}"] = sums.get(f"oracle_recall@{k}", 0.0) + oracle_recall
            sums[f"oracle_completeness@{k}"] = sums.get(f"oracle_completeness@{k}", 0.0) + oracle_completeness
        if full_pool_recall < 1.0 and len(cases) < max_cases:
            cases.append(
                {
                    "source": name,
                    "query_id": row.get("query_id"),
                    "query": row.get("query"),
                    "gold_ids": sorted(gold),
                    "covered_gold_ids": sorted(covered),
                    "missing_gold_ids": sorted(gold - covered),
                    "candidate_pool_size": len(ranked),
                    "unique_candidate_pool_size": len(unique_pool),
                    "full_pool_recall": full_pool_recall,
                }
            )
    denominator = evaluated or 1
    summary: dict[str, Any] = {
        "prediction_path": str(path),
        "available": True,
        "evaluated_queries": evaluated,
        "avg_candidate_pool_size": sum(pool_sizes) / max(len(pool_sizes), 1),
        "avg_unique_candidate_pool_size": sum(unique_pool_sizes) / max(len(unique_pool_sizes), 1),
        "full_pool_recall": full_pool_recall_sum / denominator,
        "full_pool_completeness": full_pool_completeness_sum / denominator,
    }
    for key, value in sorted(sums.items()):
        summary[key] = value / denominator
    return summary, cases
