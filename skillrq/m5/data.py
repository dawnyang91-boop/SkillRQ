"""Prepare residual coverage supervision data for M5."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..utils.io import read_jsonl, write_json, write_jsonl


def prepare_m5_data(
    m4_data_root: Path,
    output_root: Path,
    max_steps: int = 6,
    limit_queries: int | None = None,
) -> Mapping[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = {str(row["candidate_id"]): row for row in read_jsonl(m4_data_root / "candidates.jsonl")}
    queries = list(read_jsonl(m4_data_root / "queries.jsonl"))
    examples = []
    query_plans = []
    for query in queries:
        plan = _build_residual_plan(query, candidates, max_steps=max_steps)
        if not plan:
            continue
        query_plans.append(
            {
                "query_id": query["query_id"],
                "query": query["query"],
                "source_dataset": query.get("source_dataset"),
                "split": query.get("split"),
                "gold_ids": query.get("gold_ids") or [],
                "residual_plan": plan,
            }
        )
        examples.extend(_plan_to_examples(query, plan))
        if limit_queries is not None and len(query_plans) >= limit_queries:
            break

    stats = {
        "m4_data_root": str(m4_data_root),
        "output_root": str(output_root),
        "queries": len(query_plans),
        "residual_examples": len(examples),
        "max_steps": max_steps,
        "avg_steps_per_query": (len(examples) / len(query_plans)) if query_plans else 0.0,
        "avg_coverage_gain": (
            sum(float(row["coverage_gain"]) for row in examples) / len(examples)
            if examples
            else 0.0
        ),
    }
    write_jsonl(output_root / "residual_examples.jsonl", examples)
    write_jsonl(output_root / "query_residual_plans.jsonl", query_plans)
    write_json(output_root / "stats.json", stats)
    return stats


def _build_residual_plan(
    query: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    max_steps: int,
) -> list[Mapping[str, Any]]:
    gold_ids = [str(item) for item in query.get("gold_ids") or [] if str(item) in candidates]
    uncovered = set(gold_ids)
    groups: dict[str, list[str]] = defaultdict(list)
    for candidate_id in gold_ids:
        semantic_id = str(candidates[candidate_id]["semantic_id"])
        groups[semantic_id].append(candidate_id)

    plan = []
    covered: set[str] = set()
    used_paths: set[str] = set()
    for step_index in range(max_steps):
        best_semantic_id = None
        best_ids: list[str] = []
        for semantic_id, ids in groups.items():
            if semantic_id in used_paths:
                continue
            gain_ids = [candidate_id for candidate_id in ids if candidate_id in uncovered]
            if len(gain_ids) > len(best_ids):
                best_semantic_id = semantic_id
                best_ids = gain_ids
        if not best_semantic_id or not best_ids:
            break
        candidate = candidates[best_ids[0]]
        covered_before = sorted(covered)
        covered.update(best_ids)
        uncovered.difference_update(best_ids)
        used_paths.add(best_semantic_id)
        plan.append(
            {
                "step_index": step_index,
                "semantic_id": best_semantic_id,
                "code_path": list(candidate["code_path"]),
                "role_hint": candidate.get("role_hint"),
                "target_ids": best_ids,
                "covered_before": covered_before,
                "covered_after": sorted(covered),
                "remaining_after": sorted(uncovered),
                "coverage_gain": len(best_ids),
                "normalized_coverage_gain": len(best_ids) / max(len(gold_ids), 1),
            }
        )
        if not uncovered:
            break
    return plan


def _plan_to_examples(query: Mapping[str, Any], plan: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    examples = []
    for step in plan:
        examples.append(
            {
                "query_id": query["query_id"],
                "query": query["query"],
                "split": query.get("split"),
                "source_dataset": query.get("source_dataset"),
                "step_index": step["step_index"],
                "residual_state": _residual_state(step),
                "target_ids": step["target_ids"],
                "semantic_id": step["semantic_id"],
                "code_path": step["code_path"],
                "role_hint": step.get("role_hint"),
                "coverage_gain": step["coverage_gain"],
                "normalized_coverage_gain": step["normalized_coverage_gain"],
                "covered_before": step["covered_before"],
                "remaining_after": step["remaining_after"],
            }
        )
    return examples


def _residual_state(step: Mapping[str, Any]) -> str:
    covered = " ".join(str(item) for item in step.get("covered_before") or [])
    return f"step={step['step_index']} covered={covered}"
