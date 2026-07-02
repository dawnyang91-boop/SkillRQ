"""Mock tool-call simulation from code-path-guided prompt records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..utils.io import read_jsonl, write_json, write_jsonl


def simulate_mock_tool_calls(
    prompt_record_path: Path,
    output_root: Path,
    max_calls: int = 6,
    tools_per_step: int = 1,
) -> Mapping[str, Any]:
    """Generate deterministic tool-call plans from prompt records.

    The mock simulator never invents tools. It selects top prompt-grounded candidate
    tools in capability-plan order, which gives us a parser/evaluator baseline before
    adding real LLM or vLLM inference.
    """

    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in read_jsonl(prompt_record_path):
        rows.append(_simulate_record(record, max_calls=max_calls, tools_per_step=tools_per_step))
    write_jsonl(output_root / "tool_call_plans.jsonl", rows)
    summary = {
        "method": "mock_tool_call_simulation",
        "prompt_record_path": str(prompt_record_path),
        "output_root": str(output_root),
        "queries": len(rows),
        "max_calls": max_calls,
        "tools_per_step": tools_per_step,
        "avg_tool_calls": sum(len(row["tool_calls"]) for row in rows) / max(len(rows), 1),
    }
    write_json(output_root / "simulation_summary.json", summary)
    return summary


def _simulate_record(record: Mapping[str, Any], max_calls: int, tools_per_step: int) -> Mapping[str, Any]:
    calls = []
    allowed_ids = set()
    for step in record.get("capability_plan") or []:
        step_tools = [tool for tool in step.get("candidate_tools") or [] if tool.get("candidate_id")]
        for tool in step_tools:
            allowed_ids.add(str(tool["candidate_id"]))
        for tool in step_tools[:tools_per_step]:
            if len(calls) >= max_calls:
                break
            calls.append(
                {
                    "step": step.get("step"),
                    "role": step.get("role"),
                    "operation": step.get("operation"),
                    "tool_name": tool.get("name") or tool.get("candidate_id"),
                    "candidate_id": str(tool.get("candidate_id")),
                    "arguments": {},
                    "grounding": "prompt_candidate",
                    "rationale": step.get("why_needed"),
                }
            )
        if len(calls) >= max_calls:
            break
    return {
        "query_id": record.get("query_id"),
        "query": record.get("query"),
        "gold_ids": record.get("gold_ids") or [],
        "sequence_ids": record.get("sequence_ids") or [],
        "allowed_candidate_ids": sorted(allowed_ids),
        "tool_calls": calls,
        "final_answer_plan": "Mock plan: execute selected prompt-grounded tools in capability-plan order.",
    }
