"""Build code-path-guided LLM planning prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..utils.io import read_jsonl, write_json, write_jsonl


def build_code_guided_prompts(
    prediction_path: Path,
    output_root: Path,
    m5_prediction_path: Path | None = None,
    top_tools_per_step: int = 3,
    max_steps: int = 6,
    include_scores: bool = True,
) -> Mapping[str, Any]:
    """Create structured prompt records from M5/M7 prediction outputs."""

    output_root.mkdir(parents=True, exist_ok=True)
    prediction_rows = list(read_jsonl(prediction_path))
    m5_by_query_id = _load_optional_rows(m5_prediction_path)
    records = []
    for row in prediction_rows:
        query_id = str(row.get("query_id"))
        m5_row = m5_by_query_id.get(query_id)
        prompt_record = _build_prompt_record(
            row=row,
            m5_row=m5_row,
            top_tools_per_step=top_tools_per_step,
            max_steps=max_steps,
            include_scores=include_scores,
        )
        records.append(prompt_record)

    write_jsonl(output_root / "prompt_records.jsonl", records)
    (output_root / "prompts.md").write_text(_records_to_markdown(records), encoding="utf-8")
    summary = {
        "method": "code_path_guided_prompt_construction",
        "prediction_path": str(prediction_path),
        "m5_prediction_path": str(m5_prediction_path) if m5_prediction_path else None,
        "output_root": str(output_root),
        "queries": len(records),
        "top_tools_per_step": top_tools_per_step,
        "max_steps": max_steps,
        "avg_steps": sum(len(row["capability_plan"]) for row in records) / max(len(records), 1),
        "avg_tools_per_prompt": sum(_tool_count(row) for row in records) / max(len(records), 1),
    }
    write_json(output_root / "prompt_summary.json", summary)
    return summary


def _build_prompt_record(
    row: Mapping[str, Any],
    m5_row: Mapping[str, Any] | None,
    top_tools_per_step: int,
    max_steps: int,
    include_scores: bool,
) -> Mapping[str, Any]:
    query_id = str(row.get("query_id"))
    query = str(row.get("query") or (m5_row or {}).get("query") or "")
    reranked = list(row.get("reranked_capabilities") or [])
    steps = _extract_steps(m5_row or row, max_steps=max_steps)
    if not steps:
        steps = _steps_from_reranked(row, max_steps=max_steps)

    plan = []
    for index, step in enumerate(steps[:max_steps], start=1):
        tools = _select_step_tools(step, reranked, top_tools_per_step, include_scores)
        plan.append(
            {
                "step": index,
                "role": step.get("role") or step.get("role_hint") or _role_from_codes(step.get("codes") or step.get("code_path") or []),
                "operation": _operation_from_codes(step.get("codes") or step.get("code_path") or []),
                "required_code_path": list(step.get("codes") or step.get("code_path") or []),
                "why_needed": _why_needed(step),
                "candidate_tools": tools,
                "coverage_state": {
                    "expected_coverage_gain": step.get("expected_coverage_gain") or step.get("predicted_coverage_gain"),
                    "step_index": step.get("step_index"),
                },
            }
        )

    prompt_text = _render_prompt(query=query, capability_plan=plan)
    return {
        "query_id": query_id,
        "query": query,
        "source_dataset": row.get("source_dataset") or (m5_row or {}).get("source_dataset"),
        "split": row.get("split") or (m5_row or {}).get("split"),
        "gold_ids": row.get("gold_ids") or (m5_row or {}).get("gold_ids"),
        "sequence_ids": row.get("sequence_ids") or (m5_row or {}).get("sequence_ids") or [],
        "capability_plan": plan,
        "prompt": prompt_text,
    }


def _extract_steps(row: Mapping[str, Any], max_steps: int) -> list[Mapping[str, Any]]:
    raw_steps = list(row.get("code_plan") or row.get("residual_code_paths") or [])
    steps = []
    for step in raw_steps[:max_steps]:
        if not isinstance(step, Mapping):
            continue
        steps.append(step)
    return steps


def _steps_from_reranked(row: Mapping[str, Any], max_steps: int) -> list[Mapping[str, Any]]:
    steps = []
    seen_paths = set()
    for item in row.get("reranked_capabilities") or []:
        codes = list(item.get("matched_code_path") or [])
        if not codes:
            continue
        key = "/".join(str(value) for value in codes)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        steps.append(
            {
                "step_index": len(steps),
                "semantic_id": key,
                "codes": codes,
                "role_hint": item.get("suggested_role"),
                "purpose": item.get("compact_support_evidence") or item.get("code_explanation"),
                "retrieved_capabilities": [item],
            }
        )
        if len(steps) >= max_steps:
            break
    return steps


def _select_step_tools(
    step: Mapping[str, Any],
    reranked: Sequence[Mapping[str, Any]],
    top_tools_per_step: int,
    include_scores: bool,
) -> list[Mapping[str, Any]]:
    step_candidate_ids = {
        str(item.get("candidate_id"))
        for item in step.get("retrieved_capabilities") or []
        if isinstance(item, Mapping) and item.get("candidate_id") is not None
    }
    reranked_by_id = {str(item.get("candidate_id")): item for item in reranked if item.get("candidate_id") is not None}
    if step_candidate_ids:
        candidates = [reranked_by_id[candidate_id] for candidate_id in step_candidate_ids if candidate_id in reranked_by_id]
        if not candidates:
            candidates = list(step.get("retrieved_capabilities") or [])
    else:
        step_codes = list(step.get("codes") or step.get("code_path") or [])
        candidates = [item for item in reranked if _path_overlap(step_codes, list(item.get("matched_code_path") or [])) >= 2]
    candidates = sorted(candidates, key=lambda item: float(item.get("final_score") or item.get("code_match_score") or 0.0), reverse=True)
    return [_tool_record(item, include_scores) for item in candidates[:top_tools_per_step]]


def _tool_record(item: Mapping[str, Any], include_scores: bool) -> Mapping[str, Any]:
    record: dict[str, Any] = {
        "candidate_id": item.get("candidate_id"),
        "name": item.get("name"),
        "source_dataset": item.get("source_dataset"),
        "suggested_role": item.get("suggested_role") or item.get("role_hint"),
        "schema_or_evidence": item.get("code_explanation") or item.get("compact_support_evidence") or item.get("capability_text_evidence"),
    }
    if include_scores:
        record["scores"] = {
            "final": item.get("final_score"),
            "relevance": item.get("relevance_score"),
            "prompt_usefulness": item.get("prompt_usefulness_score"),
            "code_path_consistency": item.get("code_path_consistency_score"),
            "schema_compatibility": item.get("schema_compatibility_score"),
            "coverage_gain": item.get("coverage_gain_score"),
        }
    return record


def _render_prompt(query: str, capability_plan: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "User Query:",
        query,
        "",
        "Capability Plan:",
    ]
    for step in capability_plan:
        role = step.get("role") or "UNKNOWN"
        operation = step.get("operation") or "UNKNOWN"
        lines.extend(
            [
                f"Step {step['step']}: {role} / {operation}",
                "- Required capability code path:",
                f"  {step.get('required_code_path') or []}",
                "- Candidate tools:",
            ]
        )
        tools = list(step.get("candidate_tools") or [])
        if not tools:
            lines.append("  1. No candidate tool selected.")
        for index, tool in enumerate(tools, start=1):
            name = tool.get("name") or tool.get("candidate_id")
            lines.append(f"  {index}. {name}")
            evidence = tool.get("schema_or_evidence")
            if evidence:
                lines.append(f"     Evidence: {str(evidence)[:280]}")
        lines.extend(["- Why needed:", f"  {step.get('why_needed') or 'This capability supports the unresolved user request.'}", ""])
    lines.extend(
        [
            "Planner Instruction:",
            "Use the capability plan as grounded planning support. Prefer tools within the same step before moving to later steps, respect START/CHECK/FINALIZE roles, and call only tools whose schema matches the user request.",
        ]
    )
    return "\n".join(lines)


def _why_needed(step: Mapping[str, Any]) -> str:
    return str(step.get("purpose") or step.get("code_explanation") or "This code path covers one required capability in the agent plan.")


def _operation_from_codes(codes: Sequence[Any]) -> str:
    if len(codes) > 1:
        return _clean_code(codes[1])
    return "UNKNOWN"


def _role_from_codes(codes: Sequence[Any]) -> str:
    if len(codes) > 2:
        return _clean_code(codes[2]).upper()
    return "UNKNOWN"


def _clean_code(value: Any) -> str:
    text = str(value or "UNKNOWN")
    if "-" in text:
        text = text.rsplit("-", 1)[0]
    return text.replace("_", " ")


def _path_overlap(left: Sequence[Any], right: Sequence[Any]) -> int:
    return sum(1 for first, second in zip(left, right) if str(first) == str(second))


def _load_optional_rows(path: Path | None) -> dict[str, Mapping[str, Any]]:
    if not path:
        return {}
    return {str(row["query_id"]): row for row in read_jsonl(path)}


def _records_to_markdown(records: Sequence[Mapping[str, Any]]) -> str:
    blocks = []
    for record in records:
        blocks.append(f"## {record['query_id']}\n\n```text\n{record['prompt']}\n```")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _tool_count(record: Mapping[str, Any]) -> int:
    return sum(len(step.get("candidate_tools") or []) for step in record.get("capability_plan") or [])
