"""vLLM-backed tool-call plan generation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..utils.io import read_jsonl, write_json, write_jsonl


def run_vllm_tool_call_planning(
    prompt_record_path: Path,
    output_root: Path,
    model: str,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.90,
    max_model_len: int | None = None,
    trust_remote_code: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    batch_size: int = 32,
    limit: int | None = None,
    seed: int | None = 0,
) -> Mapping[str, Any]:
    """Run vLLM offline inference and write standard tool-call plans."""

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is required for `agent-sim vllm`. Install vllm in the server environment first.") from exc

    output_root.mkdir(parents=True, exist_ok=True)
    records = list(read_jsonl(prompt_record_path))
    if limit is not None:
        records = records[:limit]
    prompts = [_llm_prompt(record) for record in records]
    llm_kwargs = {
        "model": model,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    if seed is not None:
        llm_kwargs["seed"] = seed
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    output_rows = []
    raw_rows = []
    for start in range(0, len(prompts), batch_size):
        batch_records = records[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        generations = llm.generate(batch_prompts, sampling_params)
        for record, prompt, generation in zip(batch_records, batch_prompts, generations):
            text = generation.outputs[0].text if generation.outputs else ""
            parsed = parse_tool_call_plan(text)
            output_rows.append(_normalize_llm_plan(record, parsed, raw_text=text))
            raw_rows.append({"query_id": record.get("query_id"), "prompt": prompt, "raw_output": text, "parsed": parsed})

    write_jsonl(output_root / "tool_call_plans.jsonl", output_rows)
    write_jsonl(output_root / "raw_generations.jsonl", raw_rows)
    summary = {
        "method": "vllm_tool_call_plan_generation",
        "prompt_record_path": str(prompt_record_path),
        "output_root": str(output_root),
        "model": model,
        "queries": len(output_rows),
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
        "limit": limit,
        "avg_tool_calls": sum(len(row["tool_calls"]) for row in output_rows) / max(len(output_rows), 1),
        "parse_success_rate": sum(1 for row in output_rows if row.get("parse_success")) / max(len(output_rows), 1),
    }
    write_json(output_root / "vllm_summary.json", summary)
    return summary


def parse_tool_call_plan(text: str) -> Mapping[str, Any]:
    """Parse model output into the expected tool-call plan object."""

    try:
        value = json.loads(text)
        if isinstance(value, Mapping):
            return {"parse_success": True, **dict(value)}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, Mapping):
                return {"parse_success": True, **dict(value)}
        except json.JSONDecodeError:
            pass
    return {"parse_success": False, "tool_calls": [], "final_answer_plan": "", "parse_error": "Could not parse JSON object."}


def _normalize_llm_plan(record: Mapping[str, Any], parsed: Mapping[str, Any], raw_text: str) -> Mapping[str, Any]:
    allowed_by_id = _allowed_tools(record)
    calls = []
    for call in parsed.get("tool_calls") or []:
        if not isinstance(call, Mapping):
            continue
        candidate_id = str(call.get("candidate_id") or "")
        tool = allowed_by_id.get(candidate_id)
        calls.append(
            {
                "step": call.get("step"),
                "role": call.get("role") or (tool or {}).get("suggested_role"),
                "operation": call.get("operation"),
                "tool_name": call.get("tool_name") or (tool or {}).get("name") or candidate_id,
                "candidate_id": candidate_id,
                "arguments": call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {},
                "grounding": "prompt_candidate" if candidate_id in allowed_by_id else "invalid_or_hallucinated",
                "rationale": call.get("rationale") or call.get("reason"),
            }
        )
    return {
        "query_id": record.get("query_id"),
        "query": record.get("query"),
        "gold_ids": record.get("gold_ids") or [],
        "sequence_ids": record.get("sequence_ids") or [],
        "allowed_candidate_ids": sorted(allowed_by_id),
        "tool_calls": calls,
        "final_answer_plan": parsed.get("final_answer_plan") or parsed.get("answer_plan") or "",
        "parse_success": bool(parsed.get("parse_success")),
        "raw_output": raw_text,
    }


def _llm_prompt(record: Mapping[str, Any]) -> str:
    allowed_tools = _allowed_tools(record)
    tool_lines = []
    for candidate_id, tool in allowed_tools.items():
        tool_lines.append(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "tool_name": tool.get("name") or candidate_id,
                    "suggested_role": tool.get("suggested_role"),
                    "schema_or_evidence": str(tool.get("schema_or_evidence") or "")[:500],
                },
                ensure_ascii=False,
            )
        )
    schema = {
        "tool_calls": [
            {
                "step": 1,
                "tool_name": "string, must be one of the candidate tools",
                "candidate_id": "string, must be one of the allowed candidate_id values",
                "arguments": {},
                "rationale": "short reason grounded in the capability plan",
            }
        ],
        "final_answer_plan": "short natural-language execution plan",
    }
    return "\n".join(
        [
            "You are an LLM agent planner. Select tools only from the allowed candidate_id list.",
            "Return valid JSON only. Do not include markdown fences or commentary.",
            "",
            "Required JSON schema:",
            json.dumps(schema, ensure_ascii=False),
            "",
            "Allowed candidate tools:",
            "\n".join(tool_lines),
            "",
            "Code-structured planning prompt:",
            str(record.get("prompt") or ""),
        ]
    )


def _allowed_tools(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    tools: dict[str, Mapping[str, Any]] = {}
    for step in record.get("capability_plan") or []:
        for tool in step.get("candidate_tools") or []:
            candidate_id = tool.get("candidate_id")
            if candidate_id is None:
                continue
            tools[str(candidate_id)] = tool
    return tools
