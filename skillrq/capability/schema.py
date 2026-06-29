"""Canonical Agent Capability Recommendation schemas."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence


def capability_object(
    *,
    capability_id: str,
    source_dataset: str,
    source_capability_id: str | None,
    capability_type: str,
    name: str,
    description: str | None = None,
    category: str | None = None,
    domain: str | None = None,
    provider: str | None = None,
    tool_name: str | None = None,
    api_name: str | None = None,
    api_schema: Mapping[str, Any] | None = None,
    parameters: Sequence[Mapping[str, Any]] | None = None,
    required_parameters: Sequence[Mapping[str, Any]] | None = None,
    optional_parameters: Sequence[Mapping[str, Any]] | None = None,
    input_schema: Mapping[str, Any] | None = None,
    output_schema: Mapping[str, Any] | None = None,
    method: str | None = None,
    endpoint: str | None = None,
    raw: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "capability_id": capability_id,
        "source_dataset": source_dataset,
        "source_capability_id": source_capability_id,
        "capability_type": capability_type,
        "name": name,
        "description": description,
        "category": category,
        "domain": domain,
        "provider": provider,
        "tool_name": tool_name,
        "api_name": api_name,
        "api_schema": _as_mapping(api_schema),
        "parameters": list(parameters or []),
        "required_parameters": list(required_parameters or []),
        "optional_parameters": list(optional_parameters or []),
        "input_schema": _as_mapping(input_schema),
        "output_schema": _as_mapping(output_schema),
        "method": method,
        "endpoint": endpoint,
        "raw": _as_mapping(raw),
    }


def capability_query(
    *,
    query_id: str,
    source_dataset: str,
    source_query_id: str,
    source_split: str,
    query: str,
    gold_capability_ids: Sequence[str],
    available_capability_ids: Sequence[str] | None = None,
    tool_call_sequence: Sequence[str] | None = None,
    tool_arguments: Sequence[Mapping[str, Any]] | None = None,
    intermediate_observations: Sequence[Any] | None = None,
    final_answer: Any = None,
    success: bool | None = None,
    raw: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    unique_gold = list(dict.fromkeys(gold_capability_ids))
    sequence = list(tool_call_sequence or [])
    return {
        "query_id": query_id,
        "source_dataset": source_dataset,
        "source_query_id": source_query_id,
        "source_split": source_split,
        "query": query,
        "gold_capability_ids": unique_gold,
        "available_capability_ids": list(dict.fromkeys(available_capability_ids or [])),
        "tool_call_sequence": sequence,
        "tool_calls_per_trajectory": len(sequence),
        "unique_tools_per_query": len(unique_gold),
        "tool_arguments": list(tool_arguments or []),
        "intermediate_observations": list(intermediate_observations or []),
        "final_answer": final_answer,
        "success": success,
        "raw": _as_mapping(raw),
    }


def capability_qrels(query: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    for capability_id in query["gold_capability_ids"]:
        yield {
            "query_id": query["query_id"],
            "capability_id": capability_id,
            "relevance": 1,
            "source_dataset": query["source_dataset"],
            "source_split": query["source_split"],
        }


def _as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    return {"value": value}


def capability_sequence_rows(query: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    arguments = list(query.get("tool_arguments") or [])
    observations = list(query.get("intermediate_observations") or [])
    for step_index, capability_id in enumerate(query.get("tool_call_sequence") or []):
        yield {
            "query_id": query["query_id"],
            "step_index": step_index,
            "capability_id": capability_id,
            "arguments": arguments[step_index] if step_index < len(arguments) else {},
            "observation": observations[step_index] if step_index < len(observations) else None,
            "source_dataset": query["source_dataset"],
            "source_split": query["source_split"],
        }
