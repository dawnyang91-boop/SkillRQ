"""API-Bank to canonical capability recommendation conversion."""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from ..ids import stable_id
from ..schema import capability_object, capability_query


def convert_api_bank(raw_root: Path) -> Mapping[str, List[Mapping[str, Any]]]:
    dataset_root = raw_root / "DAMO-ConvAI" / "api-bank"
    if not dataset_root.exists():
        raise FileNotFoundError(f"API-Bank raw directory not found: {dataset_root}")

    capabilities = _load_capabilities(dataset_root)
    api_name_to_id = {str(row["api_name"]): str(row["capability_id"]) for row in capabilities}
    queries = list(_load_dialogue_queries(dataset_root, api_name_to_id))
    return {
        "capabilities": capabilities,
        "queries": queries,
    }


def _load_capabilities(dataset_root: Path) -> List[Mapping[str, Any]]:
    csv_path = dataset_root / "data" / "all_apis.csv"
    rows: List[Mapping[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            api_name = raw.get("类名") or raw.get("API名称") or raw.get("id")
            parsed_info = _parse_api_info(raw.get("api_info") or "")
            capability_id = stable_id("api_bank", [api_name])
            rows.append(
                capability_object(
                    capability_id=capability_id,
                    source_dataset="api_bank",
                    source_capability_id=raw.get("id"),
                    capability_type="api",
                    name=str(api_name),
                    description=parsed_info.get("description"),
                    category=raw.get("类型"),
                    domain=raw.get("应用场景"),
                    provider="API-Bank",
                    tool_name=str(api_name),
                    api_name=str(api_name),
                    api_schema={"expression": raw.get("expressions"), "api_info": raw.get("api_info")},
                    parameters=_schema_to_parameters(parsed_info.get("input_parameters") or {}),
                    required_parameters=_schema_to_parameters(parsed_info.get("input_parameters") or {}),
                    optional_parameters=[],
                    input_schema=parsed_info.get("input_parameters") or {},
                    output_schema=parsed_info.get("output_parameters") or {},
                    method=None,
                    endpoint=raw.get("路径"),
                    raw=raw,
                )
            )
    return rows


def _load_dialogue_queries(dataset_root: Path, api_name_to_id: Mapping[str, str]) -> Iterable[Mapping[str, Any]]:
    samples_root = dataset_root / "lv1-lv2-samples"
    for path in sorted(samples_root.glob("**/*.jsonl")):
        messages = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not messages:
            continue

        user_messages = [msg.get("text", "") for msg in messages if msg.get("role") == "User"]
        api_messages = [msg for msg in messages if msg.get("role") == "API"]
        ai_messages = [msg.get("text") for msg in messages if msg.get("role") == "AI"]

        call_ids: List[str] = []
        arguments: List[Mapping[str, Any]] = []
        observations: List[Any] = []
        for msg in api_messages:
            api_name = str(msg.get("api_name") or "")
            capability_id = api_name_to_id.get(api_name) or stable_id("api_bank", [api_name])
            call_ids.append(capability_id)
            arguments.append(dict(msg.get("param_dict") or {}))
            observations.append(msg.get("result"))

        rel_path = path.relative_to(samples_root)
        query_id = stable_id("api_bank_query", [rel_path])
        yield capability_query(
            query_id=query_id,
            source_dataset="api_bank",
            source_query_id=str(rel_path),
            source_split="samples",
            query=user_messages[0] if user_messages else "",
            gold_capability_ids=call_ids,
            available_capability_ids=list(api_name_to_id.values()),
            tool_call_sequence=call_ids,
            tool_arguments=arguments,
            intermediate_observations=observations,
            final_answer=ai_messages[-1] if ai_messages else None,
            success=all((obs or {}).get("exception") is None for obs in observations if isinstance(obs, dict)),
            raw={"path": str(rel_path), "messages": messages},
        )


def _parse_api_info(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    desc_match = re.search(r"description\s*=\s*(['\"])(.*?)\1", text, flags=re.DOTALL)
    if desc_match:
        result["description"] = desc_match.group(2)
    for key in ("input_parameters", "output_parameters"):
        value = _extract_python_assignment(text, key)
        if value is not None:
            result[key] = value
    return result


def _extract_python_assignment(text: str, key: str) -> Any:
    marker = f"{key} ="
    start = text.find(marker)
    if start < 0:
        return None
    start = text.find("{", start)
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : index + 1]
                try:
                    return ast.literal_eval(snippet)
                except (SyntaxError, ValueError):
                    return None
    return None


def _schema_to_parameters(schema: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    if not isinstance(schema, dict):
        return []
    return [
        {
            "name": name,
            "type": value.get("type") if isinstance(value, dict) else None,
            "description": value.get("description") if isinstance(value, dict) else None,
        }
        for name, value in schema.items()
    ]
