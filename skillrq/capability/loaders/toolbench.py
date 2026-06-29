"""ToolBench to canonical capability recommendation conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple

from ..ids import stable_id
from ..schema import capability_object, capability_query


def convert_toolbench(
    raw_root: Path,
    include_answer_trees: bool = True,
    limit_tools: int | None = None,
    limit_queries: int | None = None,
) -> Mapping[str, List[Mapping[str, Any]]]:
    dataset_root = raw_root / "ToolBench" / "data"
    if not dataset_root.exists():
        raise FileNotFoundError(f"ToolBench raw directory not found: {dataset_root}")

    capabilities_by_id: MutableMapping[str, Mapping[str, Any]] = {}
    key_to_id: MutableMapping[Tuple[str | None, str, str], str] = {}
    tool_api_to_id: MutableMapping[Tuple[str, str], str] = {}

    for index, capability in enumerate(_load_toolenv_capabilities(dataset_root)):
        if limit_tools is not None and index >= limit_tools:
            break
        capabilities_by_id[capability["capability_id"]] = capability
        key = (capability.get("category"), capability.get("tool_name"), capability.get("api_name"))
        key_to_id[key] = capability["capability_id"]
        tool_api_to_id[(str(capability.get("tool_name")), str(capability.get("api_name")))] = capability["capability_id"]

    instruction_queries = list(
        _load_instruction_queries(dataset_root, capabilities_by_id, key_to_id, tool_api_to_id, limit_queries=limit_queries)
    )
    answer_queries = list(_load_answer_tree_queries(dataset_root, limit_queries=limit_queries)) if include_answer_trees else []
    _ensure_runtime_capabilities(capabilities_by_id, answer_queries)
    all_queries = _merge_answer_sequences(instruction_queries, answer_queries)

    return {
        "capabilities": sorted(capabilities_by_id.values(), key=lambda row: row["capability_id"]),
        "queries": all_queries,
    }


def iter_toolbench_answer_tree_queries(raw_root: Path) -> Iterable[Mapping[str, Any]]:
    dataset_root = raw_root / "ToolBench" / "data"
    yield from _load_answer_tree_queries(dataset_root, limit_queries=None)


def runtime_capability_from_id(capability_id: str) -> Mapping[str, Any]:
    parts = str(capability_id).split("::")
    action_name = parts[-2] if len(parts) >= 3 else parts[-1]
    return capability_object(
        capability_id=capability_id,
        source_dataset="toolbench",
        source_capability_id=None,
        capability_type="runtime_tool_action",
        name=action_name,
        description="Runtime action extracted from a ToolBench answer tree.",
        provider="ToolBench answer tree",
        tool_name=action_name,
        api_name=action_name,
        raw={"source": "answer_tree"},
    )


def _load_toolenv_capabilities(dataset_root: Path) -> Iterable[Mapping[str, Any]]:
    for path in sorted((dataset_root / "toolenv" / "tools").glob("*/*.json")):
        category = path.parent.name
        raw_tool = json.loads(path.read_text(encoding="utf-8"))
        tool_name = str(raw_tool.get("tool_name") or raw_tool.get("title") or path.stem)
        for api in raw_tool.get("api_list") or []:
            api_name = str(api.get("name") or "unknown_api")
            capability_id = _toolbench_capability_id(category, tool_name, api_name)
            yield capability_object(
                capability_id=capability_id,
                source_dataset="toolbench",
                source_capability_id=str(path.relative_to(dataset_root)),
                capability_type="tool_api",
                name=f"{tool_name}::{api_name}",
                description=api.get("description") or raw_tool.get("tool_description"),
                category=category,
                domain=category,
                provider=raw_tool.get("host"),
                tool_name=tool_name,
                api_name=api_name,
                api_schema=api,
                parameters=list(api.get("required_parameters") or []) + list(api.get("optional_parameters") or []),
                required_parameters=api.get("required_parameters") or [],
                optional_parameters=api.get("optional_parameters") or [],
                input_schema={
                    "required_parameters": api.get("required_parameters") or [],
                    "optional_parameters": api.get("optional_parameters") or [],
                },
                output_schema=api.get("template_response") or {},
                method=api.get("method"),
                endpoint=api.get("url"),
                raw={
                    "path": str(path.relative_to(dataset_root)),
                    "category": category,
                    "tool_name": tool_name,
                    "api_name": api_name,
                },
            )


def _load_instruction_queries(
    dataset_root: Path,
    capabilities_by_id: MutableMapping[str, Mapping[str, Any]],
    key_to_id: MutableMapping[Tuple[str | None, str, str], str],
    tool_api_to_id: MutableMapping[Tuple[str, str], str],
    limit_queries: int | None = None,
) -> Iterable[Mapping[str, Any]]:
    emitted = 0
    roots = [
        (dataset_root / "instruction", "train"),
        (dataset_root / "test_instruction", "test"),
    ]
    for root, split in roots:
        for path in sorted(root.glob("*.json")):
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                continue
            for record in records:
                source_query_id = str(record.get("query_id"))
                group = path.stem
                query_id = f"toolbench::{group}::{source_query_id}"
                available_ids = [
                    _capability_id_from_api_entry(api, capabilities_by_id, key_to_id, tool_api_to_id)
                    for api in record.get("api_list") or []
                ]
                gold_ids = [
                    _capability_id_from_relevant_api(pair, capabilities_by_id, tool_api_to_id)
                    for pair in record.get("relevant APIs") or []
                ]
                if not gold_ids:
                    gold_ids = available_ids
                yield capability_query(
                    query_id=query_id,
                    source_dataset="toolbench",
                    source_query_id=source_query_id,
                    source_split=split,
                    query=str(record.get("query") or ""),
                    gold_capability_ids=[capability_id for capability_id in gold_ids if capability_id],
                    available_capability_ids=[capability_id for capability_id in available_ids if capability_id],
                    tool_call_sequence=[],
                    raw={
                        "path": str(path.relative_to(dataset_root)),
                        "relevant APIs": record.get("relevant APIs"),
                        "query_id": record.get("query_id"),
                    },
                )
                emitted += 1
                if limit_queries is not None and emitted >= limit_queries:
                    return


def _load_answer_tree_queries(dataset_root: Path, limit_queries: int | None = None) -> Iterable[Mapping[str, Any]]:
    answer_root = dataset_root / "answer"
    for index, path in enumerate(sorted(answer_root.glob("*_answer/*.json"))):
        if limit_queries is not None and index >= limit_queries:
            break
        raw = json.loads(path.read_text(encoding="utf-8"))
        query_id_number = path.name.split("_", 1)[0]
        group = path.parent.name.replace("_answer", "")
        query_id = f"toolbench::{group}_query::{query_id_number}"
        sequence, arguments, observations = _extract_leftmost_action_path(raw.get("tree", {}).get("tree") or {})
        answer_generation = raw.get("answer_generation") or {}
        yield capability_query(
            query_id=query_id,
            source_dataset="toolbench",
            source_query_id=query_id_number,
            source_split="answer_tree",
            query=str(answer_generation.get("query") or ""),
            gold_capability_ids=[stable_id("toolbench_runtime", [name]) for name in sequence if name != "Finish"],
            tool_call_sequence=[stable_id("toolbench_runtime", [name]) for name in sequence if name != "Finish"],
            tool_arguments=arguments,
            intermediate_observations=observations,
            final_answer=answer_generation.get("final_answer"),
            success=bool(raw.get("win")),
            raw={"path": str(path.relative_to(dataset_root)), "forward_args": raw.get("forward_args")},
        )


def _ensure_runtime_capabilities(
    capabilities_by_id: MutableMapping[str, Mapping[str, Any]],
    answer_queries: List[Mapping[str, Any]],
) -> None:
    for query in answer_queries:
        for capability_id in query.get("tool_call_sequence") or []:
            if capability_id in capabilities_by_id:
                continue
            capabilities_by_id[capability_id] = runtime_capability_from_id(capability_id)


def _merge_answer_sequences(
    instruction_queries: List[Mapping[str, Any]],
    answer_queries: List[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    # Answer tree IDs do not always align one-to-one with instruction group IDs.
    # Keep instruction records as primary recommendation data and append answer
    # tree records as trajectory-only examples.
    return sorted([*instruction_queries, *answer_queries], key=lambda row: row["query_id"])


def _capability_id_from_api_entry(
    api: Mapping[str, Any],
    capabilities_by_id: MutableMapping[str, Mapping[str, Any]],
    key_to_id: MutableMapping[Tuple[str | None, str, str], str],
    tool_api_to_id: MutableMapping[Tuple[str, str], str],
) -> str:
    category = api.get("category_name")
    tool_name = str(api.get("tool_name") or "")
    api_name = str(api.get("api_name") or "")
    capability_id = key_to_id.get((category, tool_name, api_name)) or tool_api_to_id.get((tool_name, api_name))
    if capability_id:
        return capability_id

    capability_id = _toolbench_capability_id(category or "unknown", tool_name, api_name)
    capabilities_by_id[capability_id] = capability_object(
        capability_id=capability_id,
        source_dataset="toolbench",
        source_capability_id=None,
        capability_type="tool_api",
        name=f"{tool_name}::{api_name}",
        description=api.get("api_description"),
        category=category,
        domain=category,
        provider=None,
        tool_name=tool_name,
        api_name=api_name,
        api_schema=api,
        parameters=list(api.get("required_parameters") or []) + list(api.get("optional_parameters") or []),
        required_parameters=api.get("required_parameters") or [],
        optional_parameters=api.get("optional_parameters") or [],
        input_schema={
            "required_parameters": api.get("required_parameters") or [],
            "optional_parameters": api.get("optional_parameters") or [],
        },
        output_schema=api.get("template_response") or {},
        method=api.get("method"),
        endpoint=None,
        raw={"source": "instruction_api_list", "tool_name": tool_name, "api_name": api_name},
    )
    key_to_id[(category, tool_name, api_name)] = capability_id
    tool_api_to_id[(tool_name, api_name)] = capability_id
    return capability_id


def _capability_id_from_relevant_api(
    pair: Any,
    capabilities_by_id: MutableMapping[str, Mapping[str, Any]],
    tool_api_to_id: MutableMapping[Tuple[str, str], str],
) -> str | None:
    if not isinstance(pair, list) or len(pair) < 2:
        return None
    tool_name = str(pair[0])
    api_name = str(pair[1])
    capability_id = tool_api_to_id.get((tool_name, api_name))
    if capability_id:
        return capability_id
    capability_id = _toolbench_capability_id("unknown", tool_name, api_name)
    capabilities_by_id.setdefault(
        capability_id,
        capability_object(
            capability_id=capability_id,
            source_dataset="toolbench",
            source_capability_id=None,
            capability_type="tool_api",
            name=f"{tool_name}::{api_name}",
            tool_name=tool_name,
            api_name=api_name,
            raw={"relevant_api_pair": pair},
        ),
    )
    tool_api_to_id[(tool_name, api_name)] = capability_id
    return capability_id


def _toolbench_capability_id(category: str | None, tool_name: str, api_name: str) -> str:
    return stable_id("toolbench", [category or "unknown", tool_name, api_name])


def _extract_leftmost_action_path(node: Mapping[str, Any]) -> tuple[List[str], List[Mapping[str, Any]], List[Any]]:
    actions: List[str] = []
    arguments: List[Mapping[str, Any]] = []
    observations: List[Any] = []
    current = node
    pending_action: str | None = None
    while current:
        node_type = current.get("node_type")
        if node_type == "Action":
            pending_action = str(current.get("description") or "")
        elif node_type == "Action Input" and pending_action:
            actions.append(pending_action)
            arguments.append(_parse_json_object(current.get("description")))
            observations.append(_parse_json_object(current.get("observation")))
            pending_action = None
        children = [child for child in current.get("children") or [] if not child.get("pruned")]
        current = children[0] if children else {}
    return actions, arguments, observations


def _parse_json_object(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value}
