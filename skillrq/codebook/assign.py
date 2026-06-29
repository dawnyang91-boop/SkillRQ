"""Heuristic CapabilityRQ code assignment v1."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .ids import code_id, semantic_id
from ..utils.io import read_jsonl


ROLE_START = "START"
ROLE_SUPPORT = "SUPPORT"
ROLE_CHECK = "CHECK"
ROLE_FINALIZE = "FINALIZE"
ROLE_AVOID = "AVOID"
ROLE_UNKNOWN = "UNASSIGNED"


def build_capability_role_map(sequence_path: Path) -> Mapping[str, str]:
    role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    current_query_id: str | None = None
    current_rows: list[Mapping[str, Any]] = []
    for row in read_jsonl(sequence_path):
        query_id = str(row["query_id"])
        if current_query_id is not None and query_id != current_query_id:
            _accumulate_sequence_roles(current_rows, role_counts)
            current_rows = []
        current_query_id = query_id
        current_rows.append(row)
    if current_rows:
        _accumulate_sequence_roles(current_rows, role_counts)
    return {capability_id: counts.most_common(1)[0][0] for capability_id, counts in role_counts.items()}


def assign_capability_code(row: Mapping[str, Any], role_map: Mapping[str, str]) -> Dict[str, Any]:
    capability_id = str(row["capability_id"])
    l1_label = _first_present(
        row.get("category"),
        row.get("domain"),
        row.get("provider"),
        row.get("source_dataset"),
        default="general",
    )
    l2_label = _operation_label(row.get("api_name") or row.get("tool_name") or row.get("name") or capability_id)
    role = role_map.get(capability_id) or _capability_role_from_text(row)
    l4_label = _io_constraint_label(row)
    return _assignment_row(
        object_id=capability_id,
        object_type="capability",
        source_dataset=str(row.get("source_dataset") or "unknown"),
        name=str(row.get("name") or row.get("api_name") or row.get("tool_name") or capability_id),
        l1_label=l1_label,
        l2_label=l2_label,
        l3_label=role,
        l4_label=l4_label,
        category_label=str(l1_label),
        role_label=role,
        metadata={
            "capability_type": row.get("capability_type"),
            "tool_name": row.get("tool_name"),
            "api_name": row.get("api_name"),
            "source_capability_id": row.get("source_capability_id"),
        },
    )


def assign_skillret_code(row: Mapping[str, Any]) -> Dict[str, Any]:
    skill_id = str(row["skill_id"])
    l1_label = _first_present(row.get("domain_label"), row.get("major"), row.get("sub"), default="skill_general")
    l2_label = _first_present(
        row.get("primary_action"),
        row.get("operation_label"),
        _operation_label(row.get("name") or skill_id),
        default="skill_operation",
    )
    role = _skill_role_from_text(row)
    l4_label = _skill_detail_label(row)
    return _assignment_row(
        object_id=skill_id,
        object_type="skill",
        source_dataset="skillret",
        name=str(row.get("name") or skill_id),
        l1_label=l1_label,
        l2_label=l2_label,
        l3_label=role,
        l4_label=l4_label,
        category_label=str(l1_label),
        role_label=role,
        metadata={
            "namespace": row.get("namespace"),
            "source_skill_id": row.get("source_skill_id"),
            "source_split": row.get("source_split"),
        },
    )


def assign_skillrouter_code(row: Mapping[str, Any]) -> Dict[str, Any]:
    skill_id = str(row["skill_id"])
    source = str(row.get("source") or skill_id.split("/", 1)[0] or "skillrouter")
    l1_label = _skillrouter_domain_label(row)
    l2_label = _operation_label(row.get("name") or skill_id)
    role = _skill_role_from_text(row)
    l4_label = _skill_detail_label(row)
    return _assignment_row(
        object_id=skill_id,
        object_type="skill",
        source_dataset="skillrouter",
        name=str(row.get("name") or skill_id),
        l1_label=l1_label,
        l2_label=l2_label,
        l3_label=role,
        l4_label=l4_label,
        category_label=l1_label,
        role_label=role,
        metadata={"source": source},
    )


def _accumulate_sequence_roles(rows: list[Mapping[str, Any]], role_counts: dict[str, Counter[str]]) -> None:
    rows = sorted(rows, key=lambda item: int(item.get("step_index") or 0))
    length = len(rows)
    for index, row in enumerate(rows):
        capability_id = str(row["capability_id"])
        if length == 1 or index == 0:
            role = ROLE_START
        elif index == length - 1:
            role = ROLE_FINALIZE
        else:
            role = ROLE_SUPPORT
        role_counts[capability_id][role] += 1


def _assignment_row(
    *,
    object_id: str,
    object_type: str,
    source_dataset: str,
    name: str,
    l1_label: str,
    l2_label: str,
    l3_label: str,
    l4_label: str,
    category_label: str,
    role_label: str,
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    codes = [
        code_id("L1", l1_label),
        code_id("L2", l2_label),
        code_id("L3", l3_label),
        code_id("L4", l4_label),
    ]
    return {
        "object_id": object_id,
        "object_type": object_type,
        "source_dataset": source_dataset,
        "name": name,
        "semantic_id": semantic_id(codes),
        "code_path": codes,
        "l1_code": codes[0],
        "l1_label": l1_label,
        "l2_code": codes[1],
        "l2_label": l2_label,
        "l3_code": codes[2],
        "l3_label": l3_label,
        "l4_code": codes[3],
        "l4_label": l4_label,
        "category_label": category_label,
        "role_label": role_label,
        "code_explanation": _code_explanation(l1_label, l2_label, l3_label, l4_label),
        "metadata": dict(metadata),
    }


def _code_explanation(l1_label: str, l2_label: str, l3_label: str, l4_label: str) -> str:
    return (
        f"L1 groups the domain/scenario as {l1_label}; "
        f"L2 captures the operation as {l2_label}; "
        f"L3 assigns the execution role {l3_label}; "
        f"L4 summarizes IO or constraint evidence as {l4_label}."
    )


def _capability_role_from_text(row: Mapping[str, Any]) -> str:
    text = " ".join(str(row.get(key) or "") for key in ["name", "description", "api_name", "tool_name"]).lower()
    if any(word in text for word in ["check", "verify", "validate", "test", "status"]):
        return ROLE_CHECK
    if any(word in text for word in ["delete", "remove", "cancel", "logout", "disable"]):
        return ROLE_FINALIZE
    if any(word in text for word in ["get", "search", "list", "fetch", "retrieve", "query"]):
        return ROLE_START
    if any(word in text for word in ["add", "create", "update", "send", "set", "book"]):
        return ROLE_SUPPORT
    return ROLE_UNKNOWN


def _skill_role_from_text(row: Mapping[str, Any]) -> str:
    text = " ".join(str(row.get(key) or "") for key in ["name", "description", "body", "skill_md"]).lower()
    if any(word in text for word in ["avoid", "do not", "don't", "warning", "anti-pattern", "forbidden"]):
        return ROLE_AVOID
    if any(word in text for word in ["check", "verify", "validate", "test", "debug", "audit"]):
        return ROLE_CHECK
    if any(word in text for word in ["deploy", "commit", "save", "export", "final", "report"]):
        return ROLE_FINALIZE
    if any(word in text for word in ["support", "helper", "utility", "reference"]):
        return ROLE_SUPPORT
    return ROLE_START


def _operation_label(value: object) -> str:
    text = _split_identifier(str(value or "operation"))
    tokens = [token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if token]
    if not tokens:
        return "operation"
    if tokens[0] in {"get", "add", "create", "update", "delete", "remove", "search", "list", "send", "set"}:
        return "_".join(tokens[:2])
    return "_".join(tokens[:3])


def _io_constraint_label(row: Mapping[str, Any]) -> str:
    required = row.get("required_parameters") or []
    optional = row.get("optional_parameters") or []
    output = row.get("output_schema") or {}
    method = row.get("method") or "method_unknown"
    required_count = len(required) if isinstance(required, list) else 0
    optional_count = len(optional) if isinstance(optional, list) else 0
    output_count = len(output) if isinstance(output, Mapping) else 0
    if required_count == 0 and optional_count == 0 and output_count == 0:
        return f"{method}_schema_light"
    if required_count >= 4:
        return f"{method}_multi_input_{required_count}_out_{output_count}"
    return f"{method}_input_{required_count}_optional_{optional_count}_out_{output_count}"


def _skill_detail_label(row: Mapping[str, Any]) -> str:
    text = str(row.get("body") or row.get("skill_md") or row.get("description") or "").lower()
    flags = []
    if any(word in text for word in ["constraint", "critical", "must", "warning", "do not"]):
        flags.append("constraints")
    if any(word in text for word in ["example", "usage", "workflow", "steps"]):
        flags.append("examples")
    if any(word in text for word in ["api", "schema", "parameter", "input", "output"]):
        flags.append("io")
    if any(word in text for word in ["test", "verify", "validate", "metric"]):
        flags.append("validation")
    return "_".join(flags[:3]) if flags else "description_only"


def _skillrouter_domain_label(row: Mapping[str, Any]) -> str:
    skill_id = str(row.get("skill_id") or "")
    prefix = skill_id.split("/", 1)[0] if "/" in skill_id else str(row.get("source") or "skillrouter")
    if prefix == "gt":
        return "ground_truth_skill"
    if prefix == "degraded":
        return "hard_negative_skill"
    return prefix or "skillrouter"


def _first_present(*values: object, default: str) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _split_identifier(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    value = value.replace("-", " ").replace("_", " ").replace("/", " ")
    return value
