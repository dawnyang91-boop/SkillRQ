"""Canonical schema builders for processed datasets."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence


def canonical_skill(raw: Mapping[str, Any], split: str) -> Dict[str, Any]:
    skill_id = _required_str(raw, "id")
    return {
        "skill_id": skill_id,
        "source_skill_id": skill_id,
        "source_dataset": "skillret",
        "source_split": split,
        "name": _optional_str(raw, "name"),
        "namespace": _optional_str(raw, "namespace"),
        "description": _optional_str(raw, "description"),
        "body": _optional_str(raw, "body"),
        "skill_md": _optional_str(raw, "skill_md"),
        "domain_label": _optional_str(raw, "domain"),
        "operation_label": _optional_str(raw, "primary_action"),
        "major": _optional_str(raw, "major"),
        "sub": _optional_str(raw, "sub"),
        "primary_action": _optional_str(raw, "primary_action"),
        "primary_object": _optional_str(raw, "primary_object"),
        "author": _optional_str(raw, "author"),
        "license": _optional_str(raw, "license"),
        "repo": _optional_str(raw, "repo"),
        "source_url": _optional_str(raw, "source_url"),
        "raw_url": _optional_str(raw, "raw_url"),
        "stars": _optional_int(raw, "stars"),
        "installs": _optional_int(raw, "installs"),
    }


def canonical_query(raw: Mapping[str, Any], split: str, gold_skill_ids: Sequence[str]) -> Dict[str, Any]:
    query_id = _required_str(raw, "id")
    return {
        "query_id": query_id,
        "source_query_id": query_id,
        "source_dataset": "skillret",
        "source_split": split,
        "query": _required_str(raw, "query"),
        "original_query": _optional_str(raw, "original_query"),
        "gold_skill_ids": sorted(set(gold_skill_ids)),
        "gold_skill_names": list(raw.get("skill_names") or []),
        "k": _optional_int(raw, "k"),
        "generator_model": _optional_str(raw, "generator_model"),
        "difficulty": None,
        "domain": None,
    }


def canonical_qrel(raw: Mapping[str, Any], split: str) -> Dict[str, Any]:
    return {
        "query_id": _required_str(raw, "query_id"),
        "skill_id": _required_str(raw, "skill_id"),
        "relevance": _optional_int(raw, "relevance", default=1),
        "source_dataset": "skillret",
        "source_split": split,
    }


def canonical_task_skill_set(query: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": query["query_id"],
        "query_id": query["query_id"],
        "source_dataset": query["source_dataset"],
        "source_split": query["source_split"],
        "gold_skill_ids": query["gold_skill_ids"],
        "num_gold_skills": len(query["gold_skill_ids"]),
    }


def canonical_role_rows(query: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    for skill_id in query["gold_skill_ids"]:
        yield {
            "query_id": query["query_id"],
            "skill_id": skill_id,
            "role": None,
            "role_source": "unassigned",
            "source_dataset": query["source_dataset"],
            "source_split": query["source_split"],
        }


def split_row(query_id: str, split: str, source_split: str) -> Dict[str, Any]:
    return {
        "query_id": query_id,
        "split": split,
        "source_dataset": "skillret",
        "source_split": source_split,
    }


def _required_str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"Missing required field: {key}")
    return str(value)


def _optional_str(raw: Mapping[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    return str(value)


def _optional_int(raw: Mapping[str, Any], key: str, default: int | None = None) -> int | None:
    value = raw.get(key, default)
    if value is None or value == "":
        return default
    return int(value)

