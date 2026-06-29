"""Build canonical processed data from raw datasets."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping

from .loaders.skillret import SkillRetRawData, load_skillret
from .schemas import (
    canonical_qrel,
    canonical_query,
    canonical_role_rows,
    canonical_skill,
    canonical_task_skill_set,
    split_row,
)
from .stats import build_stats
from ..config.schema import PathsConfig
from ..utils.io import write_json, write_jsonl


DEV_EVERY_NTH_TRAIN_QUERY = 10


def build_skillret_processed_data(paths: PathsConfig) -> Mapping[str, Any]:
    raw = load_skillret(paths.raw_root)
    processed = canonicalize_skillret(raw)
    write_processed_files(paths.processed_root, processed)
    return processed["stats"]


def canonicalize_skillret(raw: SkillRetRawData) -> Mapping[str, Any]:
    skills_by_id: MutableMapping[str, Dict[str, Any]] = {}
    qrels_by_query: MutableMapping[str, List[str]] = defaultdict(list)
    qrels: List[Dict[str, Any]] = []

    for source_split, split_data in raw.by_split().items():
        for raw_skill in split_data.skills:
            skill = canonical_skill(raw_skill, split=source_split)
            existing = skills_by_id.get(skill["skill_id"])
            if existing is None or existing.get("source_split") != "train":
                skills_by_id[skill["skill_id"]] = skill

        for raw_qrel in split_data.qrels:
            qrel = canonical_qrel(raw_qrel, split=source_split)
            if qrel["relevance"] and qrel["relevance"] > 0:
                qrels_by_query[qrel["query_id"]].append(qrel["skill_id"])
            qrels.append(qrel)

    queries: List[Dict[str, Any]] = []
    split_rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for source_split, split_data in raw.by_split().items():
        for index, raw_query in enumerate(split_data.queries):
            query_id = str(raw_query["id"])
            gold_skill_ids = qrels_by_query.get(query_id) or list(raw_query.get("skill_ids") or [])
            query = canonical_query(raw_query, split=source_split, gold_skill_ids=gold_skill_ids)
            queries.append(query)

            canonical_split = _canonical_split(source_split, index)
            split_rows[canonical_split].append(split_row(query["query_id"], canonical_split, source_split))

    task_skill_sets = [canonical_task_skill_set(query) for query in queries]
    roles = [role for query in queries for role in canonical_role_rows(query)]

    skills = sorted(skills_by_id.values(), key=lambda row: row["skill_id"])
    queries = sorted(queries, key=lambda row: row["query_id"])
    qrels = sorted(qrels, key=lambda row: (row["query_id"], row["skill_id"]))
    task_skill_sets = sorted(task_skill_sets, key=lambda row: row["query_id"])
    roles = sorted(roles, key=lambda row: (row["query_id"], row["skill_id"]))

    split_counts = {name: len(rows) for name, rows in split_rows.items()}
    stats = build_stats(skills, queries, qrels, split_counts=split_counts)

    return {
        "skills": skills,
        "queries": queries,
        "qrels": qrels,
        "task_skill_sets": task_skill_sets,
        "roles": roles,
        "splits": split_rows,
        "stats": stats,
    }


def write_processed_files(processed_root: Path, processed: Mapping[str, Any]) -> None:
    processed_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(processed_root / "skills.jsonl", processed["skills"])
    write_jsonl(processed_root / "queries.jsonl", processed["queries"])
    write_jsonl(processed_root / "qrels.jsonl", processed["qrels"])
    write_jsonl(processed_root / "task_skill_sets.jsonl", processed["task_skill_sets"])
    write_jsonl(processed_root / "roles.jsonl", processed["roles"])
    write_json(processed_root / "stats.json", processed["stats"])

    splits_root = processed_root / "splits"
    for split_name, rows in processed["splits"].items():
        write_jsonl(splits_root / f"{split_name}.jsonl", rows)


def _canonical_split(source_split: str, index: int) -> str:
    if source_split == "test":
        return "test"
    if source_split == "train" and index % DEV_EVERY_NTH_TRAIN_QUERY == 9:
        return "dev"
    return "train"

