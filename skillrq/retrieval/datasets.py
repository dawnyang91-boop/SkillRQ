"""Dataset adapters for M2 retrieval baselines."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .text import capability_text, skill_text, skillrouter_skill_text
from .types import Candidate, Query, RetrievalDataset
from ..config.schema import PathsConfig
from ..utils.io import read_jsonl


CAPABILITY_DATASETS = {"toolbench", "api_bank"}
SKILL_DATASETS = {"skillret", "skillrouter"}
ALL_M2_DATASETS = ("toolbench", "api_bank", "skillret", "skillrouter")


def load_retrieval_dataset(
    paths: PathsConfig,
    dataset: str,
    max_queries: int | None = None,
    max_candidates: int | None = None,
) -> RetrievalDataset:
    if dataset in CAPABILITY_DATASETS:
        return _load_capability_dataset(paths, dataset, max_queries=max_queries, max_candidates=max_candidates)
    if dataset == "skillret":
        return _load_skillret_dataset(paths, max_queries=max_queries, max_candidates=max_candidates)
    if dataset == "skillrouter":
        return _load_skillrouter_dataset(paths, max_queries=max_queries, max_candidates=max_candidates)
    raise ValueError(f"Unsupported M2 dataset: {dataset}")


def _load_capability_dataset(
    paths: PathsConfig,
    dataset: str,
    max_queries: int | None,
    max_candidates: int | None,
) -> RetrievalDataset:
    forced_ids: set[str] = set()
    non_sequence_rows = []
    sequence_rows = []
    for row in read_jsonl(paths.capability_processed_root / "capability_queries.jsonl"):
        if row.get("source_dataset") != dataset:
            continue
        gold_ids = list(row.get("gold_capability_ids") or [])
        if not gold_ids:
            continue
        has_sequence = bool(row.get("tool_call_sequence"))
        if max_queries is None:
            non_sequence_rows.append(row)
        elif has_sequence and len(sequence_rows) < max_queries:
            sequence_rows.append(row)
        elif not has_sequence and len(non_sequence_rows) < max_queries:
            non_sequence_rows.append(row)
        if max_queries is not None and len(non_sequence_rows) >= max_queries and len(sequence_rows) >= max_queries:
            break

    query_rows = _select_query_rows(non_sequence_rows, sequence_rows, max_queries)
    for row in query_rows:
        forced_ids.update(str(item) for item in row.get("gold_capability_ids") or [])

    candidates: Dict[str, Candidate] = {}
    for row in read_jsonl(paths.capability_processed_root / "capabilities.jsonl"):
        if row.get("source_dataset") != dataset:
            continue
        candidate_id = str(row["capability_id"])
        if not _should_keep(candidate_id, len(candidates), max_candidates, forced_ids):
            continue
        candidates[candidate_id] = Candidate(
            candidate_id=candidate_id,
            text=capability_text(row),
            source_dataset=dataset,
            metadata={
                "capability_type": row.get("capability_type"),
                "name": row.get("name"),
                "category": row.get("category"),
            },
        )

    queries = [
        Query(
            query_id=str(row["query_id"]),
            text=str(row.get("query") or row.get("final_answer") or ""),
            gold_ids=[str(item) for item in row.get("gold_capability_ids") or []],
            source_dataset=dataset,
            source_split=row.get("source_split"),
            sequence_ids=[str(item) for item in row.get("tool_call_sequence") or []],
            metadata={"unique_tools_per_query": row.get("unique_tools_per_query")},
        )
        for row in query_rows
    ]
    return RetrievalDataset(name=dataset, task_type="tool", candidates=list(candidates.values()), queries=queries)


def _load_skillret_dataset(
    paths: PathsConfig,
    max_queries: int | None,
    max_candidates: int | None,
) -> RetrievalDataset:
    query_rows = []
    forced_ids: set[str] = set()
    for row in read_jsonl(paths.processed_root / "queries.jsonl"):
        gold_ids = list(row.get("gold_skill_ids") or [])
        if not gold_ids:
            continue
        query_rows.append(row)
        forced_ids.update(str(item) for item in gold_ids)
        if max_queries is not None and len(query_rows) >= max_queries:
            break

    candidates: Dict[str, Candidate] = {}
    for row in read_jsonl(paths.processed_root / "skills.jsonl"):
        candidate_id = str(row["skill_id"])
        if not _should_keep(candidate_id, len(candidates), max_candidates, forced_ids):
            continue
        candidates[candidate_id] = Candidate(
            candidate_id=candidate_id,
            text=skill_text(row),
            source_dataset="skillret",
            metadata={"name": row.get("name"), "namespace": row.get("namespace")},
        )

    queries = [
        Query(
            query_id=str(row["query_id"]),
            text=str(row.get("query") or ""),
            gold_ids=[str(item) for item in row.get("gold_skill_ids") or []],
            source_dataset="skillret",
            source_split=row.get("source_split"),
        )
        for row in query_rows
    ]
    return RetrievalDataset(name="skillret", task_type="skill", candidates=list(candidates.values()), queries=queries)


def _load_skillrouter_dataset(
    paths: PathsConfig,
    max_queries: int | None,
    max_candidates: int | None,
) -> RetrievalDataset:
    root = paths.raw_root / "skillrouter" / "eval_core"
    relevance = json.loads((root / "relevance.json").read_text(encoding="utf-8"))
    query_rows = []
    forced_ids: set[str] = set()
    for row in read_jsonl(root / "tasks.jsonl"):
        if row.get("excluded"):
            continue
        task_id = str(row["task_id"])
        rel = relevance.get(task_id) or {}
        gold_ids = [str(item) for item in rel.get("gt_skill_ids") or row.get("skill_names") or []]
        if not gold_ids:
            continue
        query_rows.append((row, gold_ids))
        forced_ids.update(gold_ids)
        forced_ids.update(str(item) for item in (rel.get("relevance") or {}).keys())
        if max_queries is not None and len(query_rows) >= max_queries:
            break

    candidates: Dict[str, Candidate] = {}
    for row in _iter_skillrouter_candidates(root):
        candidate_id = str(row["skill_id"])
        if not _should_keep(candidate_id, len(candidates), max_candidates, forced_ids):
            continue
        candidates[candidate_id] = Candidate(
            candidate_id=candidate_id,
            text=skillrouter_skill_text(row),
            source_dataset="skillrouter",
            metadata={"name": row.get("name"), "source": row.get("source")},
        )

    queries = [
        Query(
            query_id=str(row["task_id"]),
            text=str(row.get("instruction_text") or ""),
            gold_ids=gold_ids,
            source_dataset="skillrouter",
            source_split=str(row.get("difficulty") or "eval_core"),
            metadata={"domain": row.get("domain"), "num_skills": row.get("num_skills")},
        )
        for row, gold_ids in query_rows
    ]
    return RetrievalDataset(name="skillrouter", task_type="skill", candidates=list(candidates.values()), queries=queries)


def _iter_skillrouter_candidates(root: Path) -> Iterable[Mapping[str, Any]]:
    for split in ("easy", "hard"):
        for path in sorted((root / split).glob("*.jsonl.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)


def _should_keep(candidate_id: str, current_count: int, max_candidates: int | None, forced_ids: set[str]) -> bool:
    return max_candidates is None or current_count < max_candidates or candidate_id in forced_ids


def _select_query_rows(
    non_sequence_rows: list[Mapping[str, Any]],
    sequence_rows: list[Mapping[str, Any]],
    max_queries: int | None,
) -> list[Mapping[str, Any]]:
    if max_queries is None:
        return [*non_sequence_rows, *sequence_rows]
    if not non_sequence_rows or not sequence_rows:
        return [*non_sequence_rows, *sequence_rows][:max_queries]

    non_sequence_target = max_queries // 2
    sequence_target = max_queries - non_sequence_target
    selected = [*non_sequence_rows[:non_sequence_target], *sequence_rows[:sequence_target]]
    selected_ids = {str(row["query_id"]) for row in selected}
    if len(selected) < max_queries:
        for row in [*non_sequence_rows[non_sequence_target:], *sequence_rows[sequence_target:]]:
            if str(row["query_id"]) in selected_ids:
                continue
            selected.append(row)
            if len(selected) >= max_queries:
                break
    return selected
