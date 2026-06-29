"""Shared retrieval data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    text: str
    source_dataset: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    gold_ids: Sequence[str]
    source_dataset: str
    source_split: str | None = None
    sequence_ids: Sequence[str] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalDataset:
    name: str
    task_type: str
    candidates: Sequence[Candidate]
    queries: Sequence[Query]
