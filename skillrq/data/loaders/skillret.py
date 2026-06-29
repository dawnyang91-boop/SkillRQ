"""Loader for the raw SkillRet dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

from ...utils.io import read_jsonl


SKILLRET_SPLITS = ("train", "test")


@dataclass(frozen=True)
class SkillRetSplit:
    skills: List[Mapping[str, Any]]
    queries: List[Mapping[str, Any]]
    qrels: List[Mapping[str, Any]]


@dataclass(frozen=True)
class SkillRetRawData:
    train: SkillRetSplit
    test: SkillRetSplit

    def by_split(self) -> Dict[str, SkillRetSplit]:
        return {
            "train": self.train,
            "test": self.test,
        }


def load_skillret(raw_root: Path) -> SkillRetRawData:
    dataset_root = raw_root / "skillret"
    if not dataset_root.exists():
        raise FileNotFoundError(f"SkillRet raw directory not found: {dataset_root}")

    splits = {split: _load_split(dataset_root, split) for split in SKILLRET_SPLITS}
    return SkillRetRawData(train=splits["train"], test=splits["test"])


def _load_split(dataset_root: Path, split: str) -> SkillRetSplit:
    split_root = dataset_root / split
    required_files = {
        "skills": split_root / "skills.jsonl",
        "queries": split_root / "queries.jsonl",
        "qrels": split_root / "qrels.jsonl",
    }
    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing SkillRet files for split {split}: {', '.join(missing)}")

    return SkillRetSplit(
        skills=list(read_jsonl(required_files["skills"])),
        queries=list(read_jsonl(required_files["queries"])),
        qrels=list(read_jsonl(required_files["qrels"])),
    )

