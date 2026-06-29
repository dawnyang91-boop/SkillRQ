"""Configuration schemas for SkillRQ."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class PathsConfig:
    """Resolved filesystem paths used by local experiments."""

    raw_root: Path
    project_data_root: Path
    processed_root: Path
    index_root: Path
    cache_root: Path
    run_root: Path
    report_root: Path
    capability_raw_root: Path
    capability_processed_root: Path

    @staticmethod
    def defaults() -> Mapping[str, str]:
        return {
            "raw_root": "/Users/sihan/code/skill-rec/data/raw",
            "project_data_root": "data",
            "processed_root": "data/processed",
            "index_root": "data/indexes",
            "cache_root": "data/cache",
            "run_root": "runs",
            "report_root": "reports",
            "capability_raw_root": "data/raw",
            "capability_processed_root": "data/processed/capability",
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str], project_root: Path) -> "PathsConfig":
        required = cls.defaults().keys()
        missing = sorted(key for key in required if key not in mapping)
        if missing:
            raise ValueError(f"Missing required path config keys: {', '.join(missing)}")

        return cls(
            raw_root=_resolve_path(mapping["raw_root"], project_root),
            project_data_root=_resolve_path(mapping["project_data_root"], project_root),
            processed_root=_resolve_path(mapping["processed_root"], project_root),
            index_root=_resolve_path(mapping["index_root"], project_root),
            cache_root=_resolve_path(mapping["cache_root"], project_root),
            run_root=_resolve_path(mapping["run_root"], project_root),
            report_root=_resolve_path(mapping["report_root"], project_root),
            capability_raw_root=_resolve_path(mapping["capability_raw_root"], project_root),
            capability_processed_root=_resolve_path(mapping["capability_processed_root"], project_root),
        )

    def to_json_dict(self) -> Mapping[str, str]:
        return {
            "raw_root": str(self.raw_root),
            "project_data_root": str(self.project_data_root),
            "processed_root": str(self.processed_root),
            "index_root": str(self.index_root),
            "cache_root": str(self.cache_root),
            "run_root": str(self.run_root),
            "report_root": str(self.report_root),
            "capability_raw_root": str(self.capability_raw_root),
            "capability_processed_root": str(self.capability_processed_root),
        }


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path
