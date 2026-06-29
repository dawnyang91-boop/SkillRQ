"""Build M3 CapabilityRQ code assignments."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .assign import (
    assign_capability_code,
    assign_skillret_code,
    assign_skillrouter_code,
    build_capability_role_map,
)
from .cards import write_code_cards
from .quality import build_quality_report
from ..config.schema import PathsConfig
from ..utils.io import read_jsonl, write_json, write_jsonl


DEFAULT_M3_DATASETS = ("toolbench", "api_bank", "skillret", "skillrouter")


def build_m3_codebooks(
    paths: PathsConfig,
    datasets: Sequence[str] = DEFAULT_M3_DATASETS,
    limit_per_dataset: int | None = None,
) -> Mapping[str, Any]:
    datasets = tuple(datasets)
    capability_assignments: list[Mapping[str, Any]] = []
    skill_assignments: list[Mapping[str, Any]] = []

    if "toolbench" in datasets or "api_bank" in datasets:
        capability_assignments = _build_capability_assignments(paths, datasets, limit_per_dataset)
        write_jsonl(paths.capability_processed_root / "code_assignments.jsonl", capability_assignments)
        quality = build_quality_report(capability_assignments)
        write_json(paths.capability_processed_root / "code_quality.json", quality)
    else:
        quality = {"overall": {}, "by_dataset": {}}

    if "skillret" in datasets or "skillrouter" in datasets:
        skill_assignments = _build_skill_assignments(paths, datasets, limit_per_dataset)
        skill_output_root = paths.processed_root / "skill"
        write_jsonl(skill_output_root / "code_assignments.jsonl", skill_assignments)
        skill_quality = build_quality_report(skill_assignments)
        write_json(skill_output_root / "code_quality.json", skill_quality)
    else:
        skill_quality = {"overall": {}, "by_dataset": {}}

    all_assignments = [*capability_assignments, *skill_assignments]
    combined_quality = build_quality_report(all_assignments)
    code_card_paths = write_code_cards(all_assignments, combined_quality, paths.report_root)
    summary = {
        "datasets": list(datasets),
        "limit_per_dataset": limit_per_dataset,
        "capability_assignments": len(capability_assignments),
        "skill_assignments": len(skill_assignments),
        "total_assignments": len(all_assignments),
        "quality": combined_quality,
        "capability_quality_path": str(paths.capability_processed_root / "code_quality.json"),
        "skill_quality_path": str(paths.processed_root / "skill" / "code_quality.json"),
        "code_card_paths": [str(path) for path in code_card_paths],
    }
    write_json(paths.report_root / "code_cards" / "m3_codebook_summary.json", summary)
    return summary


def _build_capability_assignments(
    paths: PathsConfig,
    datasets: Sequence[str],
    limit_per_dataset: int | None,
) -> list[Mapping[str, Any]]:
    role_map = build_capability_role_map(paths.capability_processed_root / "capability_sequences.jsonl")
    counts = {dataset: 0 for dataset in datasets}
    assignments = []
    for row in read_jsonl(paths.capability_processed_root / "capabilities.jsonl"):
        dataset = str(row.get("source_dataset") or "")
        if dataset not in datasets:
            continue
        if limit_per_dataset is not None and counts.get(dataset, 0) >= limit_per_dataset:
            continue
        assignments.append(assign_capability_code(row, role_map))
        counts[dataset] = counts.get(dataset, 0) + 1
    return assignments


def _build_skill_assignments(
    paths: PathsConfig,
    datasets: Sequence[str],
    limit_per_dataset: int | None,
) -> list[Mapping[str, Any]]:
    assignments: list[Mapping[str, Any]] = []
    if "skillret" in datasets:
        count = 0
        for row in read_jsonl(paths.processed_root / "skills.jsonl"):
            if limit_per_dataset is not None and count >= limit_per_dataset:
                break
            assignments.append(assign_skillret_code(row))
            count += 1
    if "skillrouter" in datasets:
        count = 0
        seen: set[str] = set()
        for row in _iter_skillrouter_rows(paths.raw_root / "skillrouter" / "eval_core"):
            skill_id = str(row.get("skill_id") or "")
            if not skill_id or skill_id in seen:
                continue
            seen.add(skill_id)
            if limit_per_dataset is not None and count >= limit_per_dataset:
                break
            assignments.append(assign_skillrouter_code(row))
            count += 1
    return assignments


def _iter_skillrouter_rows(root: Path) -> Iterable[Mapping[str, Any]]:
    for split in ("easy", "hard"):
        for path in sorted((root / split).glob("*.jsonl.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
