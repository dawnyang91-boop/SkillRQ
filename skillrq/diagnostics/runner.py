"""Unified diagnostics runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .candidate_pool import diagnose_candidate_pool
from .codebook import diagnose_codebook
from .multipositive import diagnose_multi_positive
from .negatives import diagnose_negative_sampling
from .sequence import diagnose_sequence_chain
from ..utils.io import write_json


def run_diagnostics(
    project_root: Path,
    target: str = "capability",
    output_root: Path | None = None,
    top_ks: Sequence[int] = (5, 10, 20, 50, 100),
    include_joint_predictions: bool = True,
) -> Mapping[str, Any]:
    output_root = output_root or project_root / "reports" / "diagnostics" / target
    output_root.mkdir(parents=True, exist_ok=True)
    m4_data_root = project_root / "data" / "processed" / "m4" / target
    m7_data_root = project_root / "data" / "processed" / "m7" / target
    prediction_paths = _default_prediction_paths(project_root, target, include_joint_predictions)
    summary: dict[str, Any] = {
        "project_root": str(project_root),
        "target": target,
        "output_root": str(output_root),
        "prediction_paths": {name: str(path) for name, path in prediction_paths.items()},
    }
    summary["candidate_pool"] = diagnose_candidate_pool(prediction_paths, output_root, top_ks=top_ks)
    if m4_data_root.exists():
        summary["codebook"] = diagnose_codebook(m4_data_root, output_root)
        summary["multi_positive"] = diagnose_multi_positive(m4_data_root, output_root)
        summary["sequence_chain"] = diagnose_sequence_chain(m4_data_root, prediction_paths, output_root)
    else:
        summary["m4_data_missing"] = str(m4_data_root)
    if m7_data_root.exists() and (m7_data_root / "rerank_examples.jsonl").exists():
        summary["negative_sampling"] = diagnose_negative_sampling(m7_data_root, output_root)
    else:
        summary["m7_data_missing"] = str(m7_data_root)
    write_json(output_root / "diagnostics_summary.json", summary)
    return summary


def _default_prediction_paths(project_root: Path, target: str, include_joint_predictions: bool) -> dict[str, Path]:
    paths = {
        "m4": project_root / "runs" / "m4_query_to_code" / "predictions" / target / "predictions.jsonl",
        "m5": project_root / "runs" / "m5_residual_selector" / "predictions" / target / "predictions.jsonl",
        "m7": project_root / "runs" / "m7_reranker" / "predictions" / target / "reranked_predictions.jsonl",
    }
    if include_joint_predictions:
        joint_root = project_root / "runs" / "m7_joint_reranker" / "predictions" / target
        for name in ("joint_base", "shared_encoder", "soft_code_distribution", "shared_encoder_soft_code"):
            paths[f"m7_joint_{name}"] = joint_root / name / "reranked_predictions.jsonl"
    return paths
