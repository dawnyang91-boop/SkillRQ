"""M2 baseline retrieval runner."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from .bm25 import BM25Index
from .data_stats import write_m2_data_stats
from .datasets import ALL_M2_DATASETS, load_retrieval_dataset
from .dense import HashingDenseIndex
from .metrics import evaluate_predictions
from .types import Candidate, Query, RetrievalDataset
from ..config.schema import PathsConfig
from ..utils.io import write_json, write_jsonl


DEFAULT_TOP_K = (1, 5, 10, 20)
DEFAULT_METHODS = ("bm25", "dense")
DEFAULT_DATASETS = ALL_M2_DATASETS


def run_m2_baselines(
    paths: PathsConfig,
    datasets: Sequence[str] = DEFAULT_DATASETS,
    methods: Sequence[str] = DEFAULT_METHODS,
    top_ks: Sequence[int] = DEFAULT_TOP_K,
    max_queries: int | None = 300,
    max_candidates: int | None = 10000,
    run_root: Path | None = None,
) -> Mapping[str, Any]:
    run_root = run_root or paths.run_root / "m2_baseline_retrieval"
    summary: dict[str, Any] = {
        "run_root": str(run_root),
        "datasets": {},
        "methods": list(methods),
        "top_ks": list(top_ks),
        "max_queries": max_queries,
        "max_candidates": max_candidates,
    }
    for dataset_name in datasets:
        data_stats = write_m2_data_stats(paths, dataset_name)
        dataset = load_retrieval_dataset(
            paths,
            dataset_name,
            max_queries=max_queries,
            max_candidates=max_candidates,
        )
        dataset_summary: dict[str, Any] = {
            "data_stats": data_stats,
            "run_query_count": len(dataset.queries),
            "run_candidate_count": len(dataset.candidates),
            "methods": {},
        }
        for method in methods:
            method_summary = _run_method(dataset, method, top_ks, run_root / dataset_name / method)
            dataset_summary["methods"][method] = method_summary
        summary["datasets"][dataset_name] = dataset_summary
    write_json(run_root / "summary.json", summary)
    return summary


def _run_method(
    dataset: RetrievalDataset,
    method: str,
    top_ks: Sequence[int],
    output_root: Path,
) -> Mapping[str, Any]:
    started = perf_counter()
    max_k = max(top_ks)
    index = _build_index(method, dataset.candidates)
    predictions: dict[str, list[str]] = {}
    prediction_rows: list[Mapping[str, Any]] = []
    for query in dataset.queries:
        scored = index.search(query.text, top_k=max_k)
        ranked_ids = [candidate_id for candidate_id, _score in scored]
        predictions[query.query_id] = ranked_ids
        prediction_rows.append(
            {
                "query_id": query.query_id,
                "gold_ids": list(query.gold_ids),
                "sequence_ids": list(query.sequence_ids),
                "predictions": [
                    {"candidate_id": candidate_id, "score": score}
                    for candidate_id, score in scored
                ],
            }
        )

    metrics = evaluate_predictions(dataset.queries, predictions, top_ks=top_ks, task_type=dataset.task_type)
    elapsed = perf_counter() - started
    run_config = {
        "dataset": dataset.name,
        "method": method,
        "top_ks": list(top_ks),
        "query_count": len(dataset.queries),
        "candidate_count": len(dataset.candidates),
        "elapsed_seconds": elapsed,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "metrics.json", metrics)
    write_json(output_root / "run_config.json", run_config)
    write_jsonl(output_root / "predictions.jsonl", prediction_rows)
    return {"metrics": metrics, "run_config": run_config}


def _build_index(method: str, candidates: Sequence[Candidate]) -> Any:
    if method == "bm25":
        return BM25Index(candidates)
    if method == "dense":
        return HashingDenseIndex(candidates)
    raise ValueError(f"Unsupported M2 retrieval method: {method}")
