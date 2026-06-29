"""Sequence evaluation-chain diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .pools import load_prediction_rows
from ..utils.io import read_jsonl, write_json, write_jsonl


def diagnose_sequence_chain(
    m4_data_root: Path,
    prediction_paths: Mapping[str, Path],
    output_root: Path,
    max_cases: int = 100,
) -> Mapping[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    query_metadata = {str(row["query_id"]): row for row in read_jsonl(m4_data_root / "queries.jsonl")}
    metadata_sequence_queries = {
        query_id: row
        for query_id, row in query_metadata.items()
        if row.get("sequence_ids")
    }
    summaries: dict[str, Any] = {}
    cases = []
    for name, path in prediction_paths.items():
        rows = load_prediction_rows(path)
        if not rows:
            summaries[name] = {"prediction_path": str(path), "available": False}
            continue
        direct_sequence = 0
        joinable_sequence = 0
        predicted_order = 0
        missing_sequence_cases = []
        for row in rows:
            query_id = str(row.get("query_id"))
            if row.get("sequence_ids"):
                direct_sequence += 1
            if query_metadata.get(query_id, {}).get("sequence_ids"):
                joinable_sequence += 1
                if not row.get("sequence_ids") and len(missing_sequence_cases) < max_cases:
                    missing_sequence_cases.append(
                        {
                            "source": name,
                            "query_id": query_id,
                            "metadata_sequence_ids": query_metadata[query_id].get("sequence_ids"),
                            "prediction_has_sequence_ids": bool(row.get("sequence_ids")),
                        }
                    )
            if row.get("predicted_tool_order"):
                predicted_order += 1
        summaries[name] = {
            "prediction_path": str(path),
            "available": True,
            "prediction_rows": len(rows),
            "rows_with_sequence_ids_in_prediction": direct_sequence,
            "rows_joinable_to_m4_sequence_ids": joinable_sequence,
            "rows_with_predicted_tool_order": predicted_order,
            "sequence_eval_possible_by_join_ratio": joinable_sequence / max(len(rows), 1),
            "sequence_eval_direct_ratio": direct_sequence / max(len(rows), 1),
        }
        cases.extend(missing_sequence_cases)
    result = {
        "m4_data_root": str(m4_data_root),
        "m4_queries": len(query_metadata),
        "m4_queries_with_sequence_ids": len(metadata_sequence_queries),
        "prediction_sequence_diagnostics": summaries,
    }
    write_json(output_root / "sequence_chain_diagnostics.json", result)
    write_jsonl(output_root / "sequence_missing_in_prediction_cases.jsonl", cases)
    return result
