"""Predict query code paths and retrieve explicit candidates."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model import build_query_code_model
from .swanlab_utils import SwanLabLogger
from .torch_utils import require_torch
from ..utils.io import read_jsonl, write_json, write_jsonl


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
LEVELS = ("l1", "l2", "l3", "l4")


def predict_query_codes(
    data_root: Path,
    checkpoint_root: Path,
    output_root: Path,
    top_n_paths: int = 8,
    candidate_budget: int = 100,
    split: str | None = None,
    device: str | None = None,
    swanlab_project: str | None = "SkillRQ-M4",
    swanlab_run_name: str | None = None,
) -> Mapping[str, Any]:
    torch = require_torch()
    output_root.mkdir(parents=True, exist_ok=True)
    model, vocab, code_vocabs = _load_model(checkpoint_root, device, torch)
    reverse_code_vocabs = {
        level: {index: code for code, index in values.items()}
        for level, values in code_vocabs.items()
    }
    candidates = list(read_jsonl(data_root / "candidates.jsonl"))
    queries = [
        row for row in read_jsonl(data_root / "queries.jsonl")
        if split is None or row.get("split") == split
    ]
    prediction_rows = []
    logger = SwanLabLogger(
        project=swanlab_project,
        run_name=swanlab_run_name,
        config={
            "method": "capabilityrq_predict",
            "data_root": str(data_root),
            "checkpoint_root": str(checkpoint_root),
            "output_root": str(output_root),
            "top_n_paths": top_n_paths,
            "candidate_budget": candidate_budget,
            "split": split,
            "device": str(next(model.parameters()).device),
            "candidate_count": len(candidates),
            "query_count": len(queries),
        },
        tags=["m4", "capabilityrq", "predict"],
    )
    for query in queries:
        predicted_paths = _predict_paths(
            model,
            str(query.get("query") or ""),
            vocab,
            reverse_code_vocabs,
            top_n_paths,
            next(model.parameters()).device,
            torch,
        )
        retrieved = _retrieve_candidates(predicted_paths, candidates, candidate_budget)
        prediction_rows.append(
            {
                "query_id": query["query_id"],
                "query": query.get("query"),
                "source_dataset": query.get("source_dataset"),
                "split": query.get("split"),
                "gold_ids": query.get("gold_ids"),
                "predicted_code_paths": predicted_paths,
                "retrieved_capabilities": retrieved,
            }
        )
    write_jsonl(output_root / "predictions.jsonl", prediction_rows)
    summary = {
        "data_root": str(data_root),
        "checkpoint_root": str(checkpoint_root),
        "output_root": str(output_root),
        "queries": len(prediction_rows),
        "top_n_paths": top_n_paths,
        "candidate_budget": candidate_budget,
        "split": split,
        "avg_candidate_pool_size": _avg_candidate_pool_size(prediction_rows),
        "swanlab_project": swanlab_project,
        "swanlab_run_name": swanlab_run_name,
    }
    write_json(output_root / "predict_summary.json", summary)
    logger.log(_swanlab_prediction_payload(summary, prediction_rows), step=1)
    logger.finish()
    return summary


def _avg_candidate_pool_size(prediction_rows: Sequence[Mapping[str, Any]]) -> float:
    if not prediction_rows:
        return 0.0
    return sum(len(row.get("retrieved_capabilities") or []) for row in prediction_rows) / len(prediction_rows)


def _swanlab_prediction_payload(
    summary: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "predict/queries": summary.get("queries"),
        "predict/top_n_paths": summary.get("top_n_paths"),
        "predict/candidate_budget": summary.get("candidate_budget"),
        "predict/avg_candidate_pool_size": summary.get("avg_candidate_pool_size"),
        "predict/split": summary.get("split") or "all",
    }
    for index, row in enumerate(prediction_rows[:3], start=1):
        paths = row.get("predicted_code_paths") or []
        retrieved = row.get("retrieved_capabilities") or []
        payload[f"examples/{index}/query_id"] = row.get("query_id")
        payload[f"examples/{index}/query"] = str(row.get("query") or "")[:500]
        payload[f"examples/{index}/top_code_path"] = str(paths[0].get("codes") if paths else "")
        payload[f"examples/{index}/top_candidate"] = str(retrieved[0].get("candidate_id") if retrieved else "")
    return payload


def _load_model(checkpoint_root: Path, device: str | None, torch):
    vocab = json.loads((checkpoint_root / "vocab.json").read_text(encoding="utf-8"))
    code_vocabs = json.loads((checkpoint_root / "code_vocabs.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_root / "model.pt", map_location=device or "cpu")
    config = checkpoint["config"]
    model = build_query_code_model(
        vocab_size=config["vocab_size"],
        code_vocab_sizes=config["code_vocab_sizes"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(resolved_device)
    model.eval()
    return model, vocab, code_vocabs


def _predict_paths(model, query: str, vocab, reverse_code_vocabs, top_n: int, device, torch):
    token_ids = [vocab.get(token, vocab.get("<unk>", 1)) for token in _tokens(query)] or [vocab.get("<unk>", 1)]
    with torch.no_grad():
        logits = model(
            torch.tensor(token_ids, dtype=torch.long, device=device),
            torch.tensor([0], dtype=torch.long, device=device),
        )
        top_by_level = {}
        for level in LEVELS:
            probabilities = torch.softmax(logits[level][0], dim=-1)
            values, indices = torch.topk(probabilities, k=min(top_n, probabilities.numel()))
            top_by_level[level] = [
                (reverse_code_vocabs[level][int(index.detach().cpu())], float(value.detach().cpu()))
                for value, index in zip(values, indices)
            ]

    beams = [([], 1.0)]
    for level in LEVELS:
        next_beams = []
        for codes, score in beams:
            for code, probability in top_by_level[level]:
                next_beams.append(([*codes, code], score * probability))
        beams = sorted(next_beams, key=lambda item: item[1], reverse=True)[:top_n]
    return [
        {
            "path_id": f"P{index}",
            "codes": codes,
            "role_hint": _role_from_codes(codes),
            "score": score,
        }
        for index, (codes, score) in enumerate(beams, start=1)
    ]


def _retrieve_candidates(
    predicted_paths: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    candidate_budget: int,
) -> list[Mapping[str, Any]]:
    rows = []
    for candidate in candidates:
        candidate_codes = list(candidate.get("code_path") or [])
        best_path = None
        best_score = 0.0
        best_overlap = 0
        for path in predicted_paths:
            predicted_codes = list(path.get("codes") or [])
            overlap = sum(1 for left, right in zip(predicted_codes, candidate_codes) if left == right)
            if overlap == 0:
                continue
            score = float(path.get("score") or 0.0) * (overlap / max(len(candidate_codes), 1))
            if score > best_score:
                best_score = score
                best_path = path
                best_overlap = overlap
        if best_path is None:
            continue
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "name": candidate.get("name"),
                "source_dataset": candidate.get("source_dataset"),
                "matched_code_path": best_path,
                "code_match_score": best_score,
                "matched_levels": best_overlap,
                "code_explanation": candidate.get("code_explanation"),
                "capability_text_evidence": str(candidate.get("text") or "")[:600],
            }
        )
    rows.sort(key=lambda row: (row["code_match_score"], row["matched_levels"]), reverse=True)
    return rows[:candidate_budget]


def _role_from_codes(codes: Sequence[str]) -> str | None:
    if len(codes) < 3:
        return None
    code = codes[2].lower()
    for role in ["start", "support", "check", "finalize", "avoid"]:
        if role in code:
            return role.upper()
    return None


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token]
