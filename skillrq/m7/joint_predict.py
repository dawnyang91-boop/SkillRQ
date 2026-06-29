"""Predict with the joint residual-code and reranking model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .features import build_feature_dict, feature_vector, infer_stage, normalize_role, support_evidence
from .joint_model import LEVELS, build_joint_reranker_model
from ..m4.swanlab_utils import SwanLabLogger
from ..m4.torch_utils import require_torch
from ..utils.io import read_jsonl, write_json, write_jsonl


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def predict_joint_reranked_capabilities(
    prediction_path: Path,
    m4_data_root: Path,
    checkpoint_root: Path,
    output_root: Path,
    top_k: int = 100,
    device: str | None = None,
    swanlab_project: str | None = "SkillRQ-M7",
    swanlab_run_name: str | None = None,
) -> Mapping[str, Any]:
    torch = require_torch()
    output_root.mkdir(parents=True, exist_ok=True)
    model, vocab, code_vocabs, roles, stages, config = _load_model(checkpoint_root, device, torch)
    candidates = {str(row["candidate_id"]): row for row in read_jsonl(m4_data_root / "candidates.jsonl")}
    query_metadata = {str(row["query_id"]): row for row in read_jsonl(m4_data_root / "queries.jsonl")}
    prediction_rows = list(read_jsonl(prediction_path))
    logger = SwanLabLogger(
        project=swanlab_project,
        run_name=swanlab_run_name,
        config={
            "method": "m7_joint_rerank_predict",
            "prediction_path": str(prediction_path),
            "m4_data_root": str(m4_data_root),
            "checkpoint_root": str(checkpoint_root),
            "top_k": top_k,
            "enable_shared_encoder": config.get("enable_shared_encoder"),
            "enable_soft_code_distribution": config.get("enable_soft_code_distribution"),
            "query_count": len(prediction_rows),
        },
        tags=["m7", "joint", "predict"],
    )
    output_rows = []
    for row in prediction_rows:
        query_id = str(row["query_id"])
        metadata = query_metadata.get(query_id, {})
        query_text = str(row.get("query") or metadata.get("query") or "")
        sequence_ids = [str(item) for item in metadata.get("sequence_ids") or row.get("sequence_ids") or []]
        scored = []
        seen = set()
        for item in _flatten_prediction_candidates(row):
            candidate_id = str(item.get("candidate_id"))
            if candidate_id in seen or candidate_id not in candidates:
                continue
            seen.add(candidate_id)
            candidate = {**candidates[candidate_id], **item}
            features = build_feature_dict(
                query=query_text,
                candidate=candidate,
                matched_levels=int(item.get("matched_levels") or 0),
                code_match_score=float(item.get("code_match_score") or 0.0),
                coverage_gain_score=float(item.get("predicted_coverage_gain") or item.get("coverage_gain_score") or 0.0),
                step_index=item.get("step_index"),
            )
            outputs = _score_candidate(
                model,
                vocab,
                code_vocabs,
                query_text,
                candidate,
                features,
                roles,
                stages,
                next(model.parameters()).device,
                torch,
            )
            final_score = (
                outputs["relevance_score"]
                + 0.15 * features["code_match_score"]
                + 0.10 * features["coverage_gain_score"]
                + 0.10 * outputs["soft_code_score"]
                + 0.05 * outputs["order_score"]
                - 0.10 * features["generic_penalty"]
                - 0.25 * features["constraint_violation_penalty"]
            )
            scored.append(
                {
                    "candidate_id": candidate_id,
                    "name": candidate.get("name"),
                    "source_dataset": candidate.get("source_dataset"),
                    "semantic_id": candidate.get("semantic_id"),
                    "matched_code_path": item.get("matched_code_path") or item.get("codes") or candidate.get("code_path"),
                    "suggested_role": outputs["role"],
                    "execution_stage": outputs["stage"],
                    "relevance_score": outputs["relevance_score"],
                    "optional_order_score": outputs["order_score"],
                    "soft_code_score": outputs["soft_code_score"],
                    "final_score": final_score,
                    "features": features,
                    "compact_support_evidence": support_evidence(query_text, candidate),
                    "code_explanation": candidate.get("code_explanation"),
                }
            )
        scored.sort(key=lambda value: value["final_score"], reverse=True)
        reranked = scored[:top_k]
        predicted_order = _predict_order(reranked)
        output_rows.append(
            {
                "query_id": query_id,
                "query": query_text,
                "source_dataset": row.get("source_dataset") or metadata.get("source_dataset"),
                "split": row.get("split") or metadata.get("split"),
                "gold_ids": row.get("gold_ids") or metadata.get("gold_ids"),
                "sequence_ids": sequence_ids,
                "reranked_capabilities": reranked,
                "predicted_tool_set": [item["candidate_id"] for item in reranked],
                "predicted_first_tool": predicted_order[0] if predicted_order else None,
                "predicted_tool_order": predicted_order,
            }
        )
    write_jsonl(output_root / "reranked_predictions.jsonl", output_rows)
    summary = {
        "prediction_path": str(prediction_path),
        "m4_data_root": str(m4_data_root),
        "checkpoint_root": str(checkpoint_root),
        "output_root": str(output_root),
        "queries": len(output_rows),
        "top_k": top_k,
        "enable_shared_encoder": config.get("enable_shared_encoder"),
        "enable_soft_code_distribution": config.get("enable_soft_code_distribution"),
        "avg_reranked_candidates": sum(len(row["reranked_capabilities"]) for row in output_rows) / max(len(output_rows), 1),
    }
    write_json(output_root / "predict_summary.json", summary)
    logger.log(
        {
            "predict/queries": summary["queries"],
            "predict/top_k": top_k,
            "predict/avg_reranked_candidates": summary["avg_reranked_candidates"],
            "predict/enable_shared_encoder": str(summary["enable_shared_encoder"]),
            "predict/enable_soft_code_distribution": str(summary["enable_soft_code_distribution"]),
        },
        step=1,
    )
    logger.finish()
    return summary


def _load_model(checkpoint_root: Path, device: str | None, torch):
    vocab = json.loads((checkpoint_root / "vocab.json").read_text(encoding="utf-8"))
    code_vocabs = json.loads((checkpoint_root / "code_vocabs.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_root / "model.pt", map_location=device or "cpu")
    config = checkpoint["config"]
    roles = list(config["roles"])
    stages = list(config["stages"])
    model = build_joint_reranker_model(
        vocab_size=config["vocab_size"],
        feature_dim=config["feature_dim"],
        code_vocab_sizes=config["code_vocab_sizes"],
        role_count=config["role_count"],
        stage_count=config["stage_count"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        code_embedding_dim=config["code_embedding_dim"],
        enable_shared_encoder=bool(config.get("enable_shared_encoder")),
        enable_soft_code_distribution=bool(config.get("enable_soft_code_distribution")),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(resolved_device)
    model.eval()
    return model, vocab, code_vocabs, roles, stages, config


def _score_candidate(model, vocab, code_vocabs, query, candidate, features, roles, stages, device, torch) -> Mapping[str, Any]:
    query_token_ids = [vocab.get(token, vocab.get("<unk>", 1)) for token in _tokens(query)] or [vocab.get("<unk>", 1)]
    candidate_text = f"{candidate.get('name') or ''} {candidate.get('text') or candidate.get('capability_text_evidence') or ''}"
    candidate_token_ids = [vocab.get(token, vocab.get("<unk>", 1)) for token in _tokens(candidate_text)] or [vocab.get("<unk>", 1)]
    path = list(candidate.get("code_path") or candidate.get("matched_code_path") or [])
    code_ids = [[code_vocabs[level].get(str(path[index]) if index < len(path) else "<unk>", 0) for index, level in enumerate(LEVELS)]]
    with torch.no_grad():
        outputs = model(
            torch.tensor(query_token_ids, dtype=torch.long, device=device),
            torch.tensor([0], dtype=torch.long, device=device),
            torch.tensor(candidate_token_ids, dtype=torch.long, device=device),
            torch.tensor([0], dtype=torch.long, device=device),
            torch.tensor([feature_vector(features)], dtype=torch.float32, device=device),
            torch.tensor(code_ids, dtype=torch.long, device=device),
        )
        relevance_score = float(torch.sigmoid(outputs["relevance"])[0].detach().cpu())
        order_score = float(torch.sigmoid(outputs["order"])[0].detach().cpu())
        soft_code_score = float(outputs["soft_code_scores"].mean(dim=-1)[0].detach().cpu())
        role = roles[int(outputs["role"].argmax(dim=-1)[0].detach().cpu())]
        stage = stages[int(outputs["stage"].argmax(dim=-1)[0].detach().cpu())]
    if role == "UNKNOWN":
        role = normalize_role(candidate.get("role_hint"))
    if stage == "UNKNOWN":
        stage = infer_stage(role, None, None)
    return {
        "relevance_score": relevance_score,
        "order_score": order_score,
        "soft_code_score": soft_code_score,
        "role": role,
        "stage": stage,
    }


def _flatten_prediction_candidates(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if row.get("residual_code_paths"):
        rows = []
        for path in row.get("residual_code_paths") or []:
            for candidate in path.get("retrieved_capabilities") or []:
                rows.append(
                    {
                        **candidate,
                        "step_index": path.get("step_index"),
                        "predicted_coverage_gain": path.get("predicted_coverage_gain"),
                        "codes": path.get("codes"),
                    }
                )
        return rows
    return list(row.get("retrieved_capabilities") or [])


def _predict_order(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    stage_priority = {"FIRST": 4, "START": 4, "MIDDLE": 3, "SUPPORT": 3, "CHECK": 2, "FINAL": 1, "FINALIZE": 1, "AVOID": -1}
    ordered = sorted(
        rows,
        key=lambda item: (
            stage_priority.get(str(item.get("execution_stage") or ""), stage_priority.get(str(item.get("suggested_role") or ""), 0)),
            float(item.get("optional_order_score") or 0.0),
            float(item.get("final_score") or 0.0),
        ),
        reverse=True,
    )
    return [str(item["candidate_id"]) for item in ordered]


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if token]
