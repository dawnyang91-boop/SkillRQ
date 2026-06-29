"""Predict residual code paths and retrieve coverage-aware candidates."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model import build_residual_selector_model
from ..m4.swanlab_utils import SwanLabLogger
from ..m4.torch_utils import require_torch
from ..utils.io import read_jsonl, write_json, write_jsonl


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
LEVELS = ("l1", "l2", "l3", "l4")


def predict_residual_paths(
    data_root: Path,
    checkpoint_root: Path,
    output_root: Path,
    max_steps: int = 6,
    top_n_paths: int = 8,
    candidates_per_step: int = 20,
    split: str | None = None,
    device: str | None = None,
    swanlab_project: str | None = "SkillRQ-M5",
    swanlab_run_name: str | None = None,
) -> Mapping[str, Any]:
    torch = require_torch()
    output_root.mkdir(parents=True, exist_ok=True)
    model, vocab, code_vocabs = _load_model(checkpoint_root, device, torch)
    reverse_code_vocabs = {level: {idx: code for code, idx in vocab_.items()} for level, vocab_ in code_vocabs.items()}
    candidates = list(read_jsonl(data_root / "candidates.jsonl"))
    queries = [row for row in read_jsonl(data_root / "queries.jsonl") if split is None or row.get("split") == split]
    logger = SwanLabLogger(
        project=swanlab_project,
        run_name=swanlab_run_name,
        config={
            "method": "m5_residual_predict",
            "data_root": str(data_root),
            "checkpoint_root": str(checkpoint_root),
            "max_steps": max_steps,
            "top_n_paths": top_n_paths,
            "candidates_per_step": candidates_per_step,
            "split": split,
            "query_count": len(queries),
        },
        tags=["m5", "coverage", "predict"],
    )
    rows = []
    for query in queries:
        covered: set[str] = set()
        gold_ids = set(str(item) for item in query.get("gold_ids") or [])
        used_semantic_ids: set[str] = set()
        residual_paths = []
        for step_index in range(max_steps):
            residual_state = _residual_state(step_index, covered)
            predicted = _predict_paths(
                model,
                f"{query.get('query') or ''} {residual_state}",
                vocab,
                reverse_code_vocabs,
                top_n_paths,
                next(model.parameters()).device,
                torch,
            )
            selected = None
            retrieved = []
            for path in predicted:
                semantic_id = "/".join(path["codes"])
                if semantic_id in used_semantic_ids:
                    continue
                retrieved = _retrieve_for_path(path, candidates, covered, candidates_per_step)
                selected = path
                used_semantic_ids.add(semantic_id)
                break
            if selected is None:
                break
            new_ids = [item["candidate_id"] for item in retrieved if item["candidate_id"] not in covered]
            covered.update(new_ids)
            residual_paths.append(
                {
                    "step_index": step_index,
                    "semantic_id": "/".join(selected["codes"]),
                    "codes": selected["codes"],
                    "role_hint": selected.get("role_hint"),
                    "score": selected["score"],
                    "predicted_coverage_gain": selected.get("predicted_coverage_gain"),
                    "retrieved_capabilities": retrieved,
                    "new_candidate_count": len(new_ids),
                }
            )
            if gold_ids and gold_ids.issubset(covered):
                break
        rows.append(
            {
                "query_id": query["query_id"],
                "query": query.get("query"),
                "source_dataset": query.get("source_dataset"),
                "split": query.get("split"),
                "gold_ids": query.get("gold_ids"),
                "residual_code_paths": residual_paths,
            }
        )
    write_jsonl(output_root / "predictions.jsonl", rows)
    summary = {
        "data_root": str(data_root),
        "checkpoint_root": str(checkpoint_root),
        "output_root": str(output_root),
        "queries": len(rows),
        "max_steps": max_steps,
        "top_n_paths": top_n_paths,
        "candidates_per_step": candidates_per_step,
        "split": split,
        "avg_steps": sum(len(row["residual_code_paths"]) for row in rows) / max(len(rows), 1),
    }
    write_json(output_root / "predict_summary.json", summary)
    logger.log({"predict/queries": summary["queries"], "predict/avg_steps": summary["avg_steps"]}, step=1)
    logger.finish()
    return summary


def _load_model(checkpoint_root: Path, device: str | None, torch):
    vocab = json.loads((checkpoint_root / "vocab.json").read_text(encoding="utf-8"))
    code_vocabs = json.loads((checkpoint_root / "code_vocabs.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_root / "model.pt", map_location=device or "cpu")
    config = checkpoint["config"]
    model = build_residual_selector_model(
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


def _predict_paths(model, text, vocab, reverse_code_vocabs, top_n, device, torch):
    token_ids = [vocab.get(token, vocab.get("<unk>", 1)) for token in _tokens(text)] or [vocab.get("<unk>", 1)]
    with torch.no_grad():
        outputs = model(torch.tensor(token_ids, dtype=torch.long, device=device), torch.tensor([0], dtype=torch.long, device=device))
        top_by_level = {}
        for level in LEVELS:
            probs = torch.softmax(outputs["codes"][level][0], dim=-1)
            values, indices = torch.topk(probs, k=min(top_n, probs.numel()))
            top_by_level[level] = [(reverse_code_vocabs[level][int(i.detach().cpu())], float(v.detach().cpu())) for v, i in zip(values, indices)]
        gain = float(torch.sigmoid(outputs["coverage_gain"])[0].detach().cpu())
    beams = [([], 1.0)]
    for level in LEVELS:
        beams = sorted(
            [([*codes, code], score * prob) for codes, score in beams for code, prob in top_by_level[level]],
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]
    return [
        {"codes": codes, "score": score * gain, "predicted_coverage_gain": gain, "role_hint": _role(codes)}
        for codes, score in beams
    ]


def _retrieve_for_path(path, candidates, covered: set[str], limit: int):
    rows = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate_codes = list(candidate.get("code_path") or [])
        overlap = sum(1 for left, right in zip(path["codes"], candidate_codes) if left == right)
        if overlap == 0:
            continue
        novelty = 0.5 if candidate_id not in covered else -0.5
        score = float(path["score"]) * (overlap / max(len(candidate_codes), 1)) + novelty
        rows.append(
            {
                "candidate_id": candidate_id,
                "name": candidate.get("name"),
                "source_dataset": candidate.get("source_dataset"),
                "code_match_score": score,
                "matched_levels": overlap,
                "code_explanation": candidate.get("code_explanation"),
                "capability_text_evidence": str(candidate.get("text") or "")[:600],
            }
        )
    rows.sort(key=lambda row: (row["code_match_score"], row["matched_levels"]), reverse=True)
    return rows[:limit]


def _residual_state(step_index: int, covered: set[str]) -> str:
    return f"step={step_index} covered={' '.join(sorted(covered))}"


def _role(codes: Sequence[str]) -> str | None:
    if len(codes) < 3:
        return None
    for role in ("start", "support", "check", "finalize", "avoid"):
        if role in codes[2].lower():
            return role.upper()
    return None


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token]
