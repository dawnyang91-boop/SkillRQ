"""Joint training for residual code prediction and M7 reranking ablations."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .features import FEATURE_KEYS, ROLES, STAGES, feature_vector
from .joint_model import LEVELS, build_joint_reranker_model
from ..m4.swanlab_utils import SwanLabLogger
from ..m4.torch_utils import require_torch
from ..utils.io import read_jsonl, write_json


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def train_joint_reranker(
    data_root: Path,
    output_root: Path,
    epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    embedding_dim: int = 512,
    hidden_dim: int = 1024,
    code_embedding_dim: int = 128,
    max_vocab_size: int = 300000,
    code_weight: float = 1.0,
    role_weight: float = 0.2,
    stage_weight: float = 0.2,
    order_weight: float = 0.2,
    soft_code_weight: float = 0.1,
    enable_shared_encoder: bool = False,
    enable_soft_code_distribution: bool = False,
    device: str | None = None,
    swanlab_project: str | None = "SkillRQ-M7",
    swanlab_run_name: str | None = None,
) -> Mapping[str, Any]:
    torch = require_torch()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(data_root / "rerank_examples.jsonl"))
    if not rows:
        raise ValueError(f"No rerank examples found at {data_root / 'rerank_examples.jsonl'}")
    vocab = _build_vocab(rows, max_vocab_size)
    code_vocabs = _build_code_vocabs(rows)
    role_vocab = {role: index for index, role in enumerate(ROLES)}
    stage_vocab = {stage: index for index, stage in enumerate(STAGES)}
    train_rows = [row for row in rows if row.get("split") == "train"] or rows
    dev_rows = [row for row in rows if row.get("split") in {"dev", "test"}][: max(batch_size * 4, 1)]
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_joint_reranker_model(
        vocab_size=len(vocab),
        feature_dim=len(FEATURE_KEYS),
        code_vocab_sizes={level: len(code_vocabs[level]) for level in LEVELS},
        role_count=len(role_vocab),
        stage_count=len(stage_vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        code_embedding_dim=code_embedding_dim,
        enable_shared_encoder=enable_shared_encoder,
        enable_soft_code_distribution=enable_soft_code_distribution,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    effective_soft_code_weight = soft_code_weight if enable_soft_code_distribution else 0.0
    logger = SwanLabLogger(
        project=swanlab_project,
        run_name=swanlab_run_name,
        config={
            "method": "m7_joint_residual_reranker",
            "data_root": str(data_root),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "code_weight": code_weight,
            "role_weight": role_weight,
            "stage_weight": stage_weight,
            "order_weight": order_weight,
            "soft_code_weight": soft_code_weight,
            "effective_soft_code_weight": effective_soft_code_weight,
            "enable_shared_encoder": enable_shared_encoder,
            "enable_soft_code_distribution": enable_soft_code_distribution,
            "device": str(resolved_device),
            "train_examples": len(train_rows),
        },
        tags=["m7", "joint", "ablation"],
    )
    history = []
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            totals = Counter()
            total = 0
            for batch in _batches(train_rows, batch_size):
                tensors = _tensorize(batch, vocab, code_vocabs, role_vocab, stage_vocab, resolved_device, torch)
                outputs = model(
                    tensors["query_token_ids"],
                    tensors["query_offsets"],
                    tensors["candidate_token_ids"],
                    tensors["candidate_offsets"],
                    tensors["features"],
                    tensors["code_ids"],
                )
                losses = _compute_losses(outputs, tensors, torch, code_weight, role_weight, stage_weight, order_weight, effective_soft_code_weight)
                optimizer.zero_grad()
                losses["loss"].backward()
                optimizer.step()
                for key, value in losses.items():
                    totals[key] += float(value.detach().cpu()) * len(batch)
                total += len(batch)
            metrics = {
                "epoch": epoch,
                "train_examples": total,
                **{f"train_{key}": totals[key] / max(total, 1) for key in ("loss", "code_loss", "relevance_loss", "role_loss", "stage_loss", "order_loss", "soft_code_loss")},
            }
            if dev_rows:
                metrics.update(_evaluate(model, dev_rows, vocab, code_vocabs, role_vocab, stage_vocab, resolved_device, torch, batch_size))
            history.append(metrics)
            logger.log(_swanlab_payload(metrics), step=epoch)
    finally:
        logger.finish()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": len(vocab),
                "feature_dim": len(FEATURE_KEYS),
                "code_vocab_sizes": {level: len(code_vocabs[level]) for level in LEVELS},
                "role_count": len(role_vocab),
                "stage_count": len(stage_vocab),
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "code_embedding_dim": code_embedding_dim,
                "feature_keys": FEATURE_KEYS,
                "roles": ROLES,
                "stages": STAGES,
                "levels": LEVELS,
                "enable_shared_encoder": enable_shared_encoder,
                "enable_soft_code_distribution": enable_soft_code_distribution,
            },
        },
        output_root / "model.pt",
    )
    (output_root / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "code_vocabs.json").write_text(json.dumps(code_vocabs, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "method": "m7_joint_residual_reranker",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "epochs": epochs,
        "batch_size": batch_size,
        "enable_shared_encoder": enable_shared_encoder,
        "enable_soft_code_distribution": enable_soft_code_distribution,
        "history": history,
        "swanlab_project": swanlab_project,
        "swanlab_run_name": swanlab_run_name,
    }
    write_json(output_root / "train_summary.json", summary)
    return summary


def _compute_losses(outputs, tensors, torch, code_weight, role_weight, stage_weight, order_weight, soft_code_weight):
    relevance_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["relevance"], tensors["labels"])
    roles = tensors["roles"].clamp(0, outputs["role"].shape[-1] - 1)
    stages = tensors["stages"].clamp(0, outputs["stage"].shape[-1] - 1)
    role_loss = torch.nn.functional.cross_entropy(outputs["role"], roles)
    stage_loss = torch.nn.functional.cross_entropy(outputs["stage"], stages)
    order_loss = torch.nn.functional.mse_loss(torch.sigmoid(outputs["order"]), tensors["orders"])
    code_loss = _positive_code_loss(outputs["code_logits"], tensors["code_ids"], tensors["positive_mask"], torch)
    soft_code_target = tensors["positive_mask"].float()
    soft_code_score = outputs["soft_code_scores"].mean(dim=-1)
    soft_code_loss = torch.nn.functional.mse_loss(soft_code_score, soft_code_target)
    loss = (
        relevance_loss
        + code_weight * code_loss
        + role_weight * role_loss
        + stage_weight * stage_loss
        + order_weight * order_loss
        + soft_code_weight * soft_code_loss
    )
    return {
        "loss": loss,
        "code_loss": code_loss,
        "relevance_loss": relevance_loss,
        "role_loss": role_loss,
        "stage_loss": stage_loss,
        "order_loss": order_loss,
        "soft_code_loss": soft_code_loss,
    }


def _positive_code_loss(code_logits, code_ids, positive_mask, torch):
    if not bool(positive_mask.any().detach().cpu()):
        return sum(logits.sum() * 0.0 for logits in code_logits.values())
    losses = []
    for index, level in enumerate(LEVELS):
        labels = code_ids[:, index].clamp(0, code_logits[level].shape[-1] - 1)
        losses.append(torch.nn.functional.cross_entropy(code_logits[level][positive_mask], labels[positive_mask]))
    return sum(losses)


def _evaluate(model, rows, vocab, code_vocabs, role_vocab, stage_vocab, device, torch, batch_size: int) -> Mapping[str, float]:
    model.eval()
    total = 0
    relevance_correct = 0
    role_correct = 0
    stage_correct = 0
    code_exact = 0
    positive_total = 0
    with torch.no_grad():
        for batch in _batches(rows, batch_size):
            tensors = _tensorize(batch, vocab, code_vocabs, role_vocab, stage_vocab, device, torch)
            outputs = model(
                tensors["query_token_ids"],
                tensors["query_offsets"],
                tensors["candidate_token_ids"],
                tensors["candidate_offsets"],
                tensors["features"],
                tensors["code_ids"],
            )
            predictions = (torch.sigmoid(outputs["relevance"]) >= 0.5).float()
            relevance_correct += int((predictions == tensors["labels"]).sum().detach().cpu())
            role_correct += int((outputs["role"].argmax(dim=-1) == tensors["roles"]).sum().detach().cpu())
            stage_correct += int((outputs["stage"].argmax(dim=-1) == tensors["stages"]).sum().detach().cpu())
            path_match_mask = None
            for index, level in enumerate(LEVELS):
                level_matches = outputs["code_logits"][level].argmax(dim=-1) == tensors["code_ids"][:, index]
                path_match_mask = level_matches if path_match_mask is None else path_match_mask & level_matches
            positives = tensors["positive_mask"]
            if path_match_mask is not None and bool(positives.any().detach().cpu()):
                code_exact += int((path_match_mask & positives).sum().detach().cpu())
                positive_total += int(positives.sum().detach().cpu())
            total += len(batch)
    return {
        "dev_relevance_accuracy": relevance_correct / max(total, 1),
        "dev_role_accuracy": role_correct / max(total, 1),
        "dev_stage_accuracy": stage_correct / max(total, 1),
        "dev_code_path_exact_match": code_exact / max(positive_total, 1),
    }


def _tensorize(batch, vocab, code_vocabs, role_vocab, stage_vocab, device, torch):
    query_tokens = []
    query_offsets = []
    candidate_tokens = []
    candidate_offsets = []
    features = []
    labels = []
    roles = []
    stages = []
    orders = []
    code_ids = []
    positive_mask = []
    for row in batch:
        query_offsets.append(len(query_tokens))
        candidate_offsets.append(len(candidate_tokens))
        query_ids = [vocab.get(token, vocab["<unk>"]) for token in _tokens(str(row.get("query") or ""))] or [vocab["<unk>"]]
        candidate_text = f"{row.get('candidate_name') or ''} {row.get('candidate_text') or ''}"
        candidate_ids = [vocab.get(token, vocab["<unk>"]) for token in _tokens(candidate_text)] or [vocab["<unk>"]]
        query_tokens.extend(query_ids)
        candidate_tokens.extend(candidate_ids)
        features.append(feature_vector(row.get("features") or {}))
        labels.append(float(row.get("label") or 0))
        roles.append(role_vocab.get(str(row.get("role_label") or "UNKNOWN"), role_vocab["UNKNOWN"]))
        stages.append(stage_vocab.get(str(row.get("stage_label") or "UNKNOWN"), stage_vocab["UNKNOWN"]))
        orders.append(float(row.get("order_score") or 0.0))
        path = list(row.get("code_path") or [])
        code_ids.append([code_vocabs[level].get(str(path[index]), 0) for index, level in enumerate(LEVELS)])
        positive_mask.append(bool(row.get("label")))
    return {
        "query_token_ids": torch.tensor(query_tokens, dtype=torch.long, device=device),
        "query_offsets": torch.tensor(query_offsets, dtype=torch.long, device=device),
        "candidate_token_ids": torch.tensor(candidate_tokens, dtype=torch.long, device=device),
        "candidate_offsets": torch.tensor(candidate_offsets, dtype=torch.long, device=device),
        "features": torch.tensor(features, dtype=torch.float32, device=device),
        "labels": torch.tensor(labels, dtype=torch.float32, device=device),
        "roles": torch.tensor(roles, dtype=torch.long, device=device),
        "stages": torch.tensor(stages, dtype=torch.long, device=device),
        "orders": torch.tensor(orders, dtype=torch.float32, device=device),
        "code_ids": torch.tensor(code_ids, dtype=torch.long, device=device),
        "positive_mask": torch.tensor(positive_mask, dtype=torch.bool, device=device),
    }


def _build_vocab(rows: Sequence[Mapping[str, Any]], max_vocab_size: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(_tokens(f"{row.get('query') or ''} {row.get('candidate_name') or ''} {row.get('candidate_text') or ''}"))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _count in counts.most_common(max_vocab_size - len(vocab)):
        vocab[token] = len(vocab)
    return vocab


def _build_code_vocabs(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    values = {level: ["<unk>"] for level in LEVELS}
    for row in rows:
        path = list(row.get("code_path") or [])
        if len(path) < len(LEVELS):
            continue
        for index, level in enumerate(LEVELS):
            values[level].append(str(path[index]))
    return {level: {code: index for index, code in enumerate(sorted(set(codes)))} for level, codes in values.items()}


def _swanlab_payload(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "train/loss": metrics.get("train_loss"),
        "train/code_loss": metrics.get("train_code_loss"),
        "train/relevance_loss": metrics.get("train_relevance_loss"),
        "train/role_loss": metrics.get("train_role_loss"),
        "train/stage_loss": metrics.get("train_stage_loss"),
        "train/order_loss": metrics.get("train_order_loss"),
        "train/soft_code_loss": metrics.get("train_soft_code_loss"),
        "dev/relevance_accuracy": metrics.get("dev_relevance_accuracy"),
        "dev/role_accuracy": metrics.get("dev_role_accuracy"),
        "dev/stage_accuracy": metrics.get("dev_stage_accuracy"),
        "dev/code_path_exact_match": metrics.get("dev_code_path_exact_match"),
    }


def _batches(rows: Sequence[Mapping[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if token]
