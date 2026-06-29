"""Train M7 role-aware and sequence-aware reranker."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .features import FEATURE_KEYS, ROLES, STAGES, feature_vector
from .model import build_reranker_model
from ..m4.swanlab_utils import SwanLabLogger
from ..m4.torch_utils import require_torch
from ..utils.io import read_jsonl, write_json


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def train_reranker(
    data_root: Path,
    output_root: Path,
    epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    embedding_dim: int = 512,
    hidden_dim: int = 1024,
    max_vocab_size: int = 300000,
    role_weight: float = 0.2,
    stage_weight: float = 0.2,
    order_weight: float = 0.2,
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
    role_vocab = {role: index for index, role in enumerate(ROLES)}
    stage_vocab = {stage: index for index, stage in enumerate(STAGES)}
    train_rows = [row for row in rows if row.get("split") == "train"] or rows
    dev_rows = [row for row in rows if row.get("split") in {"dev", "test"}][: max(batch_size * 4, 1)]
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_reranker_model(
        vocab_size=len(vocab),
        feature_dim=len(FEATURE_KEYS),
        role_count=len(role_vocab),
        stage_count=len(stage_vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    logger = SwanLabLogger(
        project=swanlab_project,
        run_name=swanlab_run_name,
        config={
            "method": "m7_role_sequence_reranker",
            "data_root": str(data_root),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "role_weight": role_weight,
            "stage_weight": stage_weight,
            "order_weight": order_weight,
            "device": str(resolved_device),
            "train_examples": len(train_rows),
        },
        tags=["m7", "reranker", "role", "sequence"],
    )
    history = []
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            totals = Counter()
            total = 0
            for batch in _batches(train_rows, batch_size):
                token_ids, offsets, features, labels, roles, stages, orders = _tensorize(
                    batch, vocab, role_vocab, stage_vocab, resolved_device, torch
                )
                outputs = model(token_ids, offsets, features)
                relevance_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["relevance"], labels)
                role_loss = torch.nn.functional.cross_entropy(outputs["role"], roles)
                stage_loss = torch.nn.functional.cross_entropy(outputs["stage"], stages)
                order_loss = torch.nn.functional.mse_loss(torch.sigmoid(outputs["order"]), orders)
                loss = relevance_loss + role_weight * role_loss + stage_weight * stage_loss + order_weight * order_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                totals["loss"] += float(loss.detach().cpu()) * len(batch)
                totals["relevance_loss"] += float(relevance_loss.detach().cpu()) * len(batch)
                totals["role_loss"] += float(role_loss.detach().cpu()) * len(batch)
                totals["stage_loss"] += float(stage_loss.detach().cpu()) * len(batch)
                totals["order_loss"] += float(order_loss.detach().cpu()) * len(batch)
                total += len(batch)
            metrics = {
                "epoch": epoch,
                "train_loss": totals["loss"] / max(total, 1),
                "train_relevance_loss": totals["relevance_loss"] / max(total, 1),
                "train_role_loss": totals["role_loss"] / max(total, 1),
                "train_stage_loss": totals["stage_loss"] / max(total, 1),
                "train_order_loss": totals["order_loss"] / max(total, 1),
                "train_examples": total,
            }
            if dev_rows:
                metrics.update(_evaluate(model, dev_rows, vocab, role_vocab, stage_vocab, resolved_device, torch, batch_size))
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
                "role_count": len(role_vocab),
                "stage_count": len(stage_vocab),
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "feature_keys": FEATURE_KEYS,
                "roles": ROLES,
                "stages": STAGES,
            },
        },
        output_root / "model.pt",
    )
    (output_root / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "method": "m7_role_sequence_reranker",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "epochs": epochs,
        "batch_size": batch_size,
        "history": history,
        "swanlab_project": swanlab_project,
        "swanlab_run_name": swanlab_run_name,
    }
    write_json(output_root / "train_summary.json", summary)
    return summary


def _evaluate(model, rows, vocab, role_vocab, stage_vocab, device, torch, batch_size: int) -> Mapping[str, float]:
    model.eval()
    total = 0
    relevance_correct = 0
    role_correct = 0
    stage_correct = 0
    order_loss_sum = 0.0
    with torch.no_grad():
        for batch in _batches(rows, batch_size):
            token_ids, offsets, features, labels, roles, stages, orders = _tensorize(
                batch, vocab, role_vocab, stage_vocab, device, torch
            )
            outputs = model(token_ids, offsets, features)
            predictions = (torch.sigmoid(outputs["relevance"]) >= 0.5).float()
            relevance_correct += int((predictions == labels).sum().detach().cpu())
            role_correct += int((outputs["role"].argmax(dim=-1) == roles).sum().detach().cpu())
            stage_correct += int((outputs["stage"].argmax(dim=-1) == stages).sum().detach().cpu())
            order_loss = torch.nn.functional.mse_loss(torch.sigmoid(outputs["order"]), orders)
            order_loss_sum += float(order_loss.detach().cpu()) * len(batch)
            total += len(batch)
    return {
        "dev_relevance_accuracy": relevance_correct / max(total, 1),
        "dev_role_accuracy": role_correct / max(total, 1),
        "dev_stage_accuracy": stage_correct / max(total, 1),
        "dev_order_mse": order_loss_sum / max(total, 1),
    }


def _tensorize(batch, vocab, role_vocab, stage_vocab, device, torch):
    flat_tokens = []
    offsets = []
    features = []
    labels = []
    roles = []
    stages = []
    orders = []
    for row in batch:
        offsets.append(len(flat_tokens))
        text = f"{row.get('query') or ''} {row.get('candidate_name') or ''} {row.get('candidate_text') or ''}"
        token_ids = [vocab.get(token, vocab["<unk>"]) for token in _tokens(text)] or [vocab["<unk>"]]
        flat_tokens.extend(token_ids)
        features.append(feature_vector(row.get("features") or {}))
        labels.append(float(row.get("label") or 0))
        roles.append(role_vocab.get(str(row.get("role_label") or "UNKNOWN"), role_vocab["UNKNOWN"]))
        stages.append(stage_vocab.get(str(row.get("stage_label") or "UNKNOWN"), stage_vocab["UNKNOWN"]))
        orders.append(float(row.get("order_score") or 0.0))
    return (
        torch.tensor(flat_tokens, dtype=torch.long, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
        torch.tensor(features, dtype=torch.float32, device=device),
        torch.tensor(labels, dtype=torch.float32, device=device),
        torch.tensor(roles, dtype=torch.long, device=device),
        torch.tensor(stages, dtype=torch.long, device=device),
        torch.tensor(orders, dtype=torch.float32, device=device),
    )


def _build_vocab(rows: Sequence[Mapping[str, Any]], max_vocab_size: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(_tokens(f"{row.get('query') or ''} {row.get('candidate_name') or ''} {row.get('candidate_text') or ''}"))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _count in counts.most_common(max_vocab_size - len(vocab)):
        vocab[token] = len(vocab)
    return vocab


def _swanlab_payload(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "train/loss": metrics.get("train_loss"),
        "train/relevance_loss": metrics.get("train_relevance_loss"),
        "train/role_loss": metrics.get("train_role_loss"),
        "train/stage_loss": metrics.get("train_stage_loss"),
        "train/order_loss": metrics.get("train_order_loss"),
        "dev/relevance_accuracy": metrics.get("dev_relevance_accuracy"),
        "dev/role_accuracy": metrics.get("dev_role_accuracy"),
        "dev/stage_accuracy": metrics.get("dev_stage_accuracy"),
        "dev/order_mse": metrics.get("dev_order_mse"),
    }


def _batches(rows: Sequence[Mapping[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if token]
