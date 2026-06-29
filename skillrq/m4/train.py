"""Train CapabilityRQ query-to-code models."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model import build_query_code_model
from .swanlab_utils import SwanLabLogger
from .torch_utils import require_torch
from ..utils.io import read_jsonl, write_json


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
LEVELS = ("l1", "l2", "l3", "l4")


def train_capabilityrq(
    data_root: Path,
    output_root: Path,
    epochs: int = 5,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    embedding_dim: int = 256,
    hidden_dim: int = 512,
    max_vocab_size: int = 200000,
    device: str | None = None,
    seed: int = 13,
    swanlab_project: str | None = "SkillRQ-M4",
    swanlab_run_name: str | None = None,
) -> Mapping[str, Any]:
    torch = require_torch()
    torch.manual_seed(seed)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = list(read_jsonl(data_root / "train_pairs.jsonl"))
    if not rows:
        raise ValueError(f"No train pairs found at {data_root / 'train_pairs.jsonl'}")
    vocab = _build_vocab(rows, max_vocab_size=max_vocab_size)
    code_vocabs = _build_code_vocabs(rows)
    train_rows = [row for row in rows if row.get("split") == "train"] or rows
    dev_rows = [row for row in rows if row.get("split") in {"dev", "test"}][: max(batch_size * 4, 1)]

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_query_code_model(
        vocab_size=len(vocab),
        code_vocab_sizes={level: len(code_vocabs[level]) for level in LEVELS},
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    logger = SwanLabLogger(
        project=swanlab_project,
        run_name=swanlab_run_name,
        config={
            "method": "capabilityrq",
            "data_root": str(data_root),
            "output_root": str(output_root),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "max_vocab_size": max_vocab_size,
            "device": str(resolved_device),
            "train_pairs": len(train_rows),
            "all_pairs": len(rows),
            "vocab_size": len(vocab),
            "l1_code_count": len(code_vocabs["l1"]),
            "l2_code_count": len(code_vocabs["l2"]),
            "l3_code_count": len(code_vocabs["l3"]),
            "l4_code_count": len(code_vocabs["l4"]),
        },
        tags=["m4", "capabilityrq", "query-to-code"],
    )

    history = []
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_examples = 0
            for batch in _batches(train_rows, batch_size):
                token_ids, offsets, labels = _tensorize_batch(batch, vocab, code_vocabs, resolved_device, torch)
                logits = model(token_ids, offsets)
                loss = sum(torch.nn.functional.cross_entropy(logits[level], labels[level]) for level in LEVELS)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(batch)
                total_examples += len(batch)
            metrics = {
                "epoch": epoch,
                "train_loss": total_loss / max(total_examples, 1),
                "train_examples": total_examples,
            }
            if dev_rows:
                metrics.update(_evaluate(model, dev_rows, vocab, code_vocabs, resolved_device, torch, batch_size))
            history.append(metrics)
            logger.log(_swanlab_epoch_payload(metrics), step=epoch)
    finally:
        logger.finish()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": len(vocab),
                "code_vocab_sizes": {level: len(code_vocabs[level]) for level in LEVELS},
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "levels": LEVELS,
            },
        },
        output_root / "model.pt",
    )
    (output_root / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "code_vocabs.json").write_text(json.dumps(code_vocabs, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "method": "capabilityrq",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(resolved_device),
        "train_pairs": len(train_rows),
        "all_pairs": len(rows),
        "history": history,
        "swanlab_project": swanlab_project,
        "swanlab_run_name": swanlab_run_name,
    }
    write_json(output_root / "train_summary.json", summary)
    return summary


def _swanlab_epoch_payload(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = {
        "train/loss": metrics.get("train_loss"),
        "train/examples": metrics.get("train_examples"),
    }
    for key, value in metrics.items():
        if key.startswith("dev_"):
            payload[f"dev/{key.removeprefix('dev_')}"] = value
    return payload


def _build_vocab(rows: Sequence[Mapping[str, Any]], max_vocab_size: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(_tokens(str(row.get("query") or "")))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _count in counts.most_common(max_vocab_size - len(vocab)):
        vocab[token] = len(vocab)
    return vocab


def _build_code_vocabs(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    values = {level: [] for level in LEVELS}
    for row in rows:
        path = list(row["code_path"])
        for index, level in enumerate(LEVELS):
            values[level].append(str(path[index]))
    return {level: {code: index for index, code in enumerate(sorted(set(codes)))} for level, codes in values.items()}


def _evaluate(model, rows, vocab, code_vocabs, device, torch, batch_size: int) -> Mapping[str, float]:
    model.eval()
    correct = {level: 0 for level in LEVELS}
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in _batches(rows, batch_size):
            token_ids, offsets, labels = _tensorize_batch(batch, vocab, code_vocabs, device, torch)
            logits = model(token_ids, offsets)
            loss = sum(torch.nn.functional.cross_entropy(logits[level], labels[level]) for level in LEVELS)
            total_loss += float(loss.detach().cpu()) * len(batch)
            for level in LEVELS:
                correct[level] += int((logits[level].argmax(dim=-1) == labels[level]).sum().detach().cpu())
            total += len(batch)
    result = {"dev_loss": total_loss / max(total, 1)}
    for level in LEVELS:
        result[f"dev_{level}_accuracy"] = correct[level] / max(total, 1)
    result["dev_path_exact_match"] = math.prod(result[f"dev_{level}_accuracy"] for level in LEVELS)
    return result


def _tensorize_batch(batch, vocab, code_vocabs, device, torch):
    flat_tokens = []
    offsets = []
    labels = {level: [] for level in LEVELS}
    for row in batch:
        offsets.append(len(flat_tokens))
        token_ids = [vocab.get(token, vocab["<unk>"]) for token in _tokens(str(row.get("query") or ""))]
        flat_tokens.extend(token_ids or [vocab["<unk>"]])
        path = list(row["code_path"])
        for index, level in enumerate(LEVELS):
            labels[level].append(code_vocabs[level][str(path[index])])
    return (
        torch.tensor(flat_tokens, dtype=torch.long, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
        {level: torch.tensor(values, dtype=torch.long, device=device) for level, values in labels.items()},
    )


def _batches(rows: Sequence[Mapping[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token]
