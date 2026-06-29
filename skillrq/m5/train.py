"""Train residual multi-code path selector with coverage supervision."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model import build_residual_selector_model
from ..m4.swanlab_utils import SwanLabLogger
from ..m4.torch_utils import require_torch
from ..splits import is_eval_split
from ..utils.io import read_jsonl, write_json


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
LEVELS = ("l1", "l2", "l3", "l4")


def train_residual_selector(
    data_root: Path,
    output_root: Path,
    epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    embedding_dim: int = 512,
    hidden_dim: int = 1024,
    coverage_weight: float = 1.0,
    max_vocab_size: int = 200000,
    device: str | None = None,
    swanlab_project: str | None = "SkillRQ-M5",
    swanlab_run_name: str | None = None,
) -> Mapping[str, Any]:
    torch = require_torch()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(data_root / "residual_examples.jsonl"))
    if not rows:
        raise ValueError(f"No residual examples found at {data_root / 'residual_examples.jsonl'}")
    vocab = _build_vocab(rows, max_vocab_size)
    code_vocabs = _build_code_vocabs(rows)
    train_rows = [row for row in rows if row.get("split") == "train"] or rows
    dev_rows = [row for row in rows if is_eval_split(row.get("split"))][: max(batch_size * 4, 1)]
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_residual_selector_model(
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
            "method": "residual_selector_coverage",
            "data_root": str(data_root),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "coverage_weight": coverage_weight,
            "device": str(resolved_device),
            "train_examples": len(train_rows),
        },
        tags=["m5", "coverage", "residual-selector"],
    )
    history = []
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            totals = Counter()
            total = 0
            for batch in _batches(train_rows, batch_size):
                token_ids, offsets, labels, gains = _tensorize(batch, vocab, code_vocabs, resolved_device, torch)
                outputs = model(token_ids, offsets)
                code_loss = sum(torch.nn.functional.cross_entropy(outputs["codes"][level], labels[level]) for level in LEVELS)
                coverage_loss = torch.nn.functional.mse_loss(torch.sigmoid(outputs["coverage_gain"]), gains)
                loss = code_loss + coverage_weight * coverage_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                totals["loss"] += float(loss.detach().cpu()) * len(batch)
                totals["code_loss"] += float(code_loss.detach().cpu()) * len(batch)
                totals["coverage_loss"] += float(coverage_loss.detach().cpu()) * len(batch)
                total += len(batch)
            metrics = {
                "epoch": epoch,
                "train_loss": totals["loss"] / max(total, 1),
                "train_code_loss": totals["code_loss"] / max(total, 1),
                "train_coverage_loss": totals["coverage_loss"] / max(total, 1),
                "train_examples": total,
            }
            if dev_rows:
                metrics.update(
                    _evaluate(
                        model,
                        dev_rows,
                        vocab,
                        code_vocabs,
                        resolved_device,
                        torch,
                        batch_size,
                        coverage_weight=coverage_weight,
                    )
                )
            history.append(metrics)
            logger.log(_swanlab_payload(metrics), step=epoch)
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
        "method": "residual_selector_coverage",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "epochs": epochs,
        "batch_size": batch_size,
        "coverage_weight": coverage_weight,
        "history": history,
        "swanlab_project": swanlab_project,
        "swanlab_run_name": swanlab_run_name,
    }
    write_json(output_root / "train_summary.json", summary)
    return summary


def _evaluate(model, rows, vocab, code_vocabs, device, torch, batch_size: int, coverage_weight: float) -> Mapping[str, float]:
    model.eval()
    correct = Counter()
    exact_matches = 0
    total = 0
    code_loss_sum = 0.0
    coverage_loss_sum = 0.0
    with torch.no_grad():
        for batch in _batches(rows, batch_size):
            token_ids, offsets, labels, gains = _tensorize(batch, vocab, code_vocabs, device, torch)
            outputs = model(token_ids, offsets)
            code_loss = sum(torch.nn.functional.cross_entropy(outputs["codes"][level], labels[level]) for level in LEVELS)
            coverage_loss = torch.nn.functional.mse_loss(torch.sigmoid(outputs["coverage_gain"]), gains)
            code_loss_sum += float(code_loss.detach().cpu()) * len(batch)
            coverage_loss_sum += float(coverage_loss.detach().cpu()) * len(batch)
            path_match_mask = None
            for level in LEVELS:
                level_matches = outputs["codes"][level].argmax(dim=-1) == labels[level]
                correct[level] += int(level_matches.sum().detach().cpu())
                path_match_mask = level_matches if path_match_mask is None else path_match_mask & level_matches
            if path_match_mask is not None:
                exact_matches += int(path_match_mask.sum().detach().cpu())
            total += len(batch)
    code_loss_value = code_loss_sum / max(total, 1)
    coverage_loss_value = coverage_loss_sum / max(total, 1)
    result = {
        "dev_loss": code_loss_value + coverage_weight * coverage_loss_value,
        "dev_code_loss": code_loss_value,
        "dev_coverage_loss": coverage_loss_value,
        "dev_path_exact_match": exact_matches / max(total, 1),
    }
    for level in LEVELS:
        result[f"dev_{level}_accuracy"] = correct[level] / max(total, 1)
    return result


def _tensorize(batch, vocab, code_vocabs, device, torch):
    flat_tokens = []
    offsets = []
    labels = {level: [] for level in LEVELS}
    gains = []
    for row in batch:
        offsets.append(len(flat_tokens))
        text = f"{row.get('query') or ''} {row.get('residual_state') or ''}"
        token_ids = [vocab.get(token, vocab["<unk>"]) for token in _tokens(text)] or [vocab["<unk>"]]
        flat_tokens.extend(token_ids)
        path = list(row["code_path"])
        for index, level in enumerate(LEVELS):
            labels[level].append(code_vocabs[level][str(path[index])])
        gains.append(float(row.get("normalized_coverage_gain") or 0.0))
    return (
        torch.tensor(flat_tokens, dtype=torch.long, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
        {level: torch.tensor(values, dtype=torch.long, device=device) for level, values in labels.items()},
        torch.tensor(gains, dtype=torch.float32, device=device),
    )


def _build_vocab(rows: Sequence[Mapping[str, Any]], max_vocab_size: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(_tokens(f"{row.get('query') or ''} {row.get('residual_state') or ''}"))
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


def _swanlab_payload(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "train/loss": metrics.get("train_loss"),
        "train/code_loss": metrics.get("train_code_loss"),
        "train/coverage_loss": metrics.get("train_coverage_loss"),
        "dev/loss": metrics.get("dev_loss"),
        "dev/code_loss": metrics.get("dev_code_loss"),
        "dev/coverage_loss": metrics.get("dev_coverage_loss"),
        "dev/l1_accuracy": metrics.get("dev_l1_accuracy"),
        "dev/l2_accuracy": metrics.get("dev_l2_accuracy"),
        "dev/l3_accuracy": metrics.get("dev_l3_accuracy"),
        "dev/l4_accuracy": metrics.get("dev_l4_accuracy"),
        "dev/path_exact_match": metrics.get("dev_path_exact_match"),
    }


def _batches(rows: Sequence[Mapping[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token]
