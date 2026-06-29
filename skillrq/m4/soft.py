"""Soft multi-path M4 query-to-code training and prediction."""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model import build_soft_multipath_code_model
from .swanlab_utils import SwanLabLogger
from .torch_utils import require_torch
from ..splits import is_eval_split
from ..utils.io import read_jsonl, write_json, write_jsonl


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
LEVELS = ("l1", "l2", "l3", "l4")
MODEL_KIND = "soft-multipath"


def train_soft_multipath_code_predictor(
    data_root: Path,
    output_root: Path,
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    embedding_dim: int = 256,
    hidden_dim: int = 512,
    code_embedding_dim: int = 128,
    max_vocab_size: int = 200000,
    contrastive_weight: float = 1.0,
    hierarchy_weight: float = 1.0,
    path_bce_weight: float = 0.2,
    contrastive_negative_count: int = 256,
    temperature: float = 0.07,
    device: str | None = None,
    seed: int = 13,
    swanlab_project: str | None = "SkillRQ-M4",
    swanlab_run_name: str | None = None,
) -> Mapping[str, Any]:
    torch = require_torch()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = list(read_jsonl(data_root / "candidates.jsonl"))
    queries = list(read_jsonl(data_root / "queries.jsonl"))
    path_catalog = _build_path_catalog(candidates, queries)
    query_examples = _build_query_examples(queries, path_catalog)
    if not query_examples:
        raise ValueError(f"No query examples with gold code paths found in {data_root / 'queries.jsonl'}")

    vocab = _build_vocab(query_examples, max_vocab_size)
    code_vocabs = _build_code_vocabs(path_catalog)
    path_id_to_index = {row["semantic_id"]: index for index, row in enumerate(path_catalog)}
    path_id_to_codes = {row["semantic_id"]: list(row["codes"]) for row in path_catalog}
    path_level_ids = _path_level_id_tensors(path_catalog, code_vocabs, torch, device=None)
    train_rows = [row for row in query_examples if row.get("split") == "train"] or query_examples
    dev_rows = [row for row in query_examples if is_eval_split(row.get("split"))][: max(batch_size * 4, 1)]

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_soft_multipath_code_model(
        vocab_size=len(vocab),
        code_vocab_sizes={level: len(code_vocabs[level]) for level in LEVELS},
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        code_embedding_dim=code_embedding_dim,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    path_level_ids = {level: tensor.to(resolved_device) for level, tensor in path_level_ids.items()}

    logger = SwanLabLogger(
        project=swanlab_project,
        run_name=swanlab_run_name,
        config={
            "method": MODEL_KIND,
            "data_root": str(data_root),
            "output_root": str(output_root),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "code_embedding_dim": code_embedding_dim,
            "contrastive_weight": contrastive_weight,
            "hierarchy_weight": hierarchy_weight,
            "path_bce_weight": path_bce_weight,
            "contrastive_negative_count": contrastive_negative_count,
            "temperature": temperature,
            "device": str(resolved_device),
            "train_queries": len(train_rows),
            "all_queries": len(query_examples),
            "path_count": len(path_catalog),
            "vocab_size": len(vocab),
        },
        tags=["m4", "soft-multipath", "query-to-code"],
    )

    history = []
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            rng.shuffle(train_rows)
            totals = Counter()
            for batch in _batches(train_rows, batch_size):
                tensors = _tensorize_query_batch(
                    batch,
                    vocab,
                    code_vocabs,
                    path_id_to_index,
                    path_id_to_codes,
                    resolved_device,
                    torch,
                )
                query_hidden = model.encode_query(tensors["token_ids"], tensors["offsets"])
                hierarchy_loss = _hierarchy_loss(model, query_hidden, tensors, code_vocabs, torch)
                contrastive_loss, path_bce_loss = _path_alignment_losses(
                    model=model,
                    query_hidden=query_hidden,
                    gold_path_indices=tensors["gold_path_indices"],
                    all_path_level_ids=path_level_ids,
                    total_paths=len(path_catalog),
                    contrastive_negative_count=contrastive_negative_count,
                    temperature=temperature,
                    torch=torch,
                    rng=rng,
                )
                loss = (
                    hierarchy_weight * hierarchy_loss
                    + contrastive_weight * contrastive_loss
                    + path_bce_weight * path_bce_loss
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                size = len(batch)
                totals["loss"] += float(loss.detach().cpu()) * size
                totals["hierarchy_loss"] += float(hierarchy_loss.detach().cpu()) * size
                totals["contrastive_loss"] += float(contrastive_loss.detach().cpu()) * size
                totals["path_bce_loss"] += float(path_bce_loss.detach().cpu()) * size
                totals["examples"] += size

            metrics = {
                "epoch": epoch,
                "train_loss": totals["loss"] / max(totals["examples"], 1),
                "train_hierarchy_loss": totals["hierarchy_loss"] / max(totals["examples"], 1),
                "train_contrastive_loss": totals["contrastive_loss"] / max(totals["examples"], 1),
                "train_path_bce_loss": totals["path_bce_loss"] / max(totals["examples"], 1),
                "train_examples": int(totals["examples"]),
            }
            if dev_rows:
                metrics.update(
                    _evaluate_soft_model(
                        model=model,
                        rows=dev_rows,
                        vocab=vocab,
                        code_vocabs=code_vocabs,
                        path_catalog=path_catalog,
                        path_id_to_index=path_id_to_index,
                        path_level_ids=path_level_ids,
                        device=resolved_device,
                        torch=torch,
                        top_n_paths=16,
                        temperature=temperature,
                    )
                )
            history.append(metrics)
            logger.log(_swanlab_epoch_payload(metrics), step=epoch)
    finally:
        logger.finish()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_kind": MODEL_KIND,
                "vocab_size": len(vocab),
                "code_vocab_sizes": {level: len(code_vocabs[level]) for level in LEVELS},
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "code_embedding_dim": code_embedding_dim,
                "levels": LEVELS,
                "temperature": temperature,
            },
        },
        output_root / "model.pt",
    )
    (output_root / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "code_vocabs.json").write_text(json.dumps(code_vocabs, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(output_root / "path_catalog.jsonl", path_catalog)
    summary = {
        "method": MODEL_KIND,
        "data_root": str(data_root),
        "output_root": str(output_root),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(resolved_device),
        "train_queries": len(train_rows),
        "all_queries": len(query_examples),
        "path_count": len(path_catalog),
        "history": history,
        "swanlab_project": swanlab_project,
        "swanlab_run_name": swanlab_run_name,
    }
    write_json(output_root / "train_summary.json", summary)
    return summary


def predict_soft_multipath_codes(
    data_root: Path,
    checkpoint_root: Path,
    output_root: Path,
    top_n_paths: int = 16,
    candidate_budget: int = 100,
    split: str | None = None,
    beam_width: int = 8,
    score_blend: float = 0.65,
    device: str | None = None,
    swanlab_project: str | None = "SkillRQ-M4",
    swanlab_run_name: str | None = None,
) -> Mapping[str, Any]:
    torch = require_torch()
    output_root.mkdir(parents=True, exist_ok=True)
    model, vocab, code_vocabs, path_catalog, temperature = _load_soft_model(checkpoint_root, device, torch)
    reverse_code_vocabs = _reverse_code_vocabs(code_vocabs)
    path_id_to_row = {row["semantic_id"]: row for row in path_catalog}
    path_level_ids = _path_level_id_tensors(path_catalog, code_vocabs, torch, device=next(model.parameters()).device)
    candidates = list(read_jsonl(data_root / "candidates.jsonl"))
    queries = [
        row for row in read_jsonl(data_root / "queries.jsonl")
        if split is None or row.get("split") == split
    ]

    logger = SwanLabLogger(
        project=swanlab_project,
        run_name=swanlab_run_name,
        config={
            "method": f"{MODEL_KIND}_predict",
            "data_root": str(data_root),
            "checkpoint_root": str(checkpoint_root),
            "output_root": str(output_root),
            "top_n_paths": top_n_paths,
            "candidate_budget": candidate_budget,
            "split": split,
            "beam_width": beam_width,
            "score_blend": score_blend,
            "device": str(next(model.parameters()).device),
            "candidate_count": len(candidates),
            "query_count": len(queries),
            "path_count": len(path_catalog),
        },
        tags=["m4", "soft-multipath", "predict"],
    )

    rows = []
    for query in queries:
        predicted_paths = _predict_soft_paths(
            model=model,
            query=str(query.get("query") or ""),
            vocab=vocab,
            code_vocabs=code_vocabs,
            reverse_code_vocabs=reverse_code_vocabs,
            path_catalog=path_catalog,
            path_id_to_row=path_id_to_row,
            path_level_ids=path_level_ids,
            top_n_paths=top_n_paths,
            beam_width=beam_width,
            score_blend=score_blend,
            temperature=temperature,
            device=next(model.parameters()).device,
            torch=torch,
        )
        retrieved = _retrieve_candidates(predicted_paths, candidates, candidate_budget)
        rows.append(
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

    write_jsonl(output_root / "predictions.jsonl", rows)
    summary = {
        "method": f"{MODEL_KIND}_predict",
        "data_root": str(data_root),
        "checkpoint_root": str(checkpoint_root),
        "output_root": str(output_root),
        "queries": len(rows),
        "top_n_paths": top_n_paths,
        "candidate_budget": candidate_budget,
        "split": split,
        "beam_width": beam_width,
        "score_blend": score_blend,
        "avg_candidate_pool_size": _avg_candidate_pool_size(rows),
        "swanlab_project": swanlab_project,
        "swanlab_run_name": swanlab_run_name,
    }
    write_json(output_root / "predict_summary.json", summary)
    logger.log(_swanlab_prediction_payload(summary, rows), step=1)
    logger.finish()
    return summary


def verbalize_code_path(row: Mapping[str, Any]) -> str:
    labels = row.get("labels") or {}
    domain = labels.get("l1") or _clean_code(row.get("codes", ["UNKNOWN"])[0])
    operation = labels.get("l2") or _clean_code(row.get("codes", ["", "UNKNOWN"])[1])
    role = labels.get("l3") or row.get("role_hint") or "UNASSIGNED"
    io_constraint = labels.get("l4") or _clean_code(row.get("codes", ["", "", "", "UNKNOWN"])[3])
    return (
        f"Domain: {domain}. Operation: {operation}. Role: {role}. "
        f"IO Constraint: {io_constraint}. "
        "This code path represents capabilities with this domain, operation, execution role, "
        "and input/output constraint profile for agent planning."
    )


def _build_path_catalog(
    candidates: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    by_semantic_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        codes = [str(item) for item in candidate.get("code_path") or []]
        if len(codes) != 4:
            continue
        semantic_id = str(candidate.get("semantic_id") or "/".join(codes))
        labels = dict(candidate.get("labels") or {})
        row = by_semantic_id.setdefault(
            semantic_id,
            {
                "semantic_id": semantic_id,
                "codes": codes,
                "labels": labels,
                "role_hint": candidate.get("role_hint"),
                "code_explanation": candidate.get("code_explanation"),
                "example_candidate_ids": [],
                "example_names": [],
            },
        )
        if len(row["example_candidate_ids"]) < 5:
            row["example_candidate_ids"].append(candidate.get("candidate_id"))
            row["example_names"].append(candidate.get("name"))
    for query in queries:
        for path in query.get("gold_code_paths") or []:
            codes = [str(item) for item in path.get("codes") or []]
            if len(codes) != 4:
                continue
            semantic_id = str(path.get("semantic_id") or "/".join(codes))
            by_semantic_id.setdefault(
                semantic_id,
                {
                    "semantic_id": semantic_id,
                    "codes": codes,
                    "labels": {},
                    "role_hint": path.get("role_hint"),
                    "code_explanation": None,
                    "example_candidate_ids": [path.get("candidate_id")],
                    "example_names": [],
                },
            )
    rows = sorted(by_semantic_id.values(), key=lambda row: row["semantic_id"])
    for row in rows:
        row["verbalization"] = verbalize_code_path(row)
    return rows


def _build_query_examples(
    queries: Sequence[Mapping[str, Any]],
    path_catalog: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    known_paths = {str(row["semantic_id"]) for row in path_catalog}
    rows = []
    for query in queries:
        gold_paths = []
        for path in query.get("gold_code_paths") or []:
            semantic_id = str(path.get("semantic_id") or "/".join(str(item) for item in path.get("codes") or []))
            if semantic_id in known_paths:
                gold_paths.append(semantic_id)
        gold_paths = list(dict.fromkeys(gold_paths))
        if not gold_paths:
            continue
        rows.append(
            {
                "query_id": query["query_id"],
                "query": str(query.get("query") or ""),
                "split": query.get("split"),
                "source_dataset": query.get("source_dataset"),
                "gold_path_ids": gold_paths,
                "gold_ids": query.get("gold_ids") or [],
            }
        )
    return rows


def _build_vocab(rows: Sequence[Mapping[str, Any]], max_vocab_size: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(_tokens(str(row.get("query") or "")))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _count in counts.most_common(max_vocab_size - len(vocab)):
        vocab[token] = len(vocab)
    return vocab


def _build_code_vocabs(path_catalog: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    values = {level: [] for level in LEVELS}
    for row in path_catalog:
        codes = list(row["codes"])
        for index, level in enumerate(LEVELS):
            values[level].append(str(codes[index]))
    return {level: {code: index for index, code in enumerate(sorted(set(codes)))} for level, codes in values.items()}


def _path_level_id_tensors(path_catalog, code_vocabs, torch, device):
    tensors = {}
    for index, level in enumerate(LEVELS):
        tensors[level] = torch.tensor(
            [code_vocabs[level][str(row["codes"][index])] for row in path_catalog],
            dtype=torch.long,
            device=device,
        )
    return tensors


def _tensorize_query_batch(batch, vocab, code_vocabs, path_id_to_index, path_id_to_codes, device, torch):
    flat_tokens = []
    offsets = []
    flat_query_indices = []
    labels = {level: [] for level in LEVELS}
    l1_targets = []
    gold_path_indices = []
    for query_index, row in enumerate(batch):
        offsets.append(len(flat_tokens))
        token_ids = [vocab.get(token, vocab["<unk>"]) for token in _tokens(str(row.get("query") or ""))]
        flat_tokens.extend(token_ids or [vocab["<unk>"]])
        query_gold_path_indices = []
        l1_target = [0.0] * len(code_vocabs["l1"])
        for semantic_id in row["gold_path_ids"]:
            path_index = path_id_to_index[str(semantic_id)]
            query_gold_path_indices.append(path_index)
            codes = path_id_to_codes[str(semantic_id)]
            if len(codes) != 4:
                continue
            flat_query_indices.append(query_index)
            for level_index, level in enumerate(LEVELS):
                label = code_vocabs[level][codes[level_index]]
                labels[level].append(label)
                if level == "l1":
                    l1_target[label] = 1.0
        gold_path_indices.append(sorted(set(query_gold_path_indices)))
        l1_targets.append(l1_target)
    return {
        "token_ids": torch.tensor(flat_tokens, dtype=torch.long, device=device),
        "offsets": torch.tensor(offsets, dtype=torch.long, device=device),
        "flat_query_indices": torch.tensor(flat_query_indices, dtype=torch.long, device=device),
        "labels": {level: torch.tensor(values, dtype=torch.long, device=device) for level, values in labels.items()},
        "l1_targets": torch.tensor(l1_targets, dtype=torch.float32, device=device),
        "gold_path_indices": gold_path_indices,
    }


def _hierarchy_loss(model, query_hidden, tensors, code_vocabs, torch):
    l1_logits = model.level_logits(query_hidden)["l1"]
    l1_loss = torch.nn.functional.binary_cross_entropy_with_logits(l1_logits, tensors["l1_targets"])
    flat_indices = tensors["flat_query_indices"]
    if flat_indices.numel() == 0:
        return l1_loss
    flat_query_hidden = query_hidden.index_select(0, flat_indices)
    labels = tensors["labels"]
    l2_logits = model.level_logits(flat_query_hidden, {"l1": labels["l1"]})["l2"]
    l3_logits = model.level_logits(flat_query_hidden, {"l1": labels["l1"], "l2": labels["l2"]})["l3"]
    l4_logits = model.level_logits(
        flat_query_hidden,
        {"l1": labels["l1"], "l2": labels["l2"], "l3": labels["l3"]},
    )["l4"]
    return (
        l1_loss
        + torch.nn.functional.cross_entropy(l2_logits, labels["l2"])
        + torch.nn.functional.cross_entropy(l3_logits, labels["l3"])
        + torch.nn.functional.cross_entropy(l4_logits, labels["l4"])
    )


def _path_alignment_losses(
    *,
    model,
    query_hidden,
    gold_path_indices,
    all_path_level_ids,
    total_paths: int,
    contrastive_negative_count: int,
    temperature: float,
    torch,
    rng: random.Random,
):
    positive_indices = sorted({index for values in gold_path_indices for index in values})
    sampled = set(positive_indices)
    negative_budget = max(0, int(contrastive_negative_count))
    while len(sampled) < min(total_paths, len(positive_indices) + negative_budget):
        sampled.add(rng.randrange(total_paths))
    sampled_indices = sorted(sampled)
    sampled_tensor = torch.tensor(sampled_indices, dtype=torch.long, device=query_hidden.device)
    sampled_level_ids = {level: ids.index_select(0, sampled_tensor) for level, ids in all_path_level_ids.items()}
    logits = model.path_logits(query_hidden, sampled_level_ids, temperature=temperature)
    index_position = {index: position for position, index in enumerate(sampled_indices)}
    target = torch.zeros_like(logits)
    losses = []
    for row_index, positives in enumerate(gold_path_indices):
        positions = [index_position[index] for index in positives if index in index_position]
        if not positions:
            continue
        target[row_index, positions] = 1.0
        pos_logits = logits[row_index, positions]
        losses.append(-(torch.logsumexp(pos_logits, dim=0) - torch.logsumexp(logits[row_index], dim=0)))
    if losses:
        contrastive_loss = torch.stack(losses).mean()
    else:
        contrastive_loss = logits.sum() * 0.0
    path_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    return contrastive_loss, path_bce_loss


def _evaluate_soft_model(
    *,
    model,
    rows,
    vocab,
    code_vocabs,
    path_catalog,
    path_id_to_index,
    path_level_ids,
    device,
    torch,
    top_n_paths: int,
    temperature: float,
):
    model.eval()
    recall_sum = 0.0
    top1_sum = 0.0
    with torch.no_grad():
        for row in rows:
            predicted = _predict_soft_paths(
                model=model,
                query=str(row.get("query") or ""),
                vocab=vocab,
                code_vocabs=code_vocabs,
                reverse_code_vocabs=_reverse_code_vocabs(code_vocabs),
                path_catalog=path_catalog,
                path_id_to_row={item["semantic_id"]: item for item in path_catalog},
                path_level_ids=path_level_ids,
                top_n_paths=top_n_paths,
                beam_width=8,
                score_blend=0.65,
                temperature=temperature,
                device=device,
                torch=torch,
            )
            gold = set(row["gold_path_ids"])
            predicted_ids = [str(item["semantic_id"]) for item in predicted]
            recall_sum += len(set(predicted_ids) & gold) / max(len(gold), 1)
            top1_sum += float(bool(predicted_ids and predicted_ids[0] in gold))
    total = max(len(rows), 1)
    return {
        f"dev_path_recall@{top_n_paths}": recall_sum / total,
        "dev_path_top1_accuracy": top1_sum / total,
    }


def _predict_soft_paths(
    *,
    model,
    query: str,
    vocab,
    code_vocabs,
    reverse_code_vocabs,
    path_catalog,
    path_id_to_row,
    path_level_ids,
    top_n_paths: int,
    beam_width: int,
    score_blend: float,
    temperature: float,
    device,
    torch,
):
    token_ids = [vocab.get(token, vocab.get("<unk>", 1)) for token in _tokens(query)] or [vocab.get("<unk>", 1)]
    with torch.no_grad():
        query_hidden = model.encode_query(
            torch.tensor(token_ids, dtype=torch.long, device=device),
            torch.tensor([0], dtype=torch.long, device=device),
        )
        contrastive_logits = model.path_logits(query_hidden, path_level_ids, temperature=temperature)[0]
        contrastive_probs = torch.softmax(contrastive_logits, dim=-1)
        keep = min(max(top_n_paths * 4, beam_width), contrastive_probs.numel())
        values, indices = torch.topk(contrastive_probs, k=keep)
        candidates = {
            str(path_catalog[int(index.detach().cpu())]["semantic_id"]): {
                "contrastive_probability": float(value.detach().cpu()),
                "hierarchy_probability": 0.0,
            }
            for value, index in zip(values, indices)
        }
        for codes, probability in _hierarchical_beam(model, query_hidden, reverse_code_vocabs, beam_width, torch):
            semantic_id = "/".join(codes)
            if semantic_id in path_id_to_row:
                entry = candidates.setdefault(
                    semantic_id,
                    {"contrastive_probability": 0.0, "hierarchy_probability": 0.0},
                )
                entry["hierarchy_probability"] = max(entry["hierarchy_probability"], probability)

    blend = min(max(float(score_blend), 0.0), 1.0)
    scored = []
    for semantic_id, scores in candidates.items():
        row = path_id_to_row[semantic_id]
        score = blend * scores["contrastive_probability"] + (1.0 - blend) * scores["hierarchy_probability"]
        scored.append((score, semantic_id, row, scores))
    scored.sort(key=lambda item: item[0], reverse=True)
    normalizer = sum(max(score, 0.0) for score, *_ in scored[:top_n_paths]) or 1.0
    output = []
    for index, (score, semantic_id, row, scores) in enumerate(scored[:top_n_paths], start=1):
        probability = max(score, 0.0) / normalizer
        output.append(
            {
                "path_id": f"P{index}",
                "semantic_id": semantic_id,
                "codes": list(row["codes"]),
                "probability": probability,
                "score": score,
                "hierarchy_probability": scores["hierarchy_probability"],
                "contrastive_probability": scores["contrastive_probability"],
                "role_hint": row.get("role_hint"),
                "reason": _reason(row),
                "verbalization": row.get("verbalization") or verbalize_code_path(row),
                "code_explanation": row.get("code_explanation"),
            }
        )
    return output


def _hierarchical_beam(model, query_hidden, reverse_code_vocabs, beam_width: int, torch):
    l1_probs = torch.softmax(model.level_logits(query_hidden)["l1"][0], dim=-1)
    values, indices = torch.topk(l1_probs, k=min(beam_width, l1_probs.numel()))
    beams = [
        ([reverse_code_vocabs["l1"][int(index.detach().cpu())]], float(value.detach().cpu()), int(index.detach().cpu()))
        for value, index in zip(values, indices)
    ]
    for level in ("l2", "l3", "l4"):
        next_beams = []
        for codes, score, _last_index in beams:
            prefix_ids = {
                prev_level: torch.tensor(
                    [reverse_code_vocabs[f"{prev_level}_to_id"][codes[prev_index]]],
                    dtype=torch.long,
                    device=query_hidden.device,
                )
                for prev_index, prev_level in enumerate(LEVELS[: LEVELS.index(level)])
            }
            logits = model.level_logits(query_hidden, prefix_ids)[level][0]
            probs = torch.softmax(logits, dim=-1)
            values, indices = torch.topk(probs, k=min(beam_width, probs.numel()))
            for value, index in zip(values, indices):
                code = reverse_code_vocabs[level][int(index.detach().cpu())]
                next_beams.append(([*codes, code], score * float(value.detach().cpu()), int(index.detach().cpu())))
        beams = sorted(next_beams, key=lambda item: item[1], reverse=True)[:beam_width]
    return [(codes, score) for codes, score, _index in beams]


def _load_soft_model(checkpoint_root: Path, device: str | None, torch):
    vocab = json.loads((checkpoint_root / "vocab.json").read_text(encoding="utf-8"))
    code_vocabs = json.loads((checkpoint_root / "code_vocabs.json").read_text(encoding="utf-8"))
    path_catalog = list(read_jsonl(checkpoint_root / "path_catalog.jsonl"))
    checkpoint = torch.load(checkpoint_root / "model.pt", map_location=device or "cpu")
    config = checkpoint["config"]
    model = build_soft_multipath_code_model(
        vocab_size=config["vocab_size"],
        code_vocab_sizes=config["code_vocab_sizes"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        code_embedding_dim=config.get("code_embedding_dim", 128),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(resolved_device)
    model.eval()
    return model, vocab, code_vocabs, path_catalog, float(config.get("temperature", 0.07))


def _retrieve_candidates(
    predicted_paths: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    candidate_budget: int,
) -> list[Mapping[str, Any]]:
    by_path = {str(path.get("semantic_id")): path for path in predicted_paths}
    rows = []
    for candidate in candidates:
        semantic_id = str(candidate.get("semantic_id") or "")
        path = by_path.get(semantic_id)
        if path is None:
            continue
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "name": candidate.get("name"),
                "source_dataset": candidate.get("source_dataset"),
                "matched_code_path": path,
                "code_match_score": float(path.get("probability") or path.get("score") or 0.0),
                "matched_levels": 4,
                "code_explanation": candidate.get("code_explanation") or path.get("code_explanation"),
                "capability_text_evidence": str(candidate.get("text") or "")[:600],
            }
        )
    rows.sort(key=lambda row: row["code_match_score"], reverse=True)
    return rows[:candidate_budget]


def _reverse_code_vocabs(code_vocabs):
    reverse = {
        level: {index: code for code, index in values.items()}
        for level, values in code_vocabs.items()
    }
    for level, values in code_vocabs.items():
        reverse[f"{level}_to_id"] = values
    return reverse


def _swanlab_epoch_payload(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = {
        "train/loss": metrics.get("train_loss"),
        "train/hierarchy_loss": metrics.get("train_hierarchy_loss"),
        "train/contrastive_loss": metrics.get("train_contrastive_loss"),
        "train/path_bce_loss": metrics.get("train_path_bce_loss"),
        "train/examples": metrics.get("train_examples"),
    }
    for key, value in metrics.items():
        if key.startswith("dev_"):
            payload[f"dev/{key.removeprefix('dev_')}"] = value
    return payload


def _swanlab_prediction_payload(summary, rows):
    payload: dict[str, Any] = {
        "predict/queries": summary.get("queries"),
        "predict/top_n_paths": summary.get("top_n_paths"),
        "predict/candidate_budget": summary.get("candidate_budget"),
        "predict/avg_candidate_pool_size": summary.get("avg_candidate_pool_size"),
        "predict/split": summary.get("split") or "all",
    }
    for index, row in enumerate(rows[:3], start=1):
        paths = row.get("predicted_code_paths") or []
        retrieved = row.get("retrieved_capabilities") or []
        payload[f"examples/{index}/query_id"] = row.get("query_id")
        payload[f"examples/{index}/query"] = str(row.get("query") or "")[:500]
        payload[f"examples/{index}/top_code_path"] = str(paths[0].get("codes") if paths else "")
        payload[f"examples/{index}/top_path_reason"] = str(paths[0].get("reason") if paths else "")[:300]
        payload[f"examples/{index}/top_candidate"] = str(retrieved[0].get("candidate_id") if retrieved else "")
    return payload


def _avg_candidate_pool_size(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(len(row.get("retrieved_capabilities") or []) for row in rows) / len(rows)


def _reason(row: Mapping[str, Any]) -> str:
    labels = row.get("labels") or {}
    operation = labels.get("l2") or _clean_code((row.get("codes") or ["", "UNKNOWN"])[1])
    role = labels.get("l3") or row.get("role_hint") or "UNASSIGNED"
    return f"query is aligned with operation={operation} and role={role}"


def _clean_code(value: Any) -> str:
    text = str(value or "UNKNOWN")
    if "-" in text:
        text = text.rsplit("-", 1)[0]
    return text.replace("_", " ")


def _batches(rows: Sequence[Mapping[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token]
