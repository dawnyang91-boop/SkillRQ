"""Residual code path planning for M5."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model import build_code_path_planner_model
from ..m4.swanlab_utils import SwanLabLogger
from ..m4.torch_utils import require_torch
from ..splits import is_eval_split
from ..utils.io import read_jsonl, write_json, write_jsonl


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
LEVELS = ("l1", "l2", "l3", "l4")
ROLES = ("START", "SUPPORT", "CHECK", "FINALIZE", "AVOID", "UNASSIGNED", "STOP")
MODEL_KIND = "code-plan"
GENERIC_PATH_MARKERS = ("get_all", "search_for", "method_unknown_schema_light", "toolbench_answer_tree")


def prepare_code_path_planning_data(
    m4_data_root: Path,
    output_root: Path,
    m4_prediction_path: Path | None = None,
    max_steps: int = 6,
    limit_queries: int | None = None,
) -> Mapping[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = {str(row["candidate_id"]): row for row in read_jsonl(m4_data_root / "candidates.jsonl")}
    predictions = _load_m4_predictions(m4_prediction_path)
    examples = []
    query_plans = []
    for query in read_jsonl(m4_data_root / "queries.jsonl"):
        m4_item = predictions.get(str(query["query_id"])) or {}
        predicted_paths = list(m4_item.get("predicted_code_paths") or []) or _oracle_predicted_paths(query)
        plan = _build_code_path_plan(query, candidates, predicted_paths, max_steps=max_steps)
        if not plan:
            continue
        query_plans.append(
            {
                "query_id": query["query_id"],
                "query": query.get("query"),
                "source_dataset": query.get("source_dataset"),
                "split": query.get("split"),
                "gold_ids": query.get("gold_ids") or [],
                "predicted_code_paths": predicted_paths,
                "code_plan": plan,
            }
        )
        examples.extend(_plan_to_training_examples(query, predicted_paths, plan))
        if limit_queries is not None and len(query_plans) >= limit_queries:
            break

    stats = {
        "method": MODEL_KIND,
        "m4_data_root": str(m4_data_root),
        "m4_prediction_path": str(m4_prediction_path) if m4_prediction_path else None,
        "output_root": str(output_root),
        "queries": len(query_plans),
        "planning_examples": len(examples),
        "max_steps": max_steps,
        "avg_steps_per_query": (sum(len(row["code_plan"]) for row in query_plans) / len(query_plans)) if query_plans else 0.0,
        "stop_examples": sum(1 for row in examples if row.get("stop_label") == 1),
    }
    write_jsonl(output_root / "code_plan_examples.jsonl", examples)
    write_jsonl(output_root / "query_code_plans.jsonl", query_plans)
    write_json(output_root / "stats.json", stats)
    return stats


def train_code_path_planner(
    data_root: Path,
    output_root: Path,
    epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    embedding_dim: int = 512,
    hidden_dim: int = 1024,
    coverage_weight: float = 1.0,
    role_weight: float = 0.3,
    stop_weight: float = 0.3,
    max_vocab_size: int = 200000,
    device: str | None = None,
    swanlab_project: str | None = "SkillRQ-M5",
    swanlab_run_name: str | None = None,
) -> Mapping[str, Any]:
    torch = require_torch()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(data_root / "code_plan_examples.jsonl"))
    if not rows:
        raise ValueError(f"No code plan examples found at {data_root / 'code_plan_examples.jsonl'}")
    vocab = _build_vocab(rows, max_vocab_size)
    code_vocabs = _build_code_vocabs(row for row in rows if not row.get("stop_label"))
    role_vocab = {role: index for index, role in enumerate(ROLES)}
    train_rows = [row for row in rows if row.get("split") == "train"] or rows
    dev_rows = [row for row in rows if is_eval_split(row.get("split"))][: max(batch_size * 4, 1)]
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_code_path_planner_model(
        vocab_size=len(vocab),
        code_vocab_sizes={level: len(code_vocabs[level]) for level in LEVELS},
        role_count=len(role_vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    logger = SwanLabLogger(
        project=swanlab_project,
        run_name=swanlab_run_name,
        config={
            "method": "residual_code_path_planner",
            "data_root": str(data_root),
            "output_root": str(output_root),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "coverage_weight": coverage_weight,
            "role_weight": role_weight,
            "stop_weight": stop_weight,
            "device": str(resolved_device),
            "train_examples": len(train_rows),
        },
        tags=["m5", "code-plan", "residual-planner"],
    )

    history = []
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            totals = Counter()
            total = 0
            for batch in _batches(train_rows, batch_size):
                tensors = _tensorize(batch, vocab, code_vocabs, role_vocab, resolved_device, torch)
                outputs = model(tensors["token_ids"], tensors["offsets"])
                losses = _compute_losses(outputs, tensors, torch, coverage_weight, role_weight, stop_weight)
                optimizer.zero_grad()
                losses["loss"].backward()
                optimizer.step()
                size = len(batch)
                for key, value in losses.items():
                    totals[key] += float(value.detach().cpu()) * size
                total += size
            metrics = {f"train_{key}": totals[key] / max(total, 1) for key in totals}
            metrics["epoch"] = epoch
            metrics["train_examples"] = total
            if dev_rows:
                metrics.update(_evaluate(model, dev_rows, vocab, code_vocabs, role_vocab, resolved_device, torch, batch_size, coverage_weight, role_weight, stop_weight))
            history.append(metrics)
            logger.log(_swanlab_payload(metrics), step=epoch)
    finally:
        logger.finish()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_kind": MODEL_KIND,
                "vocab_size": len(vocab),
                "code_vocab_sizes": {level: len(code_vocabs[level]) for level in LEVELS},
                "role_count": len(role_vocab),
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "levels": LEVELS,
            },
        },
        output_root / "model.pt",
    )
    (output_root / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "code_vocabs.json").write_text(json.dumps(code_vocabs, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "role_vocab.json").write_text(json.dumps(role_vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "method": "residual_code_path_planner",
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


def predict_code_path_plan(
    m4_data_root: Path,
    checkpoint_root: Path,
    output_root: Path,
    m4_prediction_path: Path | None = None,
    max_steps: int = 6,
    top_n_paths: int = 16,
    candidates_per_step: int = 20,
    stop_threshold: float = 0.55,
    split: str | None = None,
    device: str | None = None,
    swanlab_project: str | None = "SkillRQ-M5",
    swanlab_run_name: str | None = None,
    enable_exact_first_retrieval: bool = False,
    use_m4_candidate_prior: bool = True,
) -> Mapping[str, Any]:
    torch = require_torch()
    output_root.mkdir(parents=True, exist_ok=True)
    model, vocab, code_vocabs, role_vocab = _load_model(checkpoint_root, device, torch)
    reverse_code_vocabs = {level: {idx: code for code, idx in values.items()} for level, values in code_vocabs.items()}
    reverse_role_vocab = {idx: role for role, idx in role_vocab.items()}
    candidates = list(read_jsonl(m4_data_root / "candidates.jsonl"))
    candidate_index = _build_candidate_retrieval_index(candidates)
    candidate_by_id = candidate_index["candidate_by_id"]
    predictions = _load_m4_predictions(m4_prediction_path)
    queries = [row for row in read_jsonl(m4_data_root / "queries.jsonl") if split is None or row.get("split") == split]
    logger = SwanLabLogger(
        project=swanlab_project,
        run_name=swanlab_run_name,
        config={
            "method": "m5_code_path_plan_predict",
            "m4_data_root": str(m4_data_root),
            "checkpoint_root": str(checkpoint_root),
            "m4_prediction_path": str(m4_prediction_path) if m4_prediction_path else None,
            "max_steps": max_steps,
            "top_n_paths": top_n_paths,
            "candidates_per_step": candidates_per_step,
            "stop_threshold": stop_threshold,
            "split": split,
            "query_count": len(queries),
            "candidate_count": len(candidates),
            "candidate_exact_path_buckets": len(candidate_index["exact_path"]),
            "candidate_l123_buckets": len(candidate_index["prefix_l123"]),
            "candidate_l12_buckets": len(candidate_index["prefix_l12"]),
            "enable_exact_first_retrieval": enable_exact_first_retrieval,
            "use_m4_candidate_prior": use_m4_candidate_prior,
        },
        tags=["m5", "code-plan", "predict"],
    )

    rows = []
    m4_candidate_total = 0
    m4_candidate_reused_total = 0
    m4_hit_count = 0
    m5_hit_count = 0
    m4_hit_m5_miss_count = 0
    m4_miss_m5_hit_count = 0
    evaluated_hit_queries = 0
    for query in queries:
        m4_item = predictions.get(str(query["query_id"])) or {}
        predicted_paths = list(m4_item.get("predicted_code_paths") or []) or _oracle_predicted_paths(query)
        m4_retrieved = list(m4_item.get("retrieved_capabilities") or [])
        effective_m4_retrieved = m4_retrieved if use_m4_candidate_prior else []
        m4_candidate_ids = _candidate_id_set(m4_retrieved)
        query_gold_ids = {str(item) for item in query.get("gold_ids") or [] if item is not None}
        selected_paths = []
        used_semantic_ids: set[str] = set()
        used_candidate_ids: set[str] = set()
        covered_roles: set[str] = set()
        covered_operations: set[str] = set()
        covered_schema_constraints: set[str] = set()
        code_plan = []
        residual_code_paths = []
        for step_index in range(max_steps):
            state = _planning_state_text(
                query=query,
                predicted_paths=predicted_paths,
                selected_paths=selected_paths,
                covered_roles=covered_roles,
                covered_operations=covered_operations,
                covered_schema_constraints=covered_schema_constraints,
                step_index=step_index,
            )
            output = _predict_next_step(model, state, vocab, reverse_code_vocabs, reverse_role_vocab, top_n_paths, next(model.parameters()).device, torch)
            if output["stop_probability"] >= stop_threshold and code_plan:
                break
            selected = _select_predicted_path(output["paths"], predicted_paths, used_semantic_ids)
            if selected is None:
                break
            semantic_id = str(selected.get("semantic_id") or "/".join(selected["codes"]))
            used_semantic_ids.add(semantic_id)
            attrs = _path_attrs(selected)
            covered_roles.add(attrs["role"])
            covered_operations.add(attrs["operation"])
            covered_schema_constraints.add(attrs["schema"])
            selected_paths.append(selected)
            retrieved = _retrieve_for_path(
                selected,
                candidate_index,
                candidates_per_step,
                m4_retrieved=effective_m4_retrieved,
                used_candidate_ids=used_candidate_ids,
                exact_first_retrieval=enable_exact_first_retrieval,
            )
            used_candidate_ids.update(str(item["candidate_id"]) for item in retrieved)
            purpose = _purpose(attrs)
            plan_step = {
                "step": step_index + 1,
                "step_index": step_index,
                "semantic_id": semantic_id,
                "code_path": list(selected["codes"]),
                "codes": list(selected["codes"]),
                "role": attrs["role"],
                "purpose": purpose,
                "expected_coverage_gain": output["expected_coverage_gain"],
                "stop_probability": output["stop_probability"],
                "retrieved_capabilities": retrieved,
            }
            code_plan.append(plan_step)
            residual_code_paths.append(
                {
                    "step_index": step_index,
                    "semantic_id": semantic_id,
                    "codes": list(selected["codes"]),
                    "role_hint": attrs["role"],
                    "score": float(selected.get("probability") or selected.get("score") or 0.0),
                    "predicted_coverage_gain": output["expected_coverage_gain"],
                    "retrieved_capabilities": retrieved,
                    "new_candidate_count": len(retrieved),
                    "m4_candidate_reuse_count": sum(1 for item in retrieved if item.get("m4_candidate_prior")),
                    "purpose": purpose,
                }
            )
        m5_candidate_ids = set(used_candidate_ids)
        m4_reused_ids = m4_candidate_ids & m5_candidate_ids
        m4_candidate_total += len(m4_candidate_ids)
        m4_candidate_reused_total += len(m4_reused_ids)
        if query_gold_ids:
            evaluated_hit_queries += 1
            m4_hit = bool(query_gold_ids & m4_candidate_ids)
            m5_hit = bool(query_gold_ids & m5_candidate_ids)
            m4_hit_count += int(m4_hit)
            m5_hit_count += int(m5_hit)
            m4_hit_m5_miss_count += int(m4_hit and not m5_hit)
            m4_miss_m5_hit_count += int((not m4_hit) and m5_hit)
        rows.append(
            {
                "query_id": query["query_id"],
                "query": query.get("query"),
                "source_dataset": query.get("source_dataset"),
                "split": query.get("split"),
                "gold_ids": query.get("gold_ids"),
                "predicted_code_paths": predicted_paths,
                "code_plan": code_plan,
                "residual_code_paths": residual_code_paths,
                "m4_candidate_count": len(m4_candidate_ids),
                "m4_candidate_reuse_count": len(m4_reused_ids),
            }
        )

    write_jsonl(output_root / "predictions.jsonl", rows)
    summary = {
        "method": "m5_code_path_plan_predict",
        "m4_data_root": str(m4_data_root),
        "checkpoint_root": str(checkpoint_root),
        "m4_prediction_path": str(m4_prediction_path) if m4_prediction_path else None,
        "output_root": str(output_root),
        "queries": len(rows),
        "max_steps": max_steps,
        "top_n_paths": top_n_paths,
        "candidates_per_step": candidates_per_step,
        "stop_threshold": stop_threshold,
        "split": split,
        "candidate_count": len(candidates),
        "candidate_exact_path_buckets": len(candidate_index["exact_path"]),
        "candidate_l123_buckets": len(candidate_index["prefix_l123"]),
        "candidate_l12_buckets": len(candidate_index["prefix_l12"]),
        "enable_exact_first_retrieval": enable_exact_first_retrieval,
        "use_m4_candidate_prior": use_m4_candidate_prior,
        "avg_steps": sum(len(row["code_plan"]) for row in rows) / max(len(rows), 1),
        "m4_candidate_reuse_rate": m4_candidate_reused_total / max(m4_candidate_total, 1),
        "m4_hit_rate": m4_hit_count / max(evaluated_hit_queries, 1),
        "m5_hit_rate": m5_hit_count / max(evaluated_hit_queries, 1),
        "m4_hit_m5_miss_count": m4_hit_m5_miss_count,
        "m4_hit_m5_miss_rate": m4_hit_m5_miss_count / max(evaluated_hit_queries, 1),
        "m4_miss_m5_hit_count": m4_miss_m5_hit_count,
        "m4_miss_m5_hit_rate": m4_miss_m5_hit_count / max(evaluated_hit_queries, 1),
    }
    write_json(output_root / "predict_summary.json", summary)
    logger.log(
        {
            "predict/queries": summary["queries"],
            "predict/avg_steps": summary["avg_steps"],
            "predict/stop_threshold": stop_threshold,
            "predict/m4_candidate_reuse_rate": summary["m4_candidate_reuse_rate"],
            "predict/m4_hit_rate": summary["m4_hit_rate"],
            "predict/m5_hit_rate": summary["m5_hit_rate"],
            "predict/m4_hit_m5_miss_rate": summary["m4_hit_m5_miss_rate"],
            "predict/m4_miss_m5_hit_rate": summary["m4_miss_m5_hit_rate"],
            "predict/enable_exact_first_retrieval": float(enable_exact_first_retrieval),
            "predict/use_m4_candidate_prior": float(use_m4_candidate_prior),
        },
        step=1,
    )
    logger.finish()
    return summary


def _load_m4_predictions(path: Path | None) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    if not path:
        return {}
    return {
        str(row["query_id"]): {
            "predicted_code_paths": list(row.get("predicted_code_paths") or []),
            "retrieved_capabilities": list(row.get("retrieved_capabilities") or []),
        }
        for row in read_jsonl(path)
    }


def _oracle_predicted_paths(query: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = []
    for index, path in enumerate(query.get("gold_code_paths") or [], start=1):
        codes = [str(item) for item in path.get("codes") or []]
        if len(codes) != 4:
            continue
        rows.append(
            {
                "path_id": f"G{index}",
                "semantic_id": str(path.get("semantic_id") or "/".join(codes)),
                "codes": codes,
                "probability": 1.0 / max(len(query.get("gold_code_paths") or []), 1),
                "role_hint": path.get("role_hint") or _role(codes),
            }
        )
    return rows


def _build_code_path_plan(query, candidates, predicted_paths, max_steps: int) -> list[Mapping[str, Any]]:
    gold_by_semantic_id: dict[str, dict[str, Any]] = {}
    for path in query.get("gold_code_paths") or []:
        codes = [str(item) for item in path.get("codes") or []]
        if len(codes) != 4:
            continue
        semantic_id = str(path.get("semantic_id") or "/".join(codes))
        target_ids = [str(path.get("candidate_id"))] if path.get("candidate_id") else []
        if path.get("candidate_id") and path.get("candidate_id") in candidates:
            target_ids = [str(path.get("candidate_id"))]
        gold_by_semantic_id[semantic_id] = {
            "semantic_id": semantic_id,
            "codes": codes,
            "target_ids": target_ids,
            "role_hint": path.get("role_hint") or _role(codes),
        }
    if not gold_by_semantic_id:
        return []
    pool = []
    seen = set()
    for path in predicted_paths:
        codes = [str(item) for item in path.get("codes") or []]
        if len(codes) != 4:
            continue
        semantic_id = str(path.get("semantic_id") or "/".join(codes))
        if semantic_id in seen:
            continue
        seen.add(semantic_id)
        pool.append({"semantic_id": semantic_id, "codes": codes, "probability": float(path.get("probability") or path.get("score") or 0.0), "role_hint": path.get("role_hint") or _role(codes)})
    for semantic_id, path in gold_by_semantic_id.items():
        if semantic_id not in seen:
            pool.append({"semantic_id": semantic_id, "codes": path["codes"], "probability": 0.0, "role_hint": path.get("role_hint")})

    remaining_gold = set(gold_by_semantic_id)
    selected: list[Mapping[str, Any]] = []
    covered_roles: set[str] = set()
    covered_operations: set[str] = set()
    covered_schema_constraints: set[str] = set()
    plan = []
    for step_index in range(max_steps):
        best = None
        best_score = -1.0
        best_gain = {}
        for path in pool:
            semantic_id = str(path["semantic_id"])
            if semantic_id not in remaining_gold:
                continue
            attrs = _path_attrs(path)
            gain = {
                "role": int(attrs["role"] not in covered_roles),
                "operation": int(attrs["operation"] not in covered_operations),
                "schema": int(attrs["schema"] not in covered_schema_constraints),
                "gold_path": 1,
            }
            score = gain["gold_path"] + 0.75 * gain["role"] + 0.50 * gain["operation"] + 0.50 * gain["schema"] + 0.10 * float(path.get("probability") or 0.0)
            if score > best_score:
                best = path
                best_score = score
                best_gain = gain
        if best is None:
            break
        semantic_id = str(best["semantic_id"])
        attrs = _path_attrs(best)
        covered_before = _covered_state(covered_roles, covered_operations, covered_schema_constraints)
        covered_roles.add(attrs["role"])
        covered_operations.add(attrs["operation"])
        covered_schema_constraints.add(attrs["schema"])
        remaining_gold.remove(semantic_id)
        selected.append(best)
        target_ids = gold_by_semantic_id.get(semantic_id, {}).get("target_ids", [])
        normalized_gain = (best_gain.get("gold_path", 0) + best_gain.get("role", 0) + best_gain.get("operation", 0) + best_gain.get("schema", 0)) / 4.0
        plan.append(
            {
                "step_index": step_index,
                "semantic_id": semantic_id,
                "code_path": list(best["codes"]),
                "role": attrs["role"],
                "operation": attrs["operation"],
                "schema_constraint": attrs["schema"],
                "purpose": _purpose(attrs),
                "target_ids": target_ids,
                "covered_before": covered_before,
                "covered_after": _covered_state(covered_roles, covered_operations, covered_schema_constraints),
                "remaining_gold_paths_after": sorted(remaining_gold),
                "expected_coverage_gain": normalized_gain,
                "stop_label": 0,
                "selected_paths_before": [str(item["semantic_id"]) for item in selected[:-1]],
                "predicted_path_rank": _predicted_rank(semantic_id, predicted_paths),
            }
        )
        if not remaining_gold:
            break
    return plan


def _plan_to_training_examples(query, predicted_paths, plan):
    examples = []
    selected_paths = []
    covered_roles: set[str] = set()
    covered_operations: set[str] = set()
    covered_schema_constraints: set[str] = set()
    for step in plan:
        state = _planning_state_text(query, predicted_paths, selected_paths, covered_roles, covered_operations, covered_schema_constraints, int(step["step_index"]))
        examples.append(
            {
                "query_id": query["query_id"],
                "query": query.get("query"),
                "split": query.get("split"),
                "source_dataset": query.get("source_dataset"),
                "step_index": step["step_index"],
                "planner_state": state,
                "predicted_code_paths": predicted_paths,
                "selected_code_paths_before": list(selected_paths),
                "covered_roles": sorted(covered_roles),
                "covered_operations": sorted(covered_operations),
                "covered_schema_constraints": sorted(covered_schema_constraints),
                "semantic_id": step["semantic_id"],
                "code_path": step["code_path"],
                "role": step["role"],
                "purpose": step["purpose"],
                "expected_coverage_gain": step["expected_coverage_gain"],
                "stop_label": 0,
            }
        )
        attrs = _path_attrs({"codes": step["code_path"], "role_hint": step["role"]})
        selected_paths.append(step["semantic_id"])
        covered_roles.add(attrs["role"])
        covered_operations.add(attrs["operation"])
        covered_schema_constraints.add(attrs["schema"])
    if plan:
        terminal_state = _planning_state_text(query, predicted_paths, selected_paths, covered_roles, covered_operations, covered_schema_constraints, len(plan))
        examples.append(
            {
                "query_id": query["query_id"],
                "query": query.get("query"),
                "split": query.get("split"),
                "source_dataset": query.get("source_dataset"),
                "step_index": len(plan),
                "planner_state": terminal_state,
                "predicted_code_paths": predicted_paths,
                "selected_code_paths_before": list(selected_paths),
                "covered_roles": sorted(covered_roles),
                "covered_operations": sorted(covered_operations),
                "covered_schema_constraints": sorted(covered_schema_constraints),
                "semantic_id": None,
                "code_path": None,
                "role": "STOP",
                "purpose": "stop because all gold code path requirements are covered",
                "expected_coverage_gain": 0.0,
                "stop_label": 1,
            }
        )
    return examples


def _planning_state_text(query, predicted_paths, selected_paths, covered_roles, covered_operations, covered_schema_constraints, step_index: int) -> str:
    predicted = " ".join(_path_summary(path) for path in predicted_paths[:16])
    selected = " ".join(str(item) for item in selected_paths)
    return " ".join(
        [
            str(query.get("query") or ""),
            f"step {step_index}",
            f"predicted_paths {predicted}",
            f"selected_paths {selected}",
            f"covered_roles {' '.join(sorted(covered_roles))}",
            f"covered_operations {' '.join(sorted(covered_operations))}",
            f"covered_schema_constraints {' '.join(sorted(covered_schema_constraints))}",
        ]
    )


def _path_summary(path) -> str:
    attrs = _path_attrs(path)
    probability = float(path.get("probability") or path.get("score") or 0.0)
    return f"{path.get('semantic_id') or '/'.join(path.get('codes') or [])} role {attrs['role']} operation {attrs['operation']} schema {attrs['schema']} prob {probability:.4f}"


def _compute_losses(outputs, tensors, torch, coverage_weight: float, role_weight: float, stop_weight: float):
    non_stop = tensors["non_stop_mask"]
    if bool(non_stop.any().detach().cpu()):
        code_loss = sum(torch.nn.functional.cross_entropy(outputs["codes"][level][non_stop], tensors["labels"][level][non_stop]) for level in LEVELS)
        coverage_loss = torch.nn.functional.mse_loss(torch.sigmoid(outputs["coverage_gain"][non_stop]), tensors["gains"][non_stop])
    else:
        code_loss = outputs["coverage_gain"].sum() * 0.0
        coverage_loss = outputs["coverage_gain"].sum() * 0.0
    role_loss = torch.nn.functional.cross_entropy(outputs["roles"], tensors["roles"])
    stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["stop"], tensors["stop"])
    return {
        "loss": code_loss + coverage_weight * coverage_loss + role_weight * role_loss + stop_weight * stop_loss,
        "code_loss": code_loss,
        "coverage_loss": coverage_loss,
        "role_loss": role_loss,
        "stop_loss": stop_loss,
    }


def _evaluate(model, rows, vocab, code_vocabs, role_vocab, device, torch, batch_size: int, coverage_weight: float, role_weight: float, stop_weight: float):
    model.eval()
    totals = Counter()
    total = 0
    exact = 0
    role_correct = 0
    stop_correct = 0
    non_stop_total = 0
    with torch.no_grad():
        for batch in _batches(rows, batch_size):
            tensors = _tensorize(batch, vocab, code_vocabs, role_vocab, device, torch)
            outputs = model(tensors["token_ids"], tensors["offsets"])
            losses = _compute_losses(outputs, tensors, torch, coverage_weight, role_weight, stop_weight)
            size = len(batch)
            for key, value in losses.items():
                totals[key] += float(value.detach().cpu()) * size
            total += size
            role_correct += int((outputs["roles"].argmax(dim=-1) == tensors["roles"]).sum().detach().cpu())
            stop_correct += int(((torch.sigmoid(outputs["stop"]) >= 0.5).float() == tensors["stop"]).sum().detach().cpu())
            non_stop = tensors["non_stop_mask"]
            if bool(non_stop.any().detach().cpu()):
                mask = None
                for level in LEVELS:
                    matches = outputs["codes"][level].argmax(dim=-1) == tensors["labels"][level]
                    mask = matches if mask is None else mask & matches
                exact += int((mask & non_stop).sum().detach().cpu())
                non_stop_total += int(non_stop.sum().detach().cpu())
    metrics = {f"dev_{key}": totals[key] / max(total, 1) for key in totals}
    metrics["dev_path_exact_match"] = exact / max(non_stop_total, 1)
    metrics["dev_role_accuracy"] = role_correct / max(total, 1)
    metrics["dev_stop_accuracy"] = stop_correct / max(total, 1)
    return metrics


def _tensorize(batch, vocab, code_vocabs, role_vocab, device, torch):
    flat_tokens = []
    offsets = []
    labels = {level: [] for level in LEVELS}
    roles = []
    gains = []
    stops = []
    non_stop = []
    for row in batch:
        offsets.append(len(flat_tokens))
        text = str(row.get("planner_state") or row.get("query") or "")
        token_ids = [vocab.get(token, vocab["<unk>"]) for token in _tokens(text)] or [vocab["<unk>"]]
        flat_tokens.extend(token_ids)
        is_stop = int(row.get("stop_label") or 0)
        stops.append(float(is_stop))
        non_stop.append(not is_stop)
        path = list(row.get("code_path") or [])
        for index, level in enumerate(LEVELS):
            value = str(path[index]) if len(path) == 4 else next(iter(code_vocabs[level]))
            labels[level].append(code_vocabs[level][value])
        roles.append(role_vocab.get(_normalize_role(row.get("role")), role_vocab["UNASSIGNED"]))
        gains.append(float(row.get("expected_coverage_gain") or 0.0))
    return {
        "token_ids": torch.tensor(flat_tokens, dtype=torch.long, device=device),
        "offsets": torch.tensor(offsets, dtype=torch.long, device=device),
        "labels": {level: torch.tensor(values, dtype=torch.long, device=device) for level, values in labels.items()},
        "roles": torch.tensor(roles, dtype=torch.long, device=device),
        "gains": torch.tensor(gains, dtype=torch.float32, device=device),
        "stop": torch.tensor(stops, dtype=torch.float32, device=device),
        "non_stop_mask": torch.tensor(non_stop, dtype=torch.bool, device=device),
    }


def _predict_next_step(model, state, vocab, reverse_code_vocabs, reverse_role_vocab, top_n_paths, device, torch):
    token_ids = [vocab.get(token, vocab.get("<unk>", 1)) for token in _tokens(state)] or [vocab.get("<unk>", 1)]
    with torch.no_grad():
        outputs = model(torch.tensor(token_ids, dtype=torch.long, device=device), torch.tensor([0], dtype=torch.long, device=device))
        top_by_level = {}
        for level in LEVELS:
            probs = torch.softmax(outputs["codes"][level][0], dim=-1)
            values, indices = torch.topk(probs, k=min(top_n_paths, probs.numel()))
            top_by_level[level] = [(reverse_code_vocabs[level][int(index.detach().cpu())], float(value.detach().cpu())) for value, index in zip(values, indices)]
        role_probs = torch.softmax(outputs["roles"][0], dim=-1)
        role_index = int(role_probs.argmax(dim=-1).detach().cpu())
        gain = float(torch.sigmoid(outputs["coverage_gain"])[0].detach().cpu())
        stop_probability = float(torch.sigmoid(outputs["stop"])[0].detach().cpu())
    beams = [([], 1.0)]
    for level in LEVELS:
        beams = sorted(
            [([*codes, code], score * prob) for codes, score in beams for code, prob in top_by_level[level]],
            key=lambda item: item[1],
            reverse=True,
        )[:top_n_paths]
    return {
        "paths": [{"codes": codes, "semantic_id": "/".join(codes), "score": score * gain, "probability": score, "role_hint": _role(codes)} for codes, score in beams],
        "role": reverse_role_vocab.get(role_index, "UNASSIGNED"),
        "expected_coverage_gain": gain,
        "stop_probability": stop_probability,
    }


def _select_predicted_path(predicted_model_paths, m4_predicted_paths, used_semantic_ids):
    m4_by_id = {str(path.get("semantic_id") or "/".join(path.get("codes") or [])): path for path in m4_predicted_paths}
    for path in predicted_model_paths:
        semantic_id = str(path.get("semantic_id") or "/".join(path.get("codes") or []))
        if semantic_id in used_semantic_ids:
            continue
        selected = dict(m4_by_id.get(semantic_id) or path)
        selected.setdefault("semantic_id", semantic_id)
        selected.setdefault("codes", path.get("codes") or semantic_id.split("/"))
        selected.setdefault("probability", path.get("probability") or path.get("score") or 0.0)
        return selected
    for path in m4_predicted_paths:
        semantic_id = str(path.get("semantic_id") or "/".join(path.get("codes") or []))
        if semantic_id not in used_semantic_ids:
            selected = dict(path)
            selected.setdefault("semantic_id", semantic_id)
            return selected
    return None


def _retrieve_for_path(
    path,
    candidate_index,
    limit: int,
    m4_retrieved: Sequence[Mapping[str, Any]] | None = None,
    used_candidate_ids: set[str] | None = None,
    exact_first_retrieval: bool = False,
):
    rows_by_id: dict[str, dict[str, Any]] = {}
    codes = [str(item) for item in path.get("codes") or []]
    used_candidate_ids = used_candidate_ids or set()
    m4_retrieved = list(m4_retrieved or [])
    m4_prior_by_id = _m4_prior_by_candidate_id(m4_retrieved)
    candidate_by_id = candidate_index["candidate_by_id"]
    exact_bucket_size = _exact_bucket_size(codes, candidate_index)
    for candidate, bucket_source in _iter_indexed_candidates(codes, candidate_index):
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in used_candidate_ids:
            continue
        candidate_codes = [str(item) for item in candidate.get("code_path") or []]
        overlap = sum(1 for left, right in zip(codes, candidate_codes) if left == right)
        m4_prior = m4_prior_by_id.get(candidate_id)
        score = _candidate_score(
            path,
            codes,
            candidate_codes,
            overlap,
            m4_prior,
            exact_bucket_size,
            exact_first_retrieval=exact_first_retrieval,
        )
        source = bucket_source
        if m4_prior is not None:
            source = f"{bucket_source}+m4_prior"
        _upsert_retrieved_row(
            rows_by_id,
            candidate=candidate,
            score=score,
            overlap=overlap,
            m4_prior=m4_prior,
            retrieval_source=source,
        )

    for item in m4_retrieved:
        candidate_id = item.get("candidate_id")
        if candidate_id is None:
            continue
        candidate_id = str(candidate_id)
        if candidate_id in used_candidate_ids:
            continue
        candidate = dict(candidate_by_id.get(candidate_id) or {})
        candidate.setdefault("candidate_id", candidate_id)
        candidate.setdefault("name", item.get("name"))
        candidate.setdefault("source_dataset", item.get("source_dataset"))
        candidate.setdefault("code_explanation", item.get("code_explanation"))
        candidate.setdefault("text", item.get("capability_text_evidence"))
        candidate_codes = [str(value) for value in candidate.get("code_path") or _matched_code_path_codes(item)]
        overlap = sum(1 for left, right in zip(codes, candidate_codes) if left == right)
        m4_prior = m4_prior_by_id.get(candidate_id)
        score = _candidate_score(
            path,
            codes,
            candidate_codes,
            overlap,
            m4_prior,
            exact_bucket_size,
            exact_first_retrieval=exact_first_retrieval,
        )
        _upsert_retrieved_row(
            rows_by_id,
            candidate=candidate,
            score=score,
            overlap=overlap,
            m4_prior=m4_prior,
            retrieval_source="m4_prior",
        )

    rows = list(rows_by_id.values())
    rows.sort(key=lambda row: _retrieval_sort_key(row, exact_first_retrieval), reverse=True)
    return rows[:limit]


def _build_candidate_retrieval_index(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    exact_path: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    prefix_l123: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    prefix_l12: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    candidate_by_id: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        if candidate_id is not None:
            candidate_by_id[str(candidate_id)] = candidate
        codes = tuple(str(item) for item in candidate.get("code_path") or [])
        if len(codes) >= 4:
            exact_path[codes[:4]].append(candidate)
        if len(codes) >= 3:
            prefix_l123[codes[:3]].append(candidate)
        if len(codes) >= 2:
            prefix_l12[codes[:2]].append(candidate)
    return {
        "candidate_by_id": candidate_by_id,
        "exact_path": dict(exact_path),
        "prefix_l123": dict(prefix_l123),
        "prefix_l12": dict(prefix_l12),
    }


def _iter_indexed_candidates(codes: Sequence[str], candidate_index: Mapping[str, Any]):
    seen: set[str] = set()
    lookup_plan = []
    code_tuple = tuple(str(item) for item in codes)
    if len(code_tuple) >= 4:
        lookup_plan.append(("exact_path", code_tuple[:4], "exact_code_path"))
    if len(code_tuple) >= 3:
        lookup_plan.append(("prefix_l123", code_tuple[:3], "prefix_l1_l2_l3"))
    if len(code_tuple) >= 2:
        lookup_plan.append(("prefix_l12", code_tuple[:2], "prefix_l1_l2"))
    for index_name, key, source in lookup_plan:
        for candidate in candidate_index[index_name].get(key, []):
            candidate_id = str(candidate.get("candidate_id"))
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            yield candidate, source


def _m4_prior_by_candidate_id(m4_retrieved: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    total = max(len(m4_retrieved), 1)
    priors = {}
    for rank, item in enumerate(m4_retrieved, start=1):
        candidate_id = item.get("candidate_id")
        if candidate_id is None:
            continue
        priors[str(candidate_id)] = {
            "rank": rank,
            "rank_score": (total - rank + 1) / total,
            "code_match_score": float(item.get("code_match_score") or 0.0),
        }
    return priors


def _candidate_score(
    path,
    codes: Sequence[str],
    candidate_codes: Sequence[str],
    overlap: int,
    m4_prior: Mapping[str, Any] | None,
    exact_bucket_size: int,
    exact_first_retrieval: bool,
) -> float:
    if exact_first_retrieval:
        return _exact_first_candidate_score(path, codes, candidate_codes, overlap, m4_prior, exact_bucket_size)
    return _phase2_candidate_score(path, overlap, candidate_codes, m4_prior)


def _phase2_candidate_score(path, overlap: int, candidate_codes: Sequence[str], m4_prior: Mapping[str, Any] | None) -> float:
    path_probability = float(path.get("probability") or path.get("score") or 0.0)
    overlap_ratio = overlap / max(len(candidate_codes), 1)
    score = path_probability * overlap_ratio
    if m4_prior is not None:
        score += 2.0 * float(m4_prior.get("rank_score") or 0.0)
        score += 0.5 * float(m4_prior.get("code_match_score") or 0.0)
        score += 0.2 * overlap_ratio
    return score


def _exact_first_candidate_score(
    path,
    codes: Sequence[str],
    candidate_codes: Sequence[str],
    overlap: int,
    m4_prior: Mapping[str, Any] | None,
    exact_bucket_size: int,
) -> float:
    path_probability = float(path.get("probability") or path.get("score") or 0.0)
    exact_match = int(len(codes) >= 4 and len(candidate_codes) >= 4 and tuple(codes[:4]) == tuple(candidate_codes[:4]))
    prefix_match_ratio = _prefix_match_count(codes, candidate_codes) / 4.0
    matched_level_ratio = overlap / 4.0
    m4_candidate_prior = float(m4_prior.get("rank_score") or 0.0) if m4_prior is not None else 0.0
    bucket_penalty = 0.05 * math.log1p(exact_bucket_size)
    generic_penalty = _generic_path_penalty(candidate_codes)
    return (
        2.0 * exact_match
        + 0.8 * prefix_match_ratio
        + 0.5 * matched_level_ratio
        + 0.5 * m4_candidate_prior
        + 0.2 * path_probability
        - bucket_penalty
        - generic_penalty
    )


def _prefix_match_count(left: Sequence[str], right: Sequence[str]) -> int:
    count = 0
    for first, second in zip(left[:4], right[:4]):
        if first != second:
            break
        count += 1
    return count


def _generic_path_penalty(candidate_codes: Sequence[str]) -> float:
    path_text = "/".join(str(item).lower() for item in candidate_codes)
    return 0.25 if any(marker in path_text for marker in GENERIC_PATH_MARKERS) else 0.0


def _exact_bucket_size(codes: Sequence[str], candidate_index: Mapping[str, Any]) -> int:
    code_tuple = tuple(str(item) for item in codes)
    if len(code_tuple) < 4:
        return 0
    return len(candidate_index["exact_path"].get(code_tuple[:4], []))


def _retrieval_sort_key(row: Mapping[str, Any], exact_first_retrieval: bool):
    if exact_first_retrieval:
        return (row["code_match_score"], float(row.get("m4_candidate_prior") or 0.0), row["matched_levels"])
    return (float(row.get("m4_candidate_prior") or 0.0), row["code_match_score"], row["matched_levels"])


def _upsert_retrieved_row(
    rows_by_id: dict[str, dict[str, Any]],
    candidate: Mapping[str, Any],
    score: float,
    overlap: int,
    m4_prior: Mapping[str, Any] | None,
    retrieval_source: str,
) -> None:
    candidate_id = str(candidate["candidate_id"])
    existing = rows_by_id.get(candidate_id)
    if existing is not None and float(existing.get("code_match_score") or 0.0) >= score:
        return
    rows_by_id[candidate_id] = {
        "candidate_id": candidate_id,
        "name": candidate.get("name"),
        "source_dataset": candidate.get("source_dataset"),
        "code_match_score": score,
        "matched_levels": overlap,
        "m4_candidate_prior": float(m4_prior.get("rank_score") or 0.0) if m4_prior is not None else 0.0,
        "m4_candidate_rank": int(m4_prior.get("rank")) if m4_prior is not None else None,
        "m4_code_match_score": float(m4_prior.get("code_match_score") or 0.0) if m4_prior is not None else 0.0,
        "retrieval_source": retrieval_source,
        "code_explanation": candidate.get("code_explanation"),
        "capability_text_evidence": str(candidate.get("text") or "")[:600],
    }


def _matched_code_path_codes(item: Mapping[str, Any]) -> list[str]:
    matched = item.get("matched_code_path")
    if isinstance(matched, Mapping):
        codes = matched.get("codes")
        if isinstance(codes, list):
            return [str(value) for value in codes]
    return []


def _candidate_id_set(candidates: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(item.get("candidate_id")) for item in candidates if item.get("candidate_id") is not None}


def _load_model(checkpoint_root, device, torch):
    vocab = json.loads((checkpoint_root / "vocab.json").read_text(encoding="utf-8"))
    code_vocabs = json.loads((checkpoint_root / "code_vocabs.json").read_text(encoding="utf-8"))
    role_vocab = json.loads((checkpoint_root / "role_vocab.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_root / "model.pt", map_location=device or "cpu")
    config = checkpoint["config"]
    model = build_code_path_planner_model(
        vocab_size=config["vocab_size"],
        code_vocab_sizes=config["code_vocab_sizes"],
        role_count=config["role_count"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(resolved_device)
    model.eval()
    return model, vocab, code_vocabs, role_vocab


def _build_vocab(rows, max_vocab_size):
    counts = Counter()
    for row in rows:
        counts.update(_tokens(str(row.get("planner_state") or "")))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _count in counts.most_common(max_vocab_size - len(vocab)):
        vocab[token] = len(vocab)
    return vocab


def _build_code_vocabs(rows):
    values = {level: [] for level in LEVELS}
    for row in rows:
        path = list(row.get("code_path") or [])
        if len(path) != 4:
            continue
        for index, level in enumerate(LEVELS):
            values[level].append(str(path[index]))
    return {level: {code: index for index, code in enumerate(sorted(set(codes)) or ["UNKNOWN"])} for level, codes in values.items()}


def _swanlab_payload(metrics):
    payload = {
        "train/loss": metrics.get("train_loss"),
        "train/code_loss": metrics.get("train_code_loss"),
        "train/coverage_loss": metrics.get("train_coverage_loss"),
        "train/role_loss": metrics.get("train_role_loss"),
        "train/stop_loss": metrics.get("train_stop_loss"),
        "dev/loss": metrics.get("dev_loss"),
        "dev/code_loss": metrics.get("dev_code_loss"),
        "dev/coverage_loss": metrics.get("dev_coverage_loss"),
        "dev/role_loss": metrics.get("dev_role_loss"),
        "dev/stop_loss": metrics.get("dev_stop_loss"),
        "dev/path_exact_match": metrics.get("dev_path_exact_match"),
        "dev/role_accuracy": metrics.get("dev_role_accuracy"),
        "dev/stop_accuracy": metrics.get("dev_stop_accuracy"),
    }
    return payload


def _path_attrs(path) -> Mapping[str, str]:
    codes = [str(item) for item in path.get("codes") or path.get("code_path") or []]
    return {
        "role": _normalize_role(path.get("role_hint") or (codes[2] if len(codes) > 2 else "")),
        "operation": _clean_code(codes[1] if len(codes) > 1 else "UNKNOWN"),
        "schema": _clean_code(codes[3] if len(codes) > 3 else "UNKNOWN"),
    }


def _covered_state(roles, operations, schemas):
    return {"roles": sorted(roles), "operations": sorted(operations), "schema_constraints": sorted(schemas)}


def _predicted_rank(semantic_id, predicted_paths):
    for index, path in enumerate(predicted_paths, start=1):
        if str(path.get("semantic_id") or "/".join(path.get("codes") or [])) == semantic_id:
            return index
    return None


def _purpose(attrs) -> str:
    role = attrs["role"]
    operation = attrs["operation"]
    schema = attrs["schema"]
    if role == "START":
        return f"initiate the plan with {operation} under {schema} constraints"
    if role == "CHECK":
        return f"verify unresolved {schema} constraints"
    if role == "FINALIZE":
        return f"complete the task with {operation}"
    if role == "SUPPORT":
        return f"support the plan with {operation} evidence"
    return f"cover remaining {operation} capability requirements"


def _role(codes: Sequence[str]) -> str:
    if len(codes) < 3:
        return "UNASSIGNED"
    return _normalize_role(codes[2])


def _normalize_role(value: Any) -> str:
    text = str(value or "").upper()
    for role in ROLES:
        if role in text:
            return role
    return "UNASSIGNED"


def _clean_code(value: Any) -> str:
    text = str(value or "UNKNOWN")
    parts = text.split("-")
    if parts and re.fullmatch(r"L[1-4]", parts[0], flags=re.IGNORECASE):
        parts = parts[1:]
        if len(parts) > 1:
            parts = parts[:-1]
        text = "-".join(parts) or text
    elif "-" in text:
        text = text.rsplit("-", 1)[0]
    return text.replace("_", " ")


def _batches(rows, batch_size):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if token]
