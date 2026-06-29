"""Feature helpers for M7 reranking."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Mapping, Sequence


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
FEATURE_KEYS = (
    "code_match_score",
    "matched_levels",
    "text_overlap_score",
    "schema_evidence_score",
    "parameter_compatibility_score",
    "role_compatibility_score",
    "coverage_gain_score",
    "hypergraph_support_score",
    "first_tool_prior",
    "transition_prior",
    "redundancy_penalty",
    "generic_penalty",
    "constraint_violation_penalty",
)
ROLES = ("UNKNOWN", "START", "SUPPORT", "CHECK", "FINALIZE", "AVOID", "FALLBACK")
STAGES = ("UNKNOWN", "FIRST", "MIDDLE", "FINAL", "CHECK", "AVOID")


def build_feature_dict(
    query: str,
    candidate: Mapping[str, Any],
    matched_levels: int = 0,
    code_match_score: float = 0.0,
    coverage_gain_score: float = 0.0,
    step_index: int | None = None,
    duplicate: bool = False,
    hypergraph_support_score: float = 0.0,
) -> dict[str, float]:
    candidate_text = str(candidate.get("text") or candidate.get("capability_text_evidence") or "")
    query_tokens = set(tokens(query))
    candidate_tokens = set(tokens(candidate_text))
    overlap = _jaccard(query_tokens, candidate_tokens)
    parameter_score = _parameter_score(query_tokens, candidate_text)
    role = normalize_role(candidate.get("role_hint"))
    features = {
        "code_match_score": float(code_match_score),
        "matched_levels": float(matched_levels) / 4.0,
        "text_overlap_score": overlap,
        "schema_evidence_score": min(1.0, overlap * 2.0),
        "parameter_compatibility_score": parameter_score,
        "role_compatibility_score": role_compatibility(role, step_index),
        "coverage_gain_score": float(coverage_gain_score),
        "hypergraph_support_score": float(hypergraph_support_score),
        "first_tool_prior": 1.0 if role == "START" else 0.25 if role in {"SUPPORT", "CHECK"} else 0.0,
        "transition_prior": transition_prior(role),
        "redundancy_penalty": 1.0 if duplicate else 0.0,
        "generic_penalty": generic_penalty(candidate_text),
        "constraint_violation_penalty": 1.0 if role == "AVOID" else 0.0,
    }
    return {key: float(features.get(key, 0.0)) for key in FEATURE_KEYS}


def feature_vector(features: Mapping[str, Any]) -> list[float]:
    return [float(features.get(key, 0.0) or 0.0) for key in FEATURE_KEYS]


def infer_stage(role: str | None, sequence_position: int | None, sequence_length: int | None) -> str:
    role = normalize_role(role)
    if role == "AVOID":
        return "AVOID"
    if role == "CHECK":
        return "CHECK"
    if sequence_position is None or sequence_position < 0 or not sequence_length:
        return "UNKNOWN"
    if sequence_position == 0:
        return "FIRST"
    if sequence_position == sequence_length - 1:
        return "FINAL"
    return "MIDDLE"


def normalize_role(role: Any) -> str:
    value = str(role or "UNKNOWN").upper()
    return value if value in ROLES else "UNKNOWN"


def role_compatibility(role: str, step_index: int | None) -> float:
    if step_index is None:
        return 0.0
    if step_index == 0 and role == "START":
        return 1.0
    if step_index > 0 and role in {"SUPPORT", "CHECK", "FINALIZE"}:
        return 0.75
    if role == "AVOID":
        return -1.0
    return 0.0


def transition_prior(role: str) -> float:
    if role == "START":
        return 1.0
    if role in {"SUPPORT", "CHECK"}:
        return 0.65
    if role == "FINALIZE":
        return 0.45
    if role == "AVOID":
        return -1.0
    return 0.0


def generic_penalty(text: str) -> float:
    token_counts = Counter(tokens(text))
    if not token_counts:
        return 1.0
    generic_terms = {"tool", "api", "function", "skill", "helper", "utility", "general", "generic"}
    hits = sum(token_counts.get(term, 0) for term in generic_terms)
    return min(1.0, hits / math.sqrt(sum(token_counts.values())))


def support_evidence(query: str, candidate: Mapping[str, Any], max_chars: int = 360) -> str:
    text = str(candidate.get("text") or candidate.get("capability_text_evidence") or "")
    query_terms = set(tokens(query))
    fragments = []
    for sentence in re.split(r"(?<=[.!?。！？])\s+", text):
        if set(tokens(sentence)) & query_terms:
            fragments.append(sentence.strip())
        if sum(len(item) for item in fragments) >= max_chars:
            break
    evidence = " ".join(item for item in fragments if item)
    if not evidence:
        evidence = text[:max_chars]
    return evidence[:max_chars]


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if token]


def code_overlap(left: Sequence[Any], right: Sequence[Any]) -> int:
    return sum(1 for a, b in zip(left, right) if str(a) == str(b))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _parameter_score(query_tokens: set[str], candidate_text: str) -> float:
    candidate_tokens = set(tokens(candidate_text))
    parameter_markers = {"parameter", "parameters", "input", "schema", "argument", "arguments", "required"}
    if not (candidate_tokens & parameter_markers):
        return 0.0
    return min(1.0, len(query_tokens & candidate_tokens) / max(len(query_tokens), 1))
