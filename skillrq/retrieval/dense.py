"""Dependency-free hashing dense retrieval baseline."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import DefaultDict, Sequence

from .text import tokenize
from .types import Candidate


class HashingDenseIndex:
    """A deterministic local dense baseline based on signed feature hashing."""

    def __init__(self, candidates: Sequence[Candidate], dimensions: int = 2048) -> None:
        self.candidates = list(candidates)
        self.dimensions = dimensions
        self.postings: DefaultDict[int, list[tuple[int, float]]] = defaultdict(list)
        self._build()

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        query_vector = _hashed_vector(query, self.dimensions)
        scores: Counter[int] = Counter()
        for dim, query_weight in query_vector.items():
            for doc_idx, doc_weight in self.postings.get(dim, []):
                scores[doc_idx] += query_weight * doc_weight
        return [
            (self.candidates[doc_idx].candidate_id, float(score))
            for doc_idx, score in scores.most_common(top_k)
        ]

    def _build(self) -> None:
        for doc_idx, candidate in enumerate(self.candidates):
            vector = _hashed_vector(candidate.text, self.dimensions)
            for dim, weight in vector.items():
                self.postings[dim].append((doc_idx, weight))


def _hashed_vector(text: str, dimensions: int) -> dict[int, float]:
    counts: Counter[int] = Counter()
    for feature in _features(text):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
        dim = value % dimensions
        sign = 1 if ((value >> 12) & 1) == 0 else -1
        counts[dim] += sign
    norm = math.sqrt(sum(value * value for value in counts.values()))
    if not norm:
        return {}
    return {dim: value / norm for dim, value in counts.items()}


def _features(text: str) -> list[str]:
    tokens = tokenize(text)
    features = list(tokens)
    features.extend(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))
    return features
