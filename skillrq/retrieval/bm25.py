"""BM25 lexical retrieval baseline."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import DefaultDict, Sequence

from .text import tokenize
from .types import Candidate


class BM25Index:
    def __init__(self, candidates: Sequence[Candidate], k1: float = 1.5, b: float = 0.75) -> None:
        self.candidates = list(candidates)
        self.k1 = k1
        self.b = b
        self.doc_lengths: list[int] = []
        self.avgdl = 0.0
        self.postings: DefaultDict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_freq: Counter[str] = Counter()
        self._build()

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores: Counter[int] = Counter()
        query_terms = set(tokenize(query))
        if not query_terms:
            return []
        doc_count = len(self.candidates)
        for term in query_terms:
            postings = self.postings.get(term)
            if not postings:
                continue
            df = self.doc_freq[term]
            idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            for doc_idx, tf in postings:
                dl = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[doc_idx] += idf * (tf * (self.k1 + 1)) / denom
        return [
            (self.candidates[doc_idx].candidate_id, float(score))
            for doc_idx, score in scores.most_common(top_k)
        ]

    def _build(self) -> None:
        for doc_idx, candidate in enumerate(self.candidates):
            counts = Counter(tokenize(candidate.text))
            length = sum(counts.values())
            self.doc_lengths.append(length)
            for term, tf in counts.items():
                self.postings[term].append((doc_idx, tf))
                self.doc_freq[term] += 1
        self.avgdl = (sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 0.0
