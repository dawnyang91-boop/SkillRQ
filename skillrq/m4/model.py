"""PyTorch models for query-to-code prediction."""

from __future__ import annotations

from typing import Mapping

from .torch_utils import require_torch


def build_query_code_model(
    vocab_size: int,
    code_vocab_sizes: Mapping[str, int],
    embedding_dim: int = 256,
    hidden_dim: int = 512,
    dropout: float = 0.1,
):
    torch = require_torch()
    nn = torch.nn

    class QueryCodeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean", sparse=False)
            self.encoder = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.heads = nn.ModuleDict(
                {
                    level: nn.Linear(hidden_dim, int(size))
                    for level, size in code_vocab_sizes.items()
                }
            )

        def forward(self, token_ids, offsets):
            features = self.embedding(token_ids, offsets)
            hidden = self.encoder(features)
            return {level: head(hidden) for level, head in self.heads.items()}

    return QueryCodeModel()


def build_soft_multipath_code_model(
    vocab_size: int,
    code_vocab_sizes: Mapping[str, int],
    embedding_dim: int = 256,
    hidden_dim: int = 512,
    code_embedding_dim: int = 128,
    dropout: float = 0.1,
):
    torch = require_torch()
    nn = torch.nn

    class SoftMultiPathCodeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text_embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean", sparse=False)
            self.query_encoder = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.code_embeddings = nn.ModuleDict(
                {
                    level: nn.Embedding(int(size), code_embedding_dim)
                    for level, size in code_vocab_sizes.items()
                }
            )
            self.l1_head = nn.Linear(hidden_dim, int(code_vocab_sizes["l1"]))
            self.l2_head = _conditional_head(hidden_dim, code_embedding_dim, int(code_vocab_sizes["l2"]))
            self.l3_head = _conditional_head(hidden_dim, code_embedding_dim * 2, int(code_vocab_sizes["l3"]))
            self.l4_head = _conditional_head(hidden_dim, code_embedding_dim * 3, int(code_vocab_sizes["l4"]))
            self.path_projection = nn.Sequential(
                nn.Linear(code_embedding_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def encode_query(self, token_ids, offsets):
            features = self.text_embedding(token_ids, offsets)
            return self.query_encoder(features)

        def level_logits(self, query_hidden, prefix_ids=None):
            prefix_ids = prefix_ids or {}
            logits = {"l1": self.l1_head(query_hidden)}
            if "l1" in prefix_ids:
                l1_embedding = self.code_embeddings["l1"](prefix_ids["l1"])
                logits["l2"] = self.l2_head(torch.cat([query_hidden, l1_embedding], dim=-1))
            if "l1" in prefix_ids and "l2" in prefix_ids:
                l1_embedding = self.code_embeddings["l1"](prefix_ids["l1"])
                l2_embedding = self.code_embeddings["l2"](prefix_ids["l2"])
                logits["l3"] = self.l3_head(torch.cat([query_hidden, l1_embedding, l2_embedding], dim=-1))
            if "l1" in prefix_ids and "l2" in prefix_ids and "l3" in prefix_ids:
                l1_embedding = self.code_embeddings["l1"](prefix_ids["l1"])
                l2_embedding = self.code_embeddings["l2"](prefix_ids["l2"])
                l3_embedding = self.code_embeddings["l3"](prefix_ids["l3"])
                logits["l4"] = self.l4_head(
                    torch.cat([query_hidden, l1_embedding, l2_embedding, l3_embedding], dim=-1)
                )
            return logits

        def encode_paths(self, path_level_ids):
            embeddings = [
                self.code_embeddings[level](path_level_ids[level])
                for level in ("l1", "l2", "l3", "l4")
            ]
            return self.path_projection(torch.cat(embeddings, dim=-1))

        def path_logits(self, query_hidden, path_level_ids, temperature: float = 0.07):
            query_hidden = torch.nn.functional.normalize(query_hidden, dim=-1)
            path_hidden = torch.nn.functional.normalize(self.encode_paths(path_level_ids), dim=-1)
            return query_hidden @ path_hidden.transpose(0, 1) / max(float(temperature), 1e-6)

        def forward(self, token_ids, offsets):
            query_hidden = self.encode_query(token_ids, offsets)
            return self.level_logits(query_hidden), query_hidden

    return SoftMultiPathCodeModel()


def _conditional_head(hidden_dim: int, prefix_dim: int, output_dim: int):
    torch = require_torch()
    nn = torch.nn
    return nn.Sequential(
        nn.Linear(hidden_dim + prefix_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )
