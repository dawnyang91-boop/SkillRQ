"""PyTorch model for M7 reranking."""

from __future__ import annotations


def build_reranker_model(
    vocab_size: int,
    feature_dim: int,
    role_count: int,
    stage_count: int,
    embedding_dim: int = 512,
    hidden_dim: int = 1024,
    dropout: float = 0.1,
):
    from ..m4.torch_utils import require_torch

    torch = require_torch()
    nn = torch.nn

    class RerankerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim + feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.relevance_head = nn.Linear(hidden_dim, 1)
            self.role_head = nn.Linear(hidden_dim, role_count)
            self.stage_head = nn.Linear(hidden_dim, stage_count)
            self.order_head = nn.Linear(hidden_dim, 1)

        def forward(self, token_ids, offsets, numeric_features):
            text_features = self.embedding(token_ids, offsets)
            hidden = self.mlp(torch.cat([text_features, numeric_features], dim=-1))
            return {
                "relevance": self.relevance_head(hidden).squeeze(-1),
                "role": self.role_head(hidden),
                "stage": self.stage_head(hidden),
                "order": self.order_head(hidden).squeeze(-1),
            }

    return RerankerModel()
