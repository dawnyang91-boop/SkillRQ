"""PyTorch residual code path selector model."""

from __future__ import annotations

from typing import Mapping

from ..m4.torch_utils import require_torch


def build_residual_selector_model(
    vocab_size: int,
    code_vocab_sizes: Mapping[str, int],
    embedding_dim: int = 256,
    hidden_dim: int = 512,
    dropout: float = 0.1,
):
    torch = require_torch()
    nn = torch.nn

    class ResidualSelectorModel(nn.Module):
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
            self.heads = nn.ModuleDict({level: nn.Linear(hidden_dim, size) for level, size in code_vocab_sizes.items()})
            self.coverage_gain_head = nn.Linear(hidden_dim, 1)

        def forward(self, token_ids, offsets):
            hidden = self.encoder(self.embedding(token_ids, offsets))
            return {
                "codes": {level: head(hidden) for level, head in self.heads.items()},
                "coverage_gain": self.coverage_gain_head(hidden).squeeze(-1),
            }

    return ResidualSelectorModel()
