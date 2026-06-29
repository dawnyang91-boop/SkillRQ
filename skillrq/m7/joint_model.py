"""Joint residual-code and reranking model for M7 ablations."""

from __future__ import annotations


LEVELS = ("l1", "l2", "l3", "l4")


def build_joint_reranker_model(
    vocab_size: int,
    feature_dim: int,
    code_vocab_sizes: dict[str, int],
    role_count: int,
    stage_count: int,
    embedding_dim: int = 512,
    hidden_dim: int = 1024,
    code_embedding_dim: int = 128,
    dropout: float = 0.1,
    enable_shared_encoder: bool = False,
    enable_soft_code_distribution: bool = False,
):
    from ..m4.torch_utils import require_torch

    torch = require_torch()
    nn = torch.nn

    class JointRerankerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enable_shared_encoder = enable_shared_encoder
            self.enable_soft_code_distribution = enable_soft_code_distribution
            self.rerank_query_embedding = MeanTextEncoder(vocab_size, embedding_dim)
            self.candidate_embedding = self.rerank_query_embedding if enable_shared_encoder else MeanTextEncoder(
                vocab_size,
                embedding_dim,
            )
            self.code_query_embedding = self.rerank_query_embedding if enable_shared_encoder else MeanTextEncoder(
                vocab_size,
                embedding_dim,
            )
            self.query_projection = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.code_projection = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.code_heads = nn.ModuleDict(
                {level: nn.Linear(hidden_dim, code_vocab_sizes[level]) for level in LEVELS}
            )
            self.code_embeddings = nn.ModuleDict(
                {
                    level: nn.Embedding(code_vocab_sizes[level], code_embedding_dim)
                    for level in LEVELS
                }
            )
            self.pair_mlp = nn.Sequential(
                nn.Linear(hidden_dim + embedding_dim + feature_dim + len(LEVELS) + code_embedding_dim, hidden_dim),
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

        def forward(self, query_token_ids, query_offsets, candidate_token_ids, candidate_offsets, numeric_features, candidate_code_ids):
            rerank_query = self.query_projection(self.rerank_query_embedding(query_token_ids, query_offsets))
            code_query = self.code_projection(self.code_query_embedding(query_token_ids, query_offsets))
            candidate_text = self.candidate_embedding(candidate_token_ids, candidate_offsets)
            code_logits = {level: self.code_heads[level](code_query) for level in LEVELS}
            soft_code_scores = []
            expected_code_embedding = None
            for index, level in enumerate(LEVELS):
                ids = candidate_code_ids[:, index].clamp(0, self.code_embeddings[level].num_embeddings - 1)
                if self.enable_soft_code_distribution:
                    probabilities = torch.softmax(code_logits[level], dim=-1)
                    level_score = probabilities.gather(1, ids.unsqueeze(1)).squeeze(1)
                    level_embedding = probabilities @ self.code_embeddings[level].weight
                else:
                    level_score = torch.zeros(ids.shape[0], device=ids.device)
                    level_embedding = self.code_embeddings[level](ids)
                soft_code_scores.append(level_score.unsqueeze(1))
                expected_code_embedding = level_embedding if expected_code_embedding is None else expected_code_embedding + level_embedding
            soft_code_features = torch.cat(soft_code_scores, dim=-1)
            expected_code_embedding = expected_code_embedding / len(LEVELS)
            pair_features = torch.cat(
                [rerank_query, candidate_text, numeric_features, soft_code_features, expected_code_embedding],
                dim=-1,
            )
            hidden = self.pair_mlp(pair_features)
            return {
                "code_logits": code_logits,
                "soft_code_scores": soft_code_features,
                "relevance": self.relevance_head(hidden).squeeze(-1),
                "role": self.role_head(hidden),
                "stage": self.stage_head(hidden),
                "order": self.order_head(hidden).squeeze(-1),
            }

    class MeanTextEncoder(nn.Module):
        def __init__(self, num_embeddings: int, dim: int) -> None:
            super().__init__()
            self.embedding = nn.Embedding(num_embeddings, dim)
            self.num_embeddings = num_embeddings

        def forward(self, token_ids, offsets):
            token_ids = token_ids.clamp(0, self.num_embeddings - 1)
            embeddings = self.embedding(token_ids)
            end_offsets = torch.cat([offsets[1:], token_ids.new_tensor([token_ids.numel()])])
            lengths = (end_offsets - offsets).clamp(min=1)
            bag_ids = torch.repeat_interleave(torch.arange(offsets.numel(), device=token_ids.device), lengths)
            pooled = embeddings.new_zeros((offsets.numel(), embeddings.shape[-1]))
            pooled.index_add_(0, bag_ids, embeddings)
            return pooled / lengths.unsqueeze(1).to(embeddings.dtype)

    return JointRerankerModel()
