"""Neural and quantization baselines for M4 code retrieval."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .swanlab_utils import SwanLabLogger
from .torch_utils import require_torch
from ..utils.io import read_jsonl, write_json, write_jsonl


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def train_rq_kmeans(
    data_root: Path,
    output_root: Path,
    num_levels: int = 4,
    codebook_size: int = 256,
    iterations: int = 25,
    feature_dim: int = 2048,
    device: str | None = None,
    swanlab_project: str | None = "SkillRQ-M4",
    swanlab_run_name: str | None = None,
) -> Mapping[str, Any]:
    torch = require_torch()
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = list(read_jsonl(data_root / "candidates.jsonl"))
    vectors = _candidate_matrix(candidates, feature_dim, torch)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    residual = vectors.to(resolved_device)
    codebooks = []
    assignments_by_level = []
    logger = SwanLabLogger(
        project=swanlab_project,
        run_name=swanlab_run_name,
        config={
            "method": "rq_kmeans",
            "data_root": str(data_root),
            "output_root": str(output_root),
            "candidates": len(candidates),
            "num_levels": num_levels,
            "codebook_size": codebook_size,
            "iterations": iterations,
            "feature_dim": feature_dim,
            "device": str(resolved_device),
        },
        tags=["m4", "rq-kmeans", "code-retrieval"],
    )
    for _level in range(num_levels):
        centroids, assignments = _kmeans(residual, codebook_size, iterations, torch)
        codebooks.append(centroids.detach().cpu())
        assignments_by_level.append(assignments.detach().cpu())
        residual = residual - centroids[assignments]
        logger.log(
            {
                "rq_kmeans/level": _level + 1,
                "rq_kmeans/residual_l2_mean": float(torch.linalg.norm(residual, dim=1).mean().detach().cpu()),
            },
            step=_level + 1,
        )
    logger.finish()

    rows = []
    for index, candidate in enumerate(candidates):
        codes = [f"RQK-L{level + 1}-{int(assignments_by_level[level][index])}" for level in range(num_levels)]
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "semantic_id": "/".join(codes),
                "code_path": codes,
                "method": "rq_kmeans",
            }
        )
    torch.save({"codebooks": codebooks, "feature_dim": feature_dim}, output_root / "rq_kmeans.pt")
    write_jsonl(output_root / "code_assignments.jsonl", rows)
    summary = {
        "method": "rq_kmeans",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "candidates": len(candidates),
        "num_levels": num_levels,
        "codebook_size": codebook_size,
        "iterations": iterations,
        "feature_dim": feature_dim,
        "swanlab_project": swanlab_project,
        "swanlab_run_name": swanlab_run_name,
    }
    write_json(output_root / "train_summary.json", summary)
    return summary


def train_rq_vae(
    data_root: Path,
    output_root: Path,
    epochs: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    feature_dim: int = 2048,
    latent_dim: int = 256,
    num_levels: int = 4,
    codebook_size: int = 256,
    commitment_weight: float = 0.25,
    device: str | None = None,
    swanlab_project: str | None = "SkillRQ-M4",
    swanlab_run_name: str | None = None,
) -> Mapping[str, Any]:
    torch = require_torch()
    nn = torch.nn
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = list(read_jsonl(data_root / "candidates.jsonl"))
    vectors = _candidate_matrix(candidates, feature_dim, torch)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vectors = vectors.to(resolved_device)

    class RQVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(feature_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))
            self.decoder = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, feature_dim))
            self.codebooks = nn.ParameterList(
                [nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.02) for _ in range(num_levels)]
            )

        def forward(self, x):
            z = self.encoder(x)
            residual = z
            quantized_sum = torch.zeros_like(z)
            indices = []
            for codebook in self.codebooks:
                distances = torch.cdist(residual, codebook)
                index = distances.argmin(dim=-1)
                q = codebook[index]
                quantized_sum = quantized_sum + q
                residual = residual - q
                indices.append(index)
            z_q = z + (quantized_sum - z).detach()
            recon = self.decoder(z_q)
            commit = torch.mean((z.detach() - quantized_sum) ** 2) + torch.mean((z - quantized_sum.detach()) ** 2)
            return recon, commit, indices

    model = RQVAE().to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    logger = SwanLabLogger(
        project=swanlab_project,
        run_name=swanlab_run_name,
        config={
            "method": "rq_vae",
            "data_root": str(data_root),
            "output_root": str(output_root),
            "candidates": len(candidates),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "feature_dim": feature_dim,
            "latent_dim": latent_dim,
            "num_levels": num_levels,
            "codebook_size": codebook_size,
            "commitment_weight": commitment_weight,
            "device": str(resolved_device),
        },
        tags=["m4", "rq-vae", "code-retrieval"],
    )
    history = []
    try:
        for epoch in range(1, epochs + 1):
            permutation = torch.randperm(vectors.shape[0], device=resolved_device)
            total_loss = 0.0
            total = 0
            for start in range(0, vectors.shape[0], batch_size):
                batch = vectors[permutation[start : start + batch_size]]
                recon, commit, _indices = model(batch)
                recon_loss = torch.nn.functional.mse_loss(recon, batch)
                loss = recon_loss + commitment_weight * commit
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * batch.shape[0]
                total += batch.shape[0]
            epoch_metrics = {"epoch": epoch, "loss": total_loss / max(total, 1)}
            history.append(epoch_metrics)
            logger.log({"rq_vae/loss": epoch_metrics["loss"]}, step=epoch)
    finally:
        logger.finish()

    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, vectors.shape[0], batch_size):
            _recon, _commit, indices = model(vectors[start : start + batch_size])
            stacked = [index.detach().cpu().tolist() for index in indices]
            batch_size_actual = len(stacked[0]) if stacked else 0
            for local_index in range(batch_size_actual):
                global_index = start + local_index
                codes = [f"RQVAE-L{level + 1}-{stacked[level][local_index]}" for level in range(num_levels)]
                rows.append(
                    {
                        "candidate_id": candidates[global_index]["candidate_id"],
                        "semantic_id": "/".join(codes),
                        "code_path": codes,
                        "method": "rq_vae",
                    }
                )
    torch.save({"model_state_dict": model.state_dict(), "feature_dim": feature_dim}, output_root / "rq_vae.pt")
    write_jsonl(output_root / "code_assignments.jsonl", rows)
    summary = {
        "method": "rq_vae",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "candidates": len(candidates),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "feature_dim": feature_dim,
        "latent_dim": latent_dim,
        "num_levels": num_levels,
        "codebook_size": codebook_size,
        "commitment_weight": commitment_weight,
        "history": history,
        "swanlab_project": swanlab_project,
        "swanlab_run_name": swanlab_run_name,
    }
    write_json(output_root / "train_summary.json", summary)
    return summary


def _kmeans(vectors, k: int, iterations: int, torch):
    n = vectors.shape[0]
    if n < k:
        k = n
    initial = torch.randperm(n, device=vectors.device)[:k]
    centroids = vectors[initial].clone()
    assignments = torch.zeros(n, dtype=torch.long, device=vectors.device)
    for _ in range(iterations):
        distances = torch.cdist(vectors, centroids)
        assignments = distances.argmin(dim=-1)
        for cluster_id in range(k):
            mask = assignments == cluster_id
            if bool(mask.any()):
                centroids[cluster_id] = vectors[mask].mean(dim=0)
    return centroids, assignments


def _candidate_matrix(candidates: Sequence[Mapping[str, Any]], feature_dim: int, torch):
    rows = [_hashed_vector(str(candidate.get("text") or ""), feature_dim) for candidate in candidates]
    return torch.stack(rows, dim=0)


def _hashed_vector(text: str, feature_dim: int):
    torch = require_torch()
    vector = torch.zeros(feature_dim)
    for token in TOKEN_RE.findall(text.lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
        dim = value % feature_dim
        sign = 1.0 if ((value >> 8) & 1) == 0 else -1.0
        vector[dim] += sign
    norm = torch.linalg.norm(vector)
    return vector / norm if float(norm) > 0 else vector
