"""PyTorch import helpers for cloud training commands."""

from __future__ import annotations


def require_torch():
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for this M4 training command. "
            "Install it on the training machine, e.g. `pip install torch` "
            "or use the CUDA wheel matching your server."
        ) from exc
    return torch
