"""Shared split helpers."""

from __future__ import annotations


EVAL_SPLITS = {"dev", "test", "sequence_dev", "sequence_test"}


def is_eval_split(value: object) -> bool:
    return str(value or "") in EVAL_SPLITS
