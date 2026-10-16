"""Deprecated compatibility module.

FaultyVerifierBench Step 09 scores must be reconstructed from current Step 08
outputs by running scripts/09_compute_scores.py.
"""
from __future__ import annotations


def _deprecated() -> None:
    raise RuntimeError(
        "src.evaluation.compute_metrics is deprecated. "
        "Run: python scripts/09_compute_scores.py"
    )


def compute_metrics(*args, **kwargs):
    del args, kwargs
    _deprecated()


def auto_compute_metrics(*args, **kwargs):
    del args, kwargs
    _deprecated()
