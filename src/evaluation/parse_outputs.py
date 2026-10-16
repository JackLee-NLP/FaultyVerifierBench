"""Deprecated compatibility layer for legacy imports."""
from __future__ import annotations

from src.evaluation.score_verifier_judgment import normalize_judgment
from src.utils.json_utils import safe_json_loads


def parse_output(raw_text):
    parsed = safe_json_loads(raw_text)
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else None
    return parsed if isinstance(parsed, dict) else {}


def normalize_verifier_judgment(label):
    return normalize_judgment(label)
