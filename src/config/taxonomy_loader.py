"""Shared taxonomy loader used by Step 08 prompts, validation, and Step 09."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _normalize_label(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


@lru_cache(maxsize=16)
def load_taxonomy_labels(path: str | Path, root_key: str) -> tuple[str, ...]:
    taxonomy_path = Path(path).resolve()
    if not taxonomy_path.is_file():
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")

    with taxonomy_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)

    if not isinstance(document, dict):
        raise ValueError(f"Expected a YAML mapping in {taxonomy_path}")
    if root_key not in document:
        raise KeyError(f"Missing root key {root_key!r} in {taxonomy_path}")

    values = document[root_key]
    if not isinstance(values, list):
        raise ValueError(f"{root_key!r} must be a YAML list in {taxonomy_path}")

    labels = tuple(_normalize_label(value) for value in values if str(value).strip())
    if not labels:
        raise ValueError(f"No labels found under {root_key!r} in {taxonomy_path}")
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate normalized labels in {taxonomy_path}")

    return labels


def load_benchmark_taxonomies(project_root: str | Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    root = Path(project_root)
    fallacies = load_taxonomy_labels(root / "config" / "fallacies.yaml", "fallacies")
    hallucinations = load_taxonomy_labels(
        root / "config" / "hallucinations.yaml",
        "hallucinations",
    )
    return fallacies, hallucinations
