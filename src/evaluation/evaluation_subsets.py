"""Evaluation subset construction for FaultyVerifierBench Step 09."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .metric_utils import common_ids, safe_divide


def get_common_item_ids(
    gold_ids: set[str],
    output_ids_by_model: Mapping[str, set[str]],
) -> set[str]:
    return common_ids(gold_ids, output_ids_by_model)


def filter_rows_by_ids(
    rows: Sequence[dict[str, Any]],
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("id", "")) in selected_ids]


def build_common_item_summary(
    gold_ids: set[str],
    output_ids_by_model: Mapping[str, set[str]],
) -> dict[str, Any]:
    common = get_common_item_ids(gold_ids, output_ids_by_model)
    return {
        "n_gold": len(gold_ids),
        "n_common_items": len(common),
        "common_coverage_rate": safe_divide(len(common), len(gold_ids)),
        "common_item_ids": sorted(common),
        "models": {
            model_key: {
                "n_outputs": len(model_ids),
                "n_gold_matched": len(gold_ids & model_ids),
                "n_missing_from_gold": len(gold_ids - model_ids),
                "n_excluded_from_common": len((gold_ids & model_ids) - common),
            }
            for model_key, model_ids in sorted(output_ids_by_model.items())
        },
    }
