from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.dataset.io_utils import read_jsonl, write_jsonl


DEFAULT_INPUT_PATH = "data/processed/verifier_validated.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/benchmark_dataset.jsonl"
ALLOWED_VERIFIER_CORRECTNESS = {"correct", "faulty", "mixed"}


def _normalise_label(value: Any) -> str:
    return str(value).strip().lower()


def _assert_validated_item(item: dict[str, Any], row_number: int) -> None:
    errors: list[str] = []

    if item.get("verifier_validation_status") != "validated":
        errors.append(
            "verifier_validation_status is not 'validated': "
            f"{item.get('verifier_validation_status')!r}"
        )

    if item.get("verifier_validation_errors"):
        errors.append(
            "verifier_validation_errors is non-empty: "
            f"{item.get('verifier_validation_errors')!r}"
        )

    correctness = _normalise_label(item.get("verifier_correctness", ""))
    if correctness not in ALLOWED_VERIFIER_CORRECTNESS:
        errors.append(
            "invalid verifier_correctness: "
            f"{item.get('verifier_correctness')!r}"
        )

    if not str(item.get("verifier_response", "")).strip():
        errors.append("missing verifier_response")

    if errors:
        item_id = item.get("id") or item.get("verifier_id") or "<unknown>"
        raise ValueError(
            "Refusing to build benchmark from an invalid verifier row. "
            f"row={row_number}, id={item_id}, errors={errors}"
        )


def merge_dataset(
    items: Iterable[dict[str, Any]] | None = None,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, Any]]:
    """Build benchmark_dataset.jsonl from Step 06 validated rows only.

    No invalid row is silently skipped. If any supplied row is not explicitly
    validated, the build fails before benchmark_dataset.jsonl is overwritten.
    """

    if items is None:
        source_items = list(read_jsonl(input_path))
        source_description = str(input_path)
    else:
        source_items = list(items)
        source_description = "in-memory Step 06 validated items"

    if not source_items:
        raise ValueError(
            f"No validated verifier rows were available from {source_description}."
        )

    merged_items: list[dict[str, Any]] = []

    # Validate every source row before writing anything. This prevents a partial
    # or contaminated benchmark_dataset.jsonl when one bad row is encountered.
    for row_number, source_item in enumerate(source_items, start=1):
        item = dict(source_item)
        _assert_validated_item(item, row_number)

        item["expected_final_answer"] = item.get("gold_answer", "")
        item["validation_status"] = "validated"
        merged_items.append(item)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(merged_items, output_path)

    print(f"Input source: {source_description}", flush=True)
    print(f"Validated input rows: {len(source_items)}", flush=True)
    print(f"Benchmark rows written: {len(merged_items)}", flush=True)
    print(f"Output: {output_path}", flush=True)

    return merged_items
