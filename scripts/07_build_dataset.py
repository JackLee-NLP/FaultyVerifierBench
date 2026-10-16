import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.dataset.merge_dataset import merge_dataset
from src.dataset.split_dataset import split_dataset
from src.validation.validate_verifier import validate_verifier


def main() -> None:
    print("=" * 80, flush=True)
    print("[SCRIPT] Building full benchmark dataset", flush=True)
    print("[SCRIPT] Invalid verifier rows will not be included", flush=True)
    print("=" * 80, flush=True)

    print(
        "[STEP 1/3] Revalidating raw verifier data against the configured taxonomies...",
        flush=True,
    )

    # Always regenerate verifier_validated.jsonl before building. This prevents a
    # stale file produced by the old permissive validator from entering Step 07.
    validated_items = validate_verifier()

    if validated_items is None:
        raise RuntimeError("validate_verifier() returned None.")

    if not validated_items:
        raise RuntimeError(
            "Verifier validation produced no valid rows. "
            "The benchmark dataset will not be built."
        )

    print(
        "[STEP 1/3] Validation completed | "
        f"valid_items={len(validated_items)}",
        flush=True,
    )

    print(
        "[STEP 2/3] Merging validated verifier data only...",
        flush=True,
    )

    # Pass the in-memory validated rows directly. merge_dataset() also performs
    # defensive status checks and refuses any invalid row.
    merged_items = merge_dataset(items=validated_items)

    if merged_items is None:
        raise RuntimeError(
            "merge_dataset() returned None. "
            "The full dataset cannot be passed to split_dataset()."
        )

    if not merged_items:
        raise RuntimeError("merge_dataset() returned an empty dataset.")

    if len(merged_items) != len(validated_items):
        raise RuntimeError(
            "Validated-to-merged row-count mismatch. "
            f"validated={len(validated_items)}, merged={len(merged_items)}"
        )

    invalid_merged_rows = [
        item
        for item in merged_items
        if item.get("verifier_validation_status") != "validated"
        or item.get("validation_status") != "validated"
        or item.get("verifier_validation_errors")
    ]

    if invalid_merged_rows:
        raise RuntimeError(
            "Invalid verifier rows were found after merging. "
            "No dataset split will be built. "
            f"invalid_rows={len(invalid_merged_rows)}"
        )

    print(
        "[STEP 2/3] Merge completed | "
        f"items={len(merged_items)}",
        flush=True,
    )

    print(
        "[STEP 3/3] Splitting the full merged dataset...",
        flush=True,
    )

    # Pass the newly merged full dataset directly. split_dataset() must not reload
    # an old or previously balanced dataset from disk.
    dev_items, test_items = split_dataset(items=merged_items)

    total_split_items = len(dev_items) + len(test_items)

    print("[STEP 3/3] Split completed.", flush=True)
    print(f"[INFO] Dev items: {len(dev_items)}", flush=True)
    print(f"[INFO] Test items: {len(test_items)}", flush=True)
    print(f"[INFO] Total split items: {total_split_items}", flush=True)

    if total_split_items != len(merged_items):
        raise RuntimeError(
            "Full-dataset split count mismatch. "
            f"merged={len(merged_items)}, split_total={total_split_items}"
        )

    dev_question_ids = {
        item.get("question_id")
        for item in dev_items
        if item.get("question_id")
    }
    test_question_ids = {
        item.get("question_id")
        for item in test_items
        if item.get("question_id")
    }
    overlap = dev_question_ids & test_question_ids

    print(f"[INFO] Overlapping question IDs: {len(overlap)}", flush=True)

    if overlap:
        raise RuntimeError(
            "Question-level leakage detected between development and test sets."
        )

    print("=" * 80, flush=True)
    print("[DONE] Full benchmark dataset build completed", flush=True)
    print("[DONE] Only verifier rows accepted by Step 06 were included", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
