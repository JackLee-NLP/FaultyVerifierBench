import random
from collections import defaultdict
from pathlib import Path

import yaml

from src.dataset.io_utils import (
    read_jsonl,
    write_jsonl,
)


def load_config(
    config_path="config/config.yaml",
):
    """
    Load the project configuration.
    """

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def get_split_paths(
    cfg,
    dev_output_path=None,
    test_output_path=None,
):
    """
    Resolve development and test output paths.
    """

    split_dir = cfg.get(
        "paths",
        {},
    ).get(
        "split_dir",
        "data/splits",
    )

    if dev_output_path is None:
        dev_output_path = (
            f"{split_dir}/dev.jsonl"
        )

    if test_output_path is None:
        test_output_path = (
            f"{split_dir}/test.jsonl"
        )

    return (
        dev_output_path,
        test_output_path,
    )


def group_items_by_question(
    items,
):
    """
    Group all examples derived from the same base question.

    Every row with the same question_id is assigned to the
    same split, preventing question-level leakage.

    If question_id is unavailable, id is used as the fallback.
    """

    groups = defaultdict(list)

    missing_group_id = 0

    for index, item in enumerate(items):
        group_id = item.get(
            "question_id"
        )

        if not group_id:
            group_id = item.get(
                "id"
            )

        if not group_id:
            missing_group_id += 1
            group_id = (
                f"missing_group_id_{index}"
            )

        groups[group_id].append(
            item
        )

    if missing_group_id:
        print(
            "[WARN] Rows without question_id or id: "
            f"{missing_group_id}",
            flush=True,
        )

    return groups


def split_grouped_items(
    items,
    dev_ratio=0.2,
    random_seed=42,
):
    """
    Split items while preserving question-level groups.

    The target development size is approximately:

        total rows × dev_ratio

    Because complete question groups cannot be divided, the final
    number of rows may differ slightly from the exact ratio.
    """

    if not items:
        return [], []

    if not 0.0 < dev_ratio < 1.0:
        raise ValueError(
            "dev_ratio must be between 0 and 1. "
            f"Received: {dev_ratio}"
        )

    groups = group_items_by_question(
        items
    )

    group_ids = list(
        groups.keys()
    )

    rng = random.Random(
        random_seed
    )

    rng.shuffle(
        group_ids
    )

    target_dev_rows = round(
        len(items) * dev_ratio
    )

    dev_items = []
    test_items = []

    dev_group_ids = set()
    test_group_ids = set()

    for group_id in group_ids:
        group_rows = groups[
            group_id
        ]

        current_dev_distance = abs(
            len(dev_items)
            - target_dev_rows
        )

        possible_dev_distance = abs(
            (
                len(dev_items)
                + len(group_rows)
            )
            - target_dev_rows
        )

        should_add_to_dev = (
            len(dev_items)
            < target_dev_rows
            and possible_dev_distance
            <= current_dev_distance
        )

        if should_add_to_dev:
            dev_items.extend(
                group_rows
            )

            dev_group_ids.add(
                group_id
            )

        else:
            test_items.extend(
                group_rows
            )

            test_group_ids.add(
                group_id
            )

    # If the greedy assignment left dev empty,
    # move one complete group from test to dev.
    if not dev_items and group_ids:
        first_group_id = group_ids[0]
        first_group = groups[
            first_group_id
        ]

        first_group_ids = {
            id(item)
            for item in first_group
        }

        test_items = [
            item
            for item in test_items
            if id(item)
            not in first_group_ids
        ]

        dev_items.extend(
            first_group
        )

        test_group_ids.discard(
            first_group_id
        )

        dev_group_ids.add(
            first_group_id
        )

    overlap = (
        dev_group_ids
        & test_group_ids
    )

    if overlap:
        raise RuntimeError(
            "Question-level data leakage detected. "
            f"Overlapping group count: {len(overlap)}"
        )

    if (
        len(dev_items)
        + len(test_items)
        != len(items)
    ):
        raise RuntimeError(
            "Split row-count mismatch. "
            f"Input={len(items)}, "
            f"dev={len(dev_items)}, "
            f"test={len(test_items)}"
        )

    return (
        dev_items,
        test_items,
    )


def split_dataset(
    items=None,
    input_path=None,
    config_path="config/config.yaml",
    dev_output_path=None,
    test_output_path=None,
):
    """
    Split the full benchmark into development and test sets.

    Priority for input data:

    1. items passed directly by merge_dataset()
    2. input_path passed explicitly
    3. paths.processed_data_dir/dataset_full.jsonl

    Passing items directly is preferred because it guarantees
    that the newly merged full dataset is being split.
    """

    cfg = load_config(
        config_path
    )

    if items is None:
        if input_path is None:
            processed_dir = cfg.get(
                "paths",
                {},
            ).get(
                "processed_data_dir",
                "data/processed",
            )

            input_path = (
                f"{processed_dir}/"
                "dataset_full.jsonl"
            )

        print(
            "[INFO] Loading split input from: "
            f"{input_path}",
            flush=True,
        )

        items = read_jsonl(
            input_path
        )

    else:
        print(
            "[INFO] Using merged items passed "
            "directly to split_dataset().",
            flush=True,
        )

    if not items:
        raise ValueError(
            "No dataset items were provided for splitting."
        )

    evaluation_cfg = cfg.get(
        "evaluation",
        {},
    )

    dev_ratio = evaluation_cfg.get(
        "dev_ratio",
        0.2,
    )

    random_seed = cfg.get(
        "random_seed",
        42,
    )

    (
        dev_output_path,
        test_output_path,
    ) = get_split_paths(
        cfg=cfg,
        dev_output_path=dev_output_path,
        test_output_path=test_output_path,
    )

    print(
        "=" * 80,
        flush=True,
    )
    print(
        "[SPLIT] Grouped full-dataset split",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )
    print(
        f"[INFO] Input rows: {len(items)}",
        flush=True,
    )
    print(
        f"[INFO] Development ratio: {dev_ratio}",
        flush=True,
    )
    print(
        f"[INFO] Test ratio: {1.0 - dev_ratio}",
        flush=True,
    )
    print(
        f"[INFO] Random seed: {random_seed}",
        flush=True,
    )
    print(
        "[INFO] Grouping key: question_id",
        flush=True,
    )

    dev_items, test_items = (
        split_grouped_items(
            items=items,
            dev_ratio=dev_ratio,
            random_seed=random_seed,
        )
    )

    Path(
        dev_output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        test_output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        dev_items,
        dev_output_path,
    )

    write_jsonl(
        test_items,
        test_output_path,
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

    overlap = (
        dev_question_ids
        & test_question_ids
    )

    print(
        f"[INFO] Dev rows: {len(dev_items)}",
        flush=True,
    )
    print(
        f"[INFO] Test rows: {len(test_items)}",
        flush=True,
    )
    print(
        "[INFO] Total output rows: "
        f"{len(dev_items) + len(test_items)}",
        flush=True,
    )
    print(
        "[INFO] Dev unique question IDs: "
        f"{len(dev_question_ids)}",
        flush=True,
    )
    print(
        "[INFO] Test unique question IDs: "
        f"{len(test_question_ids)}",
        flush=True,
    )
    print(
        "[INFO] Overlapping question IDs: "
        f"{len(overlap)}",
        flush=True,
    )
    print(
        f"[DONE] Dev saved to: "
        f"{dev_output_path}",
        flush=True,
    )
    print(
        f"[DONE] Test saved to: "
        f"{test_output_path}",
        flush=True,
    )

    if overlap:
        raise RuntimeError(
            "Question leakage detected after writing splits."
        )

    return (
        dev_items,
        test_items,
    )