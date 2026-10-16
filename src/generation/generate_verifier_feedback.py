import hashlib
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from prompts.verifier_generation import (
    build_correct_verifier_prompt,
    build_fallacy_verifier_prompt,
    build_hallucination_verifier_prompt,
)
from src.api.client_factory import get_client
from src.dataset.io_utils import (
    append_jsonl,
    read_jsonl,
    write_jsonl,
)
from src.utils.json_utils import safe_json_loads


VERIFIER_MODES = [
    "correct",
    "fallacy",
    "hallucination",
]


# Each worker thread receives its own model-client cache.
# This avoids sharing one mutable API client across threads.
_THREAD_LOCAL = threading.local()


def load_config(
    config_path="config/config.yaml",
):
    """
    Load the main project configuration.
    """

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def load_models(
    models_path="config/models.yaml",
):
    """
    Load API model definitions.
    """

    with open(
        models_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    return data["models"]


def load_fallacies(
    fallacies_path="config/fallacies.yaml",
):
    """
    Load the verifier fallacy taxonomy.
    """

    with open(
        fallacies_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    return data["fallacies"]


def load_hallucinations(
    hallucinations_path="config/hallucinations.yaml",
):
    """
    Load the verifier hallucination taxonomy.
    """

    with open(
        hallucinations_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    return data["hallucinations"]


def get_solver_path(
    cfg,
    solver_path=None,
):
    """
    Resolve the validated solver input path.
    """

    if solver_path is not None:
        return solver_path

    processed_dir = cfg.get(
        "paths",
        {},
    ).get(
        "processed_data_dir",
        "data/processed",
    )

    return (
        f"{processed_dir}/"
        "solver_validated.jsonl"
    )


def get_output_path(
    cfg,
    output_path=None,
):
    """
    Resolve the raw verifier output path.
    """

    if output_path is not None:
        return output_path

    raw_dir = cfg.get(
        "paths",
        {},
    ).get(
        "raw_data_dir",
        "data/raw",
    )

    return f"{raw_dir}/verifier_raw.jsonl"


def get_missing_output_path(
    cfg,
):
    """
    Resolve the missing-verifier list path.
    """

    raw_dir = cfg.get(
        "paths",
        {},
    ).get(
        "raw_data_dir",
        "data/raw",
    )

    return (
        f"{raw_dir}/"
        "verifier_missing_list.jsonl"
    )


def get_verifier_generator_models(
    cfg,
    models,
):
    """
    Determine which models generate verifier feedback.

    Priority:
    1. generation.verifier_generator_models
    2. all model keys in models.yaml
    """

    selected_models = cfg.get(
        "generation",
        {},
    ).get(
        "verifier_generator_models"
    )

    if selected_models:
        missing_models = [
            model_key
            for model_key in selected_models
            if model_key not in models
        ]

        if missing_models:
            raise ValueError(
                "The following verifier-generator models "
                "are missing from models.yaml: "
                f"{missing_models}. "
                f"Available models: {list(models.keys())}"
            )

        return selected_models

    return list(models.keys())


def get_api_settings(
    cfg,
):
    """
    Return the common API timeout and retry configuration.
    """

    api_cfg = cfg.get(
        "api",
        {},
    )

    generation_cfg = cfg.get(
        "generation",
        {},
    )

    return {
        "connect_timeout": api_cfg.get(
            "connect_timeout_seconds",
            20,
        ),
        "read_timeout": api_cfg.get(
            "read_timeout_seconds",
            240,
        ),
        "max_retries": generation_cfg.get(
            "max_retries",
            3,
        ),
        "retry_base_delay": generation_cfg.get(
            "retry_base_delay_seconds",
            5,
        ),
    }


def get_parallel_settings(
    cfg,
    max_workers=None,
    max_concurrent_per_model=None,
):
    """
    Resolve Step 05 concurrency settings.

    Explicit function/CLI arguments override config.yaml.
    """

    generation_cfg = cfg.get(
        "generation",
        {},
    )

    if max_workers is None:
        max_workers = generation_cfg.get(
            "verifier_max_workers",
            3,
        )

    if max_concurrent_per_model is None:
        max_concurrent_per_model = generation_cfg.get(
            "verifier_max_concurrent_per_model",
            1,
        )

    max_workers = max(
        1,
        int(max_workers),
    )

    max_concurrent_per_model = max(
        1,
        int(max_concurrent_per_model),
    )

    return (
        max_workers,
        max_concurrent_per_model,
    )


def create_model_client(
    cfg,
    model_cfg,
):
    """
    Create one API client using common configuration.
    """

    generation_cfg = cfg.get(
        "generation",
        {},
    )

    temperature = generation_cfg.get(
        "temperature",
        0.7,
    )

    api_settings = get_api_settings(
        cfg
    )

    return get_client(
        provider=model_cfg["provider"],
        model_name=model_cfg["model_name"],
        temperature=temperature,
        **api_settings,
    )


def get_thread_client(
    cfg,
    model_key,
    model_cfg,
):
    """
    Return a model client owned by the current worker thread.

    Each thread has an independent cache. Therefore, one API-client
    instance is never concurrently used by multiple worker threads.
    """

    client_cache = getattr(
        _THREAD_LOCAL,
        "client_cache",
        None,
    )

    if client_cache is None:
        client_cache = {}
        _THREAD_LOCAL.client_cache = (
            client_cache
        )

    if model_key not in client_cache:
        client_cache[model_key] = (
            create_model_client(
                cfg=cfg,
                model_cfg=model_cfg,
            )
        )

    return client_cache[model_key]


def make_verifier_pair_key(
    solver_id,
    verifier_mode,
    verifier_model,
):
    """
    Build the unique verifier-generation key.
    """

    return (
        solver_id,
        verifier_mode,
        verifier_model,
    )


def get_existing_verifier_pairs(
    existing_items,
):
    """
    Collect completed solver/mode/model keys.
    """

    completed = set()

    for item in existing_items:
        solver_id = item.get(
            "solver_id"
        )

        verifier_mode = item.get(
            "verifier_generation_mode"
        )

        verifier_model = item.get(
            "verifier_generator"
        )

        if (
            solver_id
            and verifier_mode
            and verifier_model
        ):
            completed.add(
                make_verifier_pair_key(
                    solver_id=solver_id,
                    verifier_mode=verifier_mode,
                    verifier_model=verifier_model,
                )
            )

    return completed


def build_expected_verifier_pairs(
    solver_items,
    verifier_model_keys,
):
    """
    Build all expected:

        solver × verifier model × verifier mode

    combinations.
    """

    expected = []

    for solver in solver_items:
        solver_id = solver.get(
            "id"
        )

        if not solver_id:
            print(
                "Solver item has no id. "
                "Skipping it in the expected verifier plan.",
                flush=True,
            )
            continue

        for verifier_model in verifier_model_keys:
            for verifier_mode in VERIFIER_MODES:
                expected.append(
                    {
                        "solver_id": solver_id,
                        "question_id": solver.get(
                            "question_id"
                        ),
                        "verifier_generation_mode": (
                            verifier_mode
                        ),
                        "verifier_generator": (
                            verifier_model
                        ),
                        "solver_correctness": solver.get(
                            "solver_correctness",
                            "",
                        ),
                        "question": solver.get(
                            "question",
                            "",
                        ),
                        "solver_response": solver.get(
                            "solver_response",
                            "",
                        ),
                    }
                )

    return expected


def _stable_random_choice(
    values,
    random_seed,
    solver_id,
    verifier_model_key,
    mode,
):
    """
    Select a fault deterministically.

    Parallel execution changes task completion order. Using the global
    random module would therefore make selected faults depend on thread
    scheduling. This function derives a stable local random seed from
    the generation key.
    """

    if not values:
        raise ValueError(
            f"No values are configured for mode={mode}."
        )

    seed_text = (
        f"{random_seed}|"
        f"{solver_id}|"
        f"{verifier_model_key}|"
        f"{mode}"
    )

    digest = hashlib.sha256(
        seed_text.encode("utf-8")
    ).hexdigest()

    index = int(
        digest[:16],
        16,
    ) % len(values)

    return values[index]


def _build_prompt_by_mode(
    item,
    mode,
    fallacies,
    hallucinations,
    random_seed,
    verifier_model_key,
):
    """
    Build one verifier-generation prompt.
    """

    solver_id = item.get(
        "id"
    )

    if mode == "correct":
        system_prompt, user_prompt = (
            build_correct_verifier_prompt(
                item["question"],
                item["gold_answer"],
                item["solver_response"],
                item["solver_correctness"],
            )
        )

        selected_fault = "none"

    elif mode == "fallacy":
        selected_fault = (
            _stable_random_choice(
                values=fallacies,
                random_seed=random_seed,
                solver_id=solver_id,
                verifier_model_key=(
                    verifier_model_key
                ),
                mode=mode,
            )
        )

        system_prompt, user_prompt = (
            build_fallacy_verifier_prompt(
                item["question"],
                item["gold_answer"],
                item["solver_response"],
                selected_fault,
            )
        )

    elif mode == "hallucination":
        selected_fault = (
            _stable_random_choice(
                values=hallucinations,
                random_seed=random_seed,
                solver_id=solver_id,
                verifier_model_key=(
                    verifier_model_key
                ),
                mode=mode,
            )
        )

        system_prompt, user_prompt = (
            build_hallucination_verifier_prompt(
                item["question"],
                item["gold_answer"],
                item["solver_response"],
                selected_fault,
            )
        )

    else:
        raise ValueError(
            f"Unknown verifier mode: {mode}"
        )

    return (
        system_prompt,
        user_prompt,
        selected_fault,
    )


def _safe_parse_raw_output(
    raw,
    solver_id,
    mode,
    verifier_model,
):
    """
    Parse one verifier response into a dictionary.
    """

    if raw is None:
        print(
            "API returned None | "
            f"solver_id={solver_id}, "
            f"mode={mode}, "
            f"verifier_model={verifier_model}",
            flush=True,
        )
        return None

    if not isinstance(
        raw,
        str,
    ):
        print(
            "API returned non-string | "
            f"type={type(raw)} | "
            f"solver_id={solver_id}, "
            f"mode={mode}, "
            f"verifier_model={verifier_model}",
            flush=True,
        )
        return None

    if not raw.strip():
        print(
            "API returned empty text | "
            f"solver_id={solver_id}, "
            f"mode={mode}, "
            f"verifier_model={verifier_model}",
            flush=True,
        )
        return None

    print(
        "Raw verifier output | "
        f"solver_id={solver_id} | "
        f"model={verifier_model} | "
        f"mode={mode} | "
        f"preview={raw[:300]}",
        flush=True,
    )

    parsed = safe_json_loads(
        raw
    )

    if parsed is None:
        print(
            "Could not parse verifier JSON | "
            f"solver_id={solver_id}, "
            f"mode={mode}, "
            f"verifier_model={verifier_model}",
            flush=True,
        )
        return None

    if isinstance(
        parsed,
        list,
    ):
        if not parsed:
            print(
                "Parsed verifier output "
                "is an empty list.",
                flush=True,
            )
            return None

        parsed = parsed[0]

    if not isinstance(
        parsed,
        dict,
    ):
        print(
            "Parsed verifier output "
            "is not a dictionary | "
            f"type={type(parsed)}",
            flush=True,
        )
        return None

    return parsed


def _build_output_item(
    item,
    parsed,
    mode,
    selected_fault,
    verifier_model_key,
    verifier_model_cfg,
):
    """
    Merge solver data, verifier output, and generation metadata.
    """

    solver_id = item.get(
        "id"
    )

    question_id = item.get(
        "question_id",
        solver_id,
    )

    verifier_id = (
        f"v_{uuid.uuid4().hex[:10]}"
    )

    output_item = {
        **item,
        **parsed,
        "id": verifier_id,
        "solver_id": solver_id,
        "question_id": question_id,
        "verifier_generator": (
            verifier_model_key
        ),
        "verifier_source_model_name": (
            verifier_model_cfg["model_name"]
        ),
        "verifier_generation_mode": mode,
        "selected_fault_type": selected_fault,
    }

    if mode == "correct":
        output_item[
            "verifier_correctness"
        ] = output_item.get(
            "verifier_correctness",
            "correct",
        )

        output_item[
            "fault_category"
        ] = output_item.get(
            "fault_category",
            [],
        )

        output_item[
            "fault_type"
        ] = output_item.get(
            "fault_type",
            [],
        )

        output_item[
            "hallucination_type"
        ] = output_item.get(
            "hallucination_type",
            [],
        )

    elif mode == "fallacy":
        output_item[
            "verifier_correctness"
        ] = output_item.get(
            "verifier_correctness",
            "faulty",
        )

        output_item[
            "fault_category"
        ] = output_item.get(
            "fault_category",
            ["fallacy"],
        )

        output_item[
            "fault_type"
        ] = output_item.get(
            "fault_type",
            [selected_fault],
        )

        output_item[
            "hallucination_type"
        ] = output_item.get(
            "hallucination_type",
            [],
        )

    elif mode == "hallucination":
        output_item[
            "verifier_correctness"
        ] = output_item.get(
            "verifier_correctness",
            "faulty",
        )

        output_item[
            "fault_category"
        ] = output_item.get(
            "fault_category",
            ["hallucination"],
        )

        output_item[
            "fault_type"
        ] = output_item.get(
            "fault_type",
            [],
        )

        output_item[
            "hallucination_type"
        ] = output_item.get(
            "hallucination_type",
            [selected_fault],
        )

    output_item[
        "expected_final_answer"
    ] = output_item.get(
        "expected_final_answer",
        item.get(
            "gold_answer",
            "",
        ),
    )

    return output_item


def generate_one_verifier_item(
    item,
    mode,
    verifier_model_key,
    verifier_model_cfg,
    client,
    fallacies,
    hallucinations,
    random_seed,
):
    """
    Generate one verifier item.

    Important:
    This worker function does not write to JSONL. It returns the
    generated item to the main thread, which performs the file write.
    """

    solver_id = item.get(
        "id"
    )

    print(
        "Generating verifier feedback | "
        f"solver_id={solver_id} | "
        f"mode={mode} | "
        f"verifier_model={verifier_model_key}",
        flush=True,
    )

    try:
        (
            system_prompt,
            user_prompt,
            selected_fault,
        ) = _build_prompt_by_mode(
            item=item,
            mode=mode,
            fallacies=fallacies,
            hallucinations=hallucinations,
            random_seed=random_seed,
            verifier_model_key=(
                verifier_model_key
            ),
        )

    except Exception as error:
        print(
            "Could not build verifier prompt | "
            f"solver_id={solver_id} | "
            f"model={verifier_model_key} | "
            f"mode={mode} | "
            f"error={error}",
            flush=True,
        )
        return None

    start_time = time.time()

    try:
        raw = client.generate(
            system_prompt,
            user_prompt,
        )

    except Exception as error:
        elapsed = (
            time.time() - start_time
        )

        print(
            "Verifier API call failed | "
            f"solver_id={solver_id} | "
            f"model={verifier_model_key} | "
            f"mode={mode} | "
            f"elapsed={elapsed:.2f}s | "
            f"error_type="
            f"{type(error).__name__} | "
            f"error={error}",
            flush=True,
        )
        return None

    elapsed = (
        time.time() - start_time
    )

    print(
        "Verifier API call finished | "
        f"solver_id={solver_id} | "
        f"model={verifier_model_key} | "
        f"mode={mode} | "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )

    parsed = _safe_parse_raw_output(
        raw=raw,
        solver_id=solver_id,
        mode=mode,
        verifier_model=verifier_model_key,
    )

    if parsed is None:
        return None

    return _build_output_item(
        item=item,
        parsed=parsed,
        mode=mode,
        selected_fault=selected_fault,
        verifier_model_key=(
            verifier_model_key
        ),
        verifier_model_cfg=(
            verifier_model_cfg
        ),
    )


def check_missing_verifier_outputs(
    solver_path=None,
    verifier_path=None,
    models_path="config/models.yaml",
    config_path="config/config.yaml",
    missing_output_path=None,
):
    """
    Check missing verifier outputs without calling any API.
    """

    cfg = load_config(
        config_path
    )

    models = load_models(
        models_path
    )

    solver_path = get_solver_path(
        cfg=cfg,
        solver_path=solver_path,
    )

    verifier_path = get_output_path(
        cfg=cfg,
        output_path=verifier_path,
    )

    if missing_output_path is None:
        missing_output_path = (
            get_missing_output_path(cfg)
        )

    solver_items = read_jsonl(
        solver_path
    )

    verifier_items = read_jsonl(
        verifier_path
    )

    verifier_model_keys = (
        get_verifier_generator_models(
            cfg=cfg,
            models=models,
        )
    )

    completed = get_existing_verifier_pairs(
        verifier_items
    )

    expected = build_expected_verifier_pairs(
        solver_items=solver_items,
        verifier_model_keys=(
            verifier_model_keys
        ),
    )

    missing = []

    for expected_item in expected:
        key = make_verifier_pair_key(
            solver_id=expected_item[
                "solver_id"
            ],
            verifier_mode=expected_item[
                "verifier_generation_mode"
            ],
            verifier_model=expected_item[
                "verifier_generator"
            ],
        )

        if key not in completed:
            missing.append(
                expected_item
            )

    pair_counts = Counter(
        make_verifier_pair_key(
            solver_id=item.get(
                "solver_id"
            ),
            verifier_mode=item.get(
                "verifier_generation_mode"
            ),
            verifier_model=item.get(
                "verifier_generator"
            ),
        )
        for item in verifier_items
        if (
            item.get("solver_id")
            and item.get(
                "verifier_generation_mode"
            )
            and item.get(
                "verifier_generator"
            )
        )
    )

    duplicates = [
        {
            "solver_id": key[0],
            "verifier_generation_mode": key[1],
            "verifier_generator": key[2],
            "count": count,
        }
        for key, count in pair_counts.items()
        if count > 1
    ]

    missing_by_model = Counter(
        item["verifier_generator"]
        for item in missing
    )

    missing_by_mode = Counter(
        item["verifier_generation_mode"]
        for item in missing
    )

    print(
        "=" * 80,
        flush=True,
    )
    print(
        "Verifier Missing Output Check",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )
    print(
        f"Solver path: {solver_path}",
        flush=True,
    )
    print(
        f"Verifier path: {verifier_path}",
        flush=True,
    )
    print(
        f"Solver rows: "
        f"{len(solver_items)}",
        flush=True,
    )
    print(
        f"Existing verifier rows: "
        f"{len(verifier_items)}",
        flush=True,
    )
    print(
        f"Verifier models: "
        f"{verifier_model_keys}",
        flush=True,
    )
    print(
        f"Verifier modes: "
        f"{VERIFIER_MODES}",
        flush=True,
    )
    print(
        f"Expected verifier rows: "
        f"{len(expected)}",
        flush=True,
    )
    print(
        "Completed unique "
        "solver/mode/model keys: "
        f"{len(completed)}",
        flush=True,
    )
    print(
        f"Missing verifier rows: "
        f"{len(missing)}",
        flush=True,
    )
    print(
        f"Duplicate keys: "
        f"{len(duplicates)}",
        flush=True,
    )

    print(
        "\nMissing by model:",
        flush=True,
    )

    for model_key in verifier_model_keys:
        print(
            f"  {model_key}: "
            f"{missing_by_model[model_key]}",
            flush=True,
        )

    print(
        "Missing by mode:",
        flush=True,
    )

    for mode in VERIFIER_MODES:
        print(
            f"  {mode}: "
            f"{missing_by_mode[mode]}",
            flush=True,
        )

    write_jsonl(
        missing,
        missing_output_path,
    )

    print(
        "\nSaved missing verifier "
        f"list to: {missing_output_path}",
        flush=True,
    )

    return missing


def _run_verifier_tasks_concurrently(
    tasks,
    solver_by_id,
    cfg,
    models,
    fallacies,
    hallucinations,
    output_path,
    max_workers,
    max_concurrent_per_model,
):
    """
    Run verifier-generation tasks concurrently.

    API calls happen in worker threads.
    JSONL writing happens only in this main thread.
    """

    if not tasks:
        return []

    random_seed = cfg.get(
        "random_seed",
        42,
    )

    model_semaphores = {
        model_key: threading.Semaphore(
            max_concurrent_per_model
        )
        for model_key in {
            task["verifier_generator"]
            for task in tasks
        }
    }

    completed_count = 0
    success_count = 0
    failure_count = 0
    new_items = []

    total_tasks = len(
        tasks
    )

    print(
        "=" * 80,
        flush=True,
    )
    print(
        "Starting verifier generation",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )
    print(
        f"Total tasks: {total_tasks}",
        flush=True,
    )
    print(
        f"Total workers: {max_workers}",
        flush=True,
    )
    print(
        "Maximum concurrent requests "
        f"per model: {max_concurrent_per_model}",
        flush=True,
    )
    print(
        "JSONL writes are restricted "
        "to the main thread.",
        flush=True,
    )

    def run_task(
        task,
    ):
        solver_id = task[
            "solver_id"
        ]

        mode = task[
            "verifier_generation_mode"
        ]

        verifier_model_key = task[
            "verifier_generator"
        ]

        item = solver_by_id.get(
            solver_id
        )

        if item is None:
            print(
                "Solver item not found | "
                f"solver_id={solver_id}",
                flush=True,
            )
            return None

        model_cfg = models.get(
            verifier_model_key
        )

        if model_cfg is None:
            print(
                "Verifier model not found | "
                f"model={verifier_model_key}",
                flush=True,
            )
            return None

        semaphore = model_semaphores[
            verifier_model_key
        ]

        with semaphore:
            client = get_thread_client(
                cfg=cfg,
                model_key=(
                    verifier_model_key
                ),
                model_cfg=model_cfg,
            )

            return generate_one_verifier_item(
                item=item,
                mode=mode,
                verifier_model_key=(
                    verifier_model_key
                ),
                verifier_model_cfg=(
                    model_cfg
                ),
                client=client,
                fallacies=fallacies,
                hallucinations=hallucinations,
                random_seed=random_seed,
            )

    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="verifier",
    ) as executor:
        future_to_task = {
            executor.submit(
                run_task,
                task,
            ): task
            for task in tasks
        }

        for future in as_completed(
            future_to_task
        ):
            task = future_to_task[
                future
            ]

            completed_count += 1

            try:
                output_item = (
                    future.result()
                )

            except Exception as error:
                output_item = None

                print(
                    "Unexpected worker failure | "
                    f"solver_id="
                    f"{task['solver_id']} | "
                    f"model="
                    f"{task['verifier_generator']} | "
                    f"mode="
                    f"{task['verifier_generation_mode']} | "
                    f"error_type="
                    f"{type(error).__name__} | "
                    f"error={error}",
                    flush=True,
                )

            if output_item is not None:
                # Only the main thread writes to JSONL.
                append_jsonl(
                    output_item,
                    output_path,
                )

                new_items.append(
                    output_item
                )

                success_count += 1

                print(
                    "[SAVED] "
                    f"{output_item['id']} | "
                    f"solver_id="
                    f"{output_item['solver_id']} | "
                    f"model="
                    f"{output_item['verifier_generator']} | "
                    f"mode="
                    f"{output_item['verifier_generation_mode']} | "
                    f"progress="
                    f"{completed_count}/{total_tasks}",
                    flush=True,
                )

            else:
                failure_count += 1

            print(
                "[PROGRESS] "
                f"completed={completed_count}/"
                f"{total_tasks} | "
                f"saved={success_count} | "
                f"failed={failure_count} | "
                f"in_flight="
                f"{total_tasks - completed_count}",
                flush=True,
            )

    print(
        "=" * 80,
        flush=True,
    )
    print(
        "Concurrent verifier generation finished",
        flush=True,
    )
    print(
        f"Tasks completed: "
        f"{completed_count}/{total_tasks}",
        flush=True,
    )
    print(
        f"Newly saved: {success_count}",
        flush=True,
    )
    print(
        f"Failed or invalid: "
        f"{failure_count}",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    return new_items


def generate_missing_verifier_feedback(
    solver_path=None,
    output_path=None,
    models_path="config/models.yaml",
    config_path="config/config.yaml",
    fallacies_path="config/fallacies.yaml",
    hallucinations_path="config/hallucinations.yaml",
    max_workers=None,
    max_concurrent_per_model=None,
):
    """
    Generate only missing verifier outputs concurrently.
    """

    cfg = load_config(
        config_path
    )

    models = load_models(
        models_path
    )

    fallacies = load_fallacies(
        fallacies_path
    )

    hallucinations = load_hallucinations(
        hallucinations_path
    )

    solver_path = get_solver_path(
        cfg=cfg,
        solver_path=solver_path,
    )

    output_path = get_output_path(
        cfg=cfg,
        output_path=output_path,
    )

    missing_items = (
        check_missing_verifier_outputs(
            solver_path=solver_path,
            verifier_path=output_path,
            models_path=models_path,
            config_path=config_path,
        )
    )

    if not missing_items:
        print(
            "Nothing missing. "
            "No verifier generation needed.",
            flush=True,
        )
        return []

    solver_items = read_jsonl(
        solver_path
    )

    solver_by_id = {
        item.get("id"): item
        for item in solver_items
        if item.get("id")
    }

    (
        max_workers,
        max_concurrent_per_model,
    ) = get_parallel_settings(
        cfg=cfg,
        max_workers=max_workers,
        max_concurrent_per_model=(
            max_concurrent_per_model
        ),
    )

    return _run_verifier_tasks_concurrently(
        tasks=missing_items,
        solver_by_id=solver_by_id,
        cfg=cfg,
        models=models,
        fallacies=fallacies,
        hallucinations=hallucinations,
        output_path=output_path,
        max_workers=max_workers,
        max_concurrent_per_model=(
            max_concurrent_per_model
        ),
    )


def full_generate_verifier_feedback(
    solver_path=None,
    output_path=None,
    models_path="config/models.yaml",
    config_path="config/config.yaml",
    fallacies_path="config/fallacies.yaml",
    hallucinations_path="config/hallucinations.yaml",
    max_workers=None,
    max_concurrent_per_model=None,
):
    """
    Clear the verifier file and regenerate all expected outputs.

    This runs only when explicitly requested with --force-full.
    """

    cfg = load_config(
        config_path
    )

    output_path = get_output_path(
        cfg=cfg,
        output_path=output_path,
    )

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[FULL] Clearing existing verifier output.",
        flush=True,
    )

    write_jsonl(
        [],
        output_path,
    )

    return generate_missing_verifier_feedback(
        solver_path=solver_path,
        output_path=output_path,
        models_path=models_path,
        config_path=config_path,
        fallacies_path=fallacies_path,
        hallucinations_path=hallucinations_path,
        max_workers=max_workers,
        max_concurrent_per_model=(
            max_concurrent_per_model
        ),
    )


def generate_verifier_feedback(
    solver_path=None,
    output_path=None,
    models_path="config/models.yaml",
    config_path="config/config.yaml",
    fallacies_path="config/fallacies.yaml",
    hallucinations_path="config/hallucinations.yaml",
    force_full_regenerate=False,
    max_workers=None,
    max_concurrent_per_model=None,
):
    """
    Safe concurrent multi-model verifier-generation entry point.

    Normal behavior:
    - preserve existing output;
    - calculate missing solver/model/mode combinations;
    - generate only missing combinations concurrently.

    Full regeneration:
    - only when force_full_regenerate=True.
    """

    cfg = load_config(
        config_path
    )

    solver_path = get_solver_path(
        cfg=cfg,
        solver_path=solver_path,
    )

    output_path = get_output_path(
        cfg=cfg,
        output_path=output_path,
    )

    existing_items = read_jsonl(
        output_path
    )

    (
        resolved_workers,
        resolved_per_model,
    ) = get_parallel_settings(
        cfg=cfg,
        max_workers=max_workers,
        max_concurrent_per_model=(
            max_concurrent_per_model
        ),
    )

    print(
        "=" * 80,
        flush=True,
    )
    print(
        "Concurrent multi-model "
        "verifier generation",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )
    print(
        f"Solver path: {solver_path}",
        flush=True,
    )
    print(
        f"Output path: {output_path}",
        flush=True,
    )
    print(
        f"Existing verifier rows: "
        f"{len(existing_items)}",
        flush=True,
    )
    print(
        f"Workers: "
        f"{resolved_workers}",
        flush=True,
    )
    print(
        "Concurrent calls per model: "
        f"{resolved_per_model}",
        flush=True,
    )
    print(
        "Force full regenerate: "
        f"{force_full_regenerate}",
        flush=True,
    )

    if force_full_regenerate:
        return full_generate_verifier_feedback(
            solver_path=solver_path,
            output_path=output_path,
            models_path=models_path,
            config_path=config_path,
            fallacies_path=fallacies_path,
            hallucinations_path=(
                hallucinations_path
            ),
            max_workers=resolved_workers,
            max_concurrent_per_model=(
                resolved_per_model
            ),
        )

    # Normal mode always checks missing output.
    # It never clears the existing JSONL automatically.
    return generate_missing_verifier_feedback(
        solver_path=solver_path,
        output_path=output_path,
        models_path=models_path,
        config_path=config_path,
        fallacies_path=fallacies_path,
        hallucinations_path=(
            hallucinations_path
        ),
        max_workers=resolved_workers,
        max_concurrent_per_model=(
            resolved_per_model
        ),
    )


def auto_generate_verifier_feedback(
    solver_path=None,
    output_path=None,
    models_path="config/models.yaml",
    config_path="config/config.yaml",
    fallacies_path="config/fallacies.yaml",
    hallucinations_path="config/hallucinations.yaml",
    force_full_regenerate=False,
    max_workers=None,
    max_concurrent_per_model=None,
):
    """
    Backward-compatible wrapper.
    """

    return generate_verifier_feedback(
        solver_path=solver_path,
        output_path=output_path,
        models_path=models_path,
        config_path=config_path,
        fallacies_path=fallacies_path,
        hallucinations_path=(
            hallucinations_path
        ),
        force_full_regenerate=(
            force_full_regenerate
        ),
        max_workers=max_workers,
        max_concurrent_per_model=(
            max_concurrent_per_model
        ),
    )