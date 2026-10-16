import time
import uuid
from collections import Counter
from pathlib import Path

import yaml

from prompts.solver_generation import (
    build_correct_solver_prompt,
    build_incorrect_solver_prompt,
)
from src.api.client_factory import get_client
from src.dataset.io_utils import (
    append_jsonl,
    read_jsonl,
    write_jsonl,
)
from src.utils.json_utils import safe_json_loads


SOLVER_MODES = [
    "correct",
    "incorrect",
]


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
    Load configured API models.
    """

    with open(
        models_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    return data["models"]


def get_solver_generator_models(
    cfg,
    models,
):
    """
    Determine which models generate solver responses.

    Priority:
    1. generation.solver_generator_models
    2. every model in models.yaml
    """

    selected_models = cfg.get(
        "generation",
        {},
    ).get(
        "solver_generator_models"
    )

    if selected_models:
        missing_models = [
            model_key
            for model_key in selected_models
            if model_key not in models
        ]

        if missing_models:
            raise ValueError(
                "The following solver generator models "
                "are not present in models.yaml: "
                f"{missing_models}. "
                f"Available models: {list(models.keys())}"
            )

        return selected_models

    return list(models.keys())


def get_question_path(
    cfg,
    question_path=None,
):
    """
    Resolve the validated-question input path.
    """

    if question_path is not None:
        return question_path

    processed_dir = cfg.get(
        "paths",
        {},
    ).get(
        "processed_data_dir",
        "data/processed",
    )

    return (
        f"{processed_dir}/"
        "questions_validated.jsonl"
    )


def get_output_path(
    cfg,
    output_path=None,
):
    """
    Resolve the raw solver output path.
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

    return f"{raw_dir}/solver_raw.jsonl"


def get_missing_output_path(
    cfg,
):
    """
    Resolve the solver missing-list path.
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
        "solver_missing_list.jsonl"
    )


def get_api_settings(
    cfg,
):
    """
    Return the common API timeout and retry policy.

    The same values are passed to every model client.
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


def create_model_client(
    cfg,
    model_cfg,
):
    """
    Create one API client using the common client interface.
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


def make_solver_key(
    question_id,
    solver_model,
    solver_mode,
):
    """
    Build the unique key for one expected solver output.
    """

    return (
        question_id,
        solver_model,
        solver_mode,
    )


def get_existing_solver_keys(
    existing_items,
):
    """
    Collect completed question/model/mode combinations.
    """

    completed = set()

    for item in existing_items:
        question_id = item.get(
            "question_id"
        )

        solver_model = item.get(
            "solver_generator"
        )

        solver_mode = item.get(
            "solver_generation_mode"
        )

        if (
            question_id
            and solver_model
            and solver_mode
        ):
            completed.add(
                make_solver_key(
                    question_id=question_id,
                    solver_model=solver_model,
                    solver_mode=solver_mode,
                )
            )

    return completed


def build_expected_solver_outputs(
    questions,
    solver_model_keys,
):
    """
    Build every expected:

        question × solver model × solver mode

    combination.
    """

    expected = []

    for question in questions:
        question_id = question.get(
            "id"
        )

        if not question_id:
            print(
                "Validated question has no id. "
                "Skipping it in the expected-output plan.",
                flush=True,
            )
            continue

        for solver_model in solver_model_keys:
            for solver_mode in SOLVER_MODES:
                expected.append(
                    {
                        "question_id": question_id,
                        "solver_generator": (
                            solver_model
                        ),
                        "solver_generation_mode": (
                            solver_mode
                        ),
                        "question": question.get(
                            "question",
                            "",
                        ),
                        "gold_answer": question.get(
                            "gold_answer",
                            "",
                        ),
                    }
                )

    return expected


def check_missing_solver_outputs(
    question_path=None,
    output_path=None,
    models_path="config/models.yaml",
    config_path="config/config.yaml",
    missing_output_path=None,
):
    """
    Check which solver outputs are missing.

    This function:
    - does not call any API;
    - does not change solver_raw.jsonl;
    - saves the missing list to JSONL;
    - checks duplicate question/model/mode keys.
    """

    cfg = load_config(
        config_path
    )

    models = load_models(
        models_path
    )

    question_path = get_question_path(
        cfg=cfg,
        question_path=question_path,
    )

    output_path = get_output_path(
        cfg=cfg,
        output_path=output_path,
    )

    if missing_output_path is None:
        missing_output_path = (
            get_missing_output_path(cfg)
        )

    questions = read_jsonl(
        question_path
    )

    existing_items = read_jsonl(
        output_path
    )

    solver_model_keys = (
        get_solver_generator_models(
            cfg=cfg,
            models=models,
        )
    )

    completed_keys = (
        get_existing_solver_keys(
            existing_items
        )
    )

    expected_outputs = (
        build_expected_solver_outputs(
            questions=questions,
            solver_model_keys=(
                solver_model_keys
            ),
        )
    )

    missing_outputs = []

    for expected in expected_outputs:
        key = make_solver_key(
            question_id=expected[
                "question_id"
            ],
            solver_model=expected[
                "solver_generator"
            ],
            solver_mode=expected[
                "solver_generation_mode"
            ],
        )

        if key not in completed_keys:
            missing_outputs.append(
                expected
            )

    print(
        "=" * 80,
        flush=True,
    )
    print(
        "Solver Missing Output Check",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )
    print(
        f"Question path: "
        f"{question_path}",
        flush=True,
    )
    print(
        f"Solver output path: "
        f"{output_path}",
        flush=True,
    )
    print(
        f"Validated questions: "
        f"{len(questions)}",
        flush=True,
    )
    print(
        f"Existing solver rows: "
        f"{len(existing_items)}",
        flush=True,
    )
    print(
        f"Solver models: "
        f"{solver_model_keys}",
        flush=True,
    )
    print(
        f"Solver modes: "
        f"{SOLVER_MODES}",
        flush=True,
    )
    print(
        f"Expected solver rows: "
        f"{len(expected_outputs)}",
        flush=True,
    )
    print(
        "Completed unique "
        "question/model/mode keys: "
        f"{len(completed_keys)}",
        flush=True,
    )
    print(
        f"Missing solver outputs: "
        f"{len(missing_outputs)}",
        flush=True,
    )

    pair_counts = Counter(
        make_solver_key(
            question_id=item.get(
                "question_id"
            ),
            solver_model=item.get(
                "solver_generator"
            ),
            solver_mode=item.get(
                "solver_generation_mode"
            ),
        )
        for item in existing_items
        if (
            item.get("question_id")
            and item.get("solver_generator")
            and item.get(
                "solver_generation_mode"
            )
        )
    )

    duplicates = [
        {
            "question_id": key[0],
            "solver_generator": key[1],
            "solver_generation_mode": key[2],
            "count": count,
        }
        for key, count in pair_counts.items()
        if count > 1
    ]

    print(
        "Duplicate "
        "question/model/mode keys: "
        f"{len(duplicates)}",
        flush=True,
    )

    missing_by_model = Counter(
        item["solver_generator"]
        for item in missing_outputs
    )

    missing_by_mode = Counter(
        item["solver_generation_mode"]
        for item in missing_outputs
    )

    print(
        "\n[SUMMARY] Missing by model:",
        flush=True,
    )

    for model_key in solver_model_keys:
        print(
            f"  {model_key}: "
            f"{missing_by_model[model_key]}",
            flush=True,
        )

    print(
        "[SUMMARY] Missing by mode:",
        flush=True,
    )

    for mode in SOLVER_MODES:
        print(
            f"  {mode}: "
            f"{missing_by_mode[mode]}",
            flush=True,
        )

    if duplicates:
        print(
            "\n" + "=" * 80,
            flush=True,
        )
        print(
            "Duplicate Solver Outputs",
            flush=True,
        )
        print(
            "=" * 80,
            flush=True,
        )

        for index, duplicate in enumerate(
            duplicates,
            start=1,
        ):
            print(
                f"duplicate {index} "
                f"question_id="
                f"{duplicate['question_id']} | "
                f"model="
                f"{duplicate['solver_generator']} | "
                f"mode="
                f"{duplicate['solver_generation_mode']} | "
                f"count={duplicate['count']}",
                flush=True,
            )

    write_jsonl(
        missing_outputs,
        missing_output_path,
    )

    print(
        "\nSaved missing solver "
        f"list to: {missing_output_path}",
        flush=True,
    )

    return missing_outputs


def build_solver_prompt(
    question,
    mode,
):
    """
    Construct the solver prompt for one mode.
    """

    if mode == "correct":
        return build_correct_solver_prompt(
            question["question"],
            question["gold_answer"],
            question.get(
                "gold_explanation",
                "",
            ),
        )

    if mode == "incorrect":
        return build_incorrect_solver_prompt(
            question["question"],
            question["gold_answer"],
        )

    raise ValueError(
        f"Unknown solver mode: {mode}"
    )


def parse_solver_output(
    raw,
    question_id,
    mode,
    solver_model,
):
    """
    Parse one solver response as a dictionary.
    """

    if raw is None:
        print(
            "API returned None | "
            f"question_id={question_id}, "
            f"model={solver_model}, "
            f"mode={mode}",
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
            f"question_id={question_id}, "
            f"model={solver_model}, "
            f"mode={mode}",
            flush=True,
        )
        return None

    if not raw.strip():
        print(
            "API returned empty text | "
            f"question_id={question_id}, "
            f"model={solver_model}, "
            f"mode={mode}",
            flush=True,
        )
        return None

    print(
        "Raw output first "
        f"300 chars:\n{raw[:300]}",
        flush=True,
    )

    parsed = safe_json_loads(
        raw
    )

    if parsed is None:
        print(
            "Could not parse solver JSON | "
            f"question_id={question_id}, "
            f"model={solver_model}, "
            f"mode={mode}",
            flush=True,
        )
        print(
            "Raw output preview:\n"
            f"{raw[:1000]}",
            flush=True,
        )
        return None

    if isinstance(
        parsed,
        list,
    ):
        if not parsed:
            print(
                "Parsed solver output "
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
            "Parsed solver output "
            "is not a dictionary | "
            f"type={type(parsed)}",
            flush=True,
        )
        return None

    return parsed


def generate_one_solver_output(
    question,
    mode,
    solver_model_key,
    model_cfg,
    client,
    output_path,
):
    """
    Generate and immediately save one solver response.
    """

    question_id = question.get(
        "id"
    )

    print(
        "[CALL] Generating solver response | "
        f"question_id={question_id} | "
        f"mode={mode} | "
        f"model={solver_model_key}",
        flush=True,
    )

    print(
        "[QUESTION] "
        f"{question.get('question', '')[:120]}",
        flush=True,
    )

    try:
        system_prompt, user_prompt = (
            build_solver_prompt(
                question=question,
                mode=mode,
            )
        )

    except Exception as error:
        print(
            "Could not build solver prompt | "
            f"question_id={question_id}, "
            f"model={solver_model_key}, "
            f"mode={mode}: {error}",
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
            "API call failed | "
            f"question_id={question_id}, "
            f"mode={mode}, "
            f"model={solver_model_key}, "
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
        "API call finished in "
        f"{elapsed:.2f}s",
        flush=True,
    )

    parsed = parse_solver_output(
        raw=raw,
        question_id=question_id,
        mode=mode,
        solver_model=solver_model_key,
    )

    if parsed is None:
        return None

    solver_id = (
        f"s_{uuid.uuid4().hex[:10]}"
    )

    output_item = {
        **question,
        **parsed,
        "id": solver_id,
        "question_id": question_id,
        "solver_generator": (
            solver_model_key
        ),
        "solver_source_model_name": (
            model_cfg["model_name"]
        ),
        "solver_generation_mode": mode,
    }

    output_item["solver_correctness"] = (
        output_item.get(
            "solver_correctness",
            mode,
        )
    )

    append_jsonl(
        output_item,
        output_path,
    )

    print(
        f"{solver_id} | "
        f"question_id={question_id} | "
        f"model={solver_model_key} | "
        f"mode={mode} | "
        f"solver_correctness="
        f"{output_item.get('solver_correctness')} | "
        f"final_answer="
        f"{str(output_item.get('solver_final_answer', ''))[:80]}",
        flush=True,
    )

    return output_item


def generate_missing_solver_responses(
    question_path=None,
    output_path=None,
    models_path="config/models.yaml",
    config_path="config/config.yaml",
):
    """
    Generate only missing question/model/mode outputs.

    Existing solver rows are never cleared.
    """

    cfg = load_config(
        config_path
    )

    models = load_models(
        models_path
    )

    question_path = get_question_path(
        cfg=cfg,
        question_path=question_path,
    )

    output_path = get_output_path(
        cfg=cfg,
        output_path=output_path,
    )

    missing_outputs = (
        check_missing_solver_outputs(
            question_path=question_path,
            output_path=output_path,
            models_path=models_path,
            config_path=config_path,
        )
    )

    if not missing_outputs:
        print(
            "No missing solver outputs.",
            flush=True,
        )
        return []

    questions = read_jsonl(
        question_path
    )

    questions_by_id = {
        question.get("id"): question
        for question in questions
        if question.get("id")
    }

    clients = {}
    new_items = []

    print(
        "=" * 80,
        flush=True,
    )
    print(
        "[MISSING] Generating missing solver responses",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    for index, missing in enumerate(
        missing_outputs,
        start=1,
    ):
        question_id = missing[
            "question_id"
        ]

        solver_model_key = missing[
            "solver_generator"
        ]

        mode = missing[
            "solver_generation_mode"
        ]

        question = questions_by_id.get(
            question_id
        )

        if question is None:
            print(
                "Missing question was not "
                "found in the validated question file | "
                f"question_id={question_id}",
                flush=True,
            )
            continue

        model_cfg = models.get(
            solver_model_key
        )

        if model_cfg is None:
            print(
                "Solver model was not found "
                "in models.yaml | "
                f"model={solver_model_key}",
                flush=True,
            )
            continue

        if solver_model_key not in clients:
            clients[solver_model_key] = (
                create_model_client(
                    cfg=cfg,
                    model_cfg=model_cfg,
                )
            )

        print(
            f"\n[MISSING CALL "
            f"{index}/{len(missing_outputs)}] "
            f"question_id={question_id} | "
            f"model={solver_model_key} | "
            f"mode={mode}",
            flush=True,
        )

        output_item = (
            generate_one_solver_output(
                question=question,
                mode=mode,
                solver_model_key=(
                    solver_model_key
                ),
                model_cfg=model_cfg,
                client=clients[
                    solver_model_key
                ],
                output_path=output_path,
            )
        )

        if output_item is not None:
            new_items.append(
                output_item
            )

    final_items = read_jsonl(
        output_path
    )

    print(
        "\nNewly generated solver "
        f"responses: {len(new_items)}",
        flush=True,
    )
    print(
        f"Total solver rows now: "
        f"{len(final_items)}",
        flush=True,
    )
    print(
        f"Saved to: {output_path}",
        flush=True,
    )

    return new_items


def full_generate_solver_responses(
    question_path=None,
    output_path=None,
    models_path="config/models.yaml",
    config_path="config/config.yaml",
):
    """
    Clear solver_raw.jsonl and generate every expected output.

    This function should only be called through --force-full.
    """

    cfg = load_config(
        config_path
    )

    output_path = get_output_path(
        cfg=cfg,
        output_path=output_path,
    )

    print(
        "[FULL] Clearing solver output file.",
        flush=True,
    )

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        [],
        output_path,
    )

    return generate_missing_solver_responses(
        question_path=question_path,
        output_path=output_path,
        models_path=models_path,
        config_path=config_path,
    )


def generate_solver_responses(
    question_path=None,
    output_path=None,
    models_path="config/models.yaml",
    config_path="config/config.yaml",
    force_full_regenerate=False,
):
    """
    Safe multi-model solver-generation entry point.

    Normal behavior:
    - preserve existing rows;
    - identify missing question/model/mode outputs;
    - generate only missing outputs.

    Full regeneration:
    - only when force_full_regenerate=True.
    """

    cfg = load_config(
        config_path
    )

    question_path = get_question_path(
        cfg=cfg,
        question_path=question_path,
    )

    output_path = get_output_path(
        cfg=cfg,
        output_path=output_path,
    )

    print(
        "=" * 80,
        flush=True,
    )
    print(
        "Multi-model solver generation",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )
    print(
        f"Question path: "
        f"{question_path}",
        flush=True,
    )
    print(
        f"Output path: "
        f"{output_path}",
        flush=True,
    )
    print(
        "Force full regenerate: "
        f"{force_full_regenerate}",
        flush=True,
    )

    existing_items = read_jsonl(
        output_path
    )

    print(
        "Existing solver rows: "
        f"{len(existing_items)}",
        flush=True,
    )

    if force_full_regenerate:
        print(
            "Full regeneration explicitly "
            "requested. Existing solver output "
            "will be cleared.",
            flush=True,
        )

        return full_generate_solver_responses(
            question_path=question_path,
            output_path=output_path,
            models_path=models_path,
            config_path=config_path,
        )

    print(
        "Checking all "
        "question/model/mode combinations. "
        "Only missing outputs will be generated.",
        flush=True,
    )

    return generate_missing_solver_responses(
        question_path=question_path,
        output_path=output_path,
        models_path=models_path,
        config_path=config_path,
    )