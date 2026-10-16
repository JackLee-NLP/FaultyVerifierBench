import time
import uuid
from collections import Counter

import yaml

from prompts.question_generation import (
    build_question_generation_prompt,
)
from src.api.client_factory import get_client
from src.dataset.io_utils import (
    append_jsonl,
    read_jsonl,
    write_jsonl,
)
from src.utils.json_utils import safe_json_loads


def load_config(
    config_path="config/config.yaml",
):
    """
    Load the main YAML configuration file.
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
    Load model definitions from models.yaml.
    """

    with open(
        models_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    return data["models"]


def load_domains(
    domains_path="config/domains.yaml",
):
    """
    Load question-generation domains.
    """

    with open(
        domains_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    return data["domains"]


def get_question_generator_models(
    cfg,
    models,
):
    """
    Determine which models generate questions.

    Priority:
    1. generation.question_generator_models
    2. all models defined in models.yaml
    """

    selected_models = cfg.get(
        "generation",
        {},
    ).get(
        "question_generator_models",
    )

    if selected_models:
        missing_models = [
            model_key
            for model_key in selected_models
            if model_key not in models
        ]

        if missing_models:
            raise ValueError(
                "The following question-generator models "
                "are missing from models.yaml: "
                f"{missing_models}. "
                f"Available models: {list(models.keys())}"
            )

        return selected_models

    return list(models.keys())


def get_output_path(
    cfg,
    output_path=None,
):
    """
    Determine the output JSONL path.
    """

    if output_path is not None:
        return output_path

    raw_data_dir = cfg.get(
        "paths",
        {},
    ).get(
        "raw_data_dir",
        "data/raw",
    )

    return f"{raw_data_dir}/questions_raw.jsonl"


def get_missing_batch_path(
    cfg,
):
    """
    Determine where the missing-batch plan is stored.
    """

    raw_data_dir = cfg.get(
        "paths",
        {},
    ).get(
        "raw_data_dir",
        "data/raw",
    )

    return (
        f"{raw_data_dir}/"
        "question_missing_batches.jsonl"
    )


def get_provider_api_settings(
    cfg,
    provider,
):
    """
    Get timeout and retry settings for one API provider.

    Provider-specific values override general API values.
    """

    api_cfg = cfg.get(
        "api",
        {},
    )

    provider_cfg = api_cfg.get(
        "provider_timeouts",
        {},
    ).get(
        provider,
        {},
    )

    connect_timeout = provider_cfg.get(
        "connect_timeout_seconds",
        api_cfg.get(
            "connect_timeout_seconds",
            20,
        ),
    )

    read_timeout = provider_cfg.get(
        "read_timeout_seconds",
        api_cfg.get(
            "read_timeout_seconds",
            180,
        ),
    )

    generation_cfg = cfg.get(
        "generation",
        {},
    )

    max_retries = generation_cfg.get(
        "max_retries",
        3,
    )

    retry_base_delay = generation_cfg.get(
        "retry_base_delay_seconds",
        5,
    )

    return {
        "connect_timeout": connect_timeout,
        "read_timeout": read_timeout,
        "max_retries": max_retries,
        "retry_base_delay": retry_base_delay,
    }


def create_model_client(
    cfg,
    model_cfg,
):
    """
    Create one provider client using configuration-driven settings.
    """

    temperature = cfg.get(
        "generation",
        {},
    ).get(
        "temperature",
        0.7,
    )

    provider = model_cfg["provider"]

    api_settings = get_provider_api_settings(
        cfg=cfg,
        provider=provider,
    )

    return get_client(
        provider=provider,
        model_name=model_cfg["model_name"],
        temperature=temperature,
        **api_settings,
    )


def make_generation_batch_id(
    model_key,
    domain,
    difficulty,
):
    """
    Create a stable ID for one generation batch.
    """

    return (
        f"{model_key}::"
        f"{domain}::"
        f"{difficulty}"
    )


def get_existing_batch_counts(
    existing_items,
):
    """
    Count existing questions by:

        question_generator
        + domain
        + difficulty
    """

    counts = Counter()

    for item in existing_items:
        model_key = item.get(
            "question_generator"
        )
        domain = item.get(
            "domain"
        )
        difficulty = item.get(
            "difficulty"
        )

        if (
            model_key
            and domain
            and difficulty
        ):
            batch_id = make_generation_batch_id(
                model_key=model_key,
                domain=domain,
                difficulty=difficulty,
            )

            counts[batch_id] += 1

    return counts


def get_existing_question_ids(
    existing_items,
):
    """
    Collect all existing question IDs.
    """

    return {
        item.get("id")
        for item in existing_items
        if item.get("id")
    }


def build_expected_batches(
    model_keys,
    domains,
    difficulties,
    n_per_batch,
):
    """
    Build all expected:

        model × domain × difficulty

    batches.
    """

    expected_batches = []

    for model_key in model_keys:
        for domain in domains:
            for difficulty in difficulties:
                batch_id = make_generation_batch_id(
                    model_key=model_key,
                    domain=domain,
                    difficulty=difficulty,
                )

                expected_batches.append(
                    {
                        "model_key": model_key,
                        "domain": domain,
                        "difficulty": difficulty,
                        "expected_n": n_per_batch,
                        "batch_id": batch_id,
                    }
                )

    return expected_batches


def check_missing_question_batches(
    config_path="config/config.yaml",
    models_path="config/models.yaml",
    domains_path="config/domains.yaml",
    output_path=None,
    missing_output_path=None,
):
    """
    Check which question-generation batches are missing
    or incomplete.

    This function:
    - does not call an API;
    - does not modify questions_raw.jsonl;
    - saves the missing-batch plan to JSONL.
    """

    cfg = load_config(
        config_path
    )
    models = load_models(
        models_path
    )
    domains = load_domains(
        domains_path
    )

    output_path = get_output_path(
        cfg=cfg,
        output_path=output_path,
    )

    if missing_output_path is None:
        missing_output_path = (
            get_missing_batch_path(cfg)
        )

    model_keys = (
        get_question_generator_models(
            cfg=cfg,
            models=models,
        )
    )

    generation_cfg = cfg.get(
        "generation",
        {},
    )

    difficulties = generation_cfg[
        "difficulties"
    ]

    n_per_batch = generation_cfg[
        "num_questions_per_domain"
    ]

    existing_items = read_jsonl(
        output_path
    )

    existing_counts = (
        get_existing_batch_counts(
            existing_items
        )
    )

    expected_batches = (
        build_expected_batches(
            model_keys=model_keys,
            domains=domains,
            difficulties=difficulties,
            n_per_batch=n_per_batch,
        )
    )

    missing_batches = []

    print(
        "=" * 80,
        flush=True,
    )
    print(
        "Question Missing Batch Check",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )
    print(
        f"[INFO] Output path: "
        f"{output_path}",
        flush=True,
    )
    print(
        f"[INFO] Existing question rows: "
        f"{len(existing_items)}",
        flush=True,
    )
    print(
        f"[INFO] Question generator models: "
        f"{model_keys}",
        flush=True,
    )
    print(
        f"[INFO] Domains: {domains}",
        flush=True,
    )
    print(
        f"[INFO] Difficulties: "
        f"{difficulties}",
        flush=True,
    )

    expected_total_questions = (
        len(expected_batches)
        * n_per_batch
    )

    print(
        f"[INFO] Expected total questions: "
        f"{expected_total_questions}",
        flush=True,
    )

    for batch in expected_batches:
        batch_id = batch["batch_id"]

        existing_n = existing_counts.get(
            batch_id,
            0,
        )

        expected_n = batch[
            "expected_n"
        ]

        if existing_n < expected_n:
            remaining_n = (
                expected_n - existing_n
            )

            missing_batches.append(
                {
                    **batch,
                    "existing_n": existing_n,
                    "remaining_n": remaining_n,
                }
            )

    recognized_batch_count = len(
        existing_counts
    )

    print(
        "[INFO] Existing completed/partial "
        f"batches: {recognized_batch_count}",
        flush=True,
    )
    print(
        "[INFO] Missing or incomplete "
        f"batches: {len(missing_batches)}",
        flush=True,
    )

    missing_by_model = Counter()
    remaining_questions_by_model = Counter()

    for batch in missing_batches:
        model_key = batch["model_key"]

        missing_by_model[
            model_key
        ] += 1

        remaining_questions_by_model[
            model_key
        ] += batch["remaining_n"]

    if missing_batches:
        print(
            "\n" + "=" * 80,
            flush=True,
        )
        print(
            "Missing Batch Summary",
            flush=True,
        )
        print(
            "=" * 80,
            flush=True,
        )

        for model_key in model_keys:
            print(
                f"[SUMMARY] model={model_key} | "
                f"missing_batches="
                f"{missing_by_model[model_key]} | "
                f"remaining_questions="
                f"{remaining_questions_by_model[model_key]}",
                flush=True,
            )

        print(
            "\n" + "=" * 80,
            flush=True,
        )
        print(
            "Missing / Incomplete "
            "Question Batches",
            flush=True,
        )
        print(
            "=" * 80,
            flush=True,
        )

        for index, batch in enumerate(
            missing_batches,
            start=1,
        ):
            print(
                "-" * 80,
                flush=True,
            )
            print(
                f"[MISSING BATCH {index}]",
                flush=True,
            )
            print(
                f"model_key: "
                f"{batch['model_key']}",
                flush=True,
            )
            print(
                f"domain: "
                f"{batch['domain']}",
                flush=True,
            )
            print(
                f"difficulty: "
                f"{batch['difficulty']}",
                flush=True,
            )
            print(
                f"existing_n: "
                f"{batch['existing_n']}",
                flush=True,
            )
            print(
                f"remaining_n: "
                f"{batch['remaining_n']}",
                flush=True,
            )

    else:
        print(
            "\n[DONE] No missing "
            "question batches found.",
            flush=True,
        )

    write_jsonl(
        missing_batches,
        missing_output_path,
    )

    print(
        "\n" + "=" * 80,
        flush=True,
    )
    print(
        "[DONE] Saved missing batch "
        f"list to: {missing_output_path}",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    return missing_batches


def parse_question_batch(
    raw,
    model_key,
    domain,
    difficulty,
):
    """
    Parse raw model output into a list of
    question dictionaries.
    """

    if raw is None:
        print(
            "[ERROR] API returned None | "
            f"model={model_key}, "
            f"domain={domain}, "
            f"difficulty={difficulty}",
            flush=True,
        )
        return []

    if not isinstance(
        raw,
        str,
    ):
        print(
            "[ERROR] API returned "
            f"non-string type={type(raw)} | "
            f"model={model_key}, "
            f"domain={domain}, "
            f"difficulty={difficulty}",
            flush=True,
        )
        return []

    if raw.strip() == "":
        print(
            "[ERROR] API returned empty "
            f"string | model={model_key}, "
            f"domain={domain}, "
            f"difficulty={difficulty}",
            flush=True,
        )
        return []

    print(
        "[DEBUG] Raw output first "
        f"300 chars:\n{raw[:300]}",
        flush=True,
    )

    parsed = safe_json_loads(
        raw
    )

    if not parsed:
        print(
            "[WARN] Could not parse JSON | "
            f"model={model_key}, "
            f"domain={domain}, "
            f"difficulty={difficulty}",
            flush=True,
        )
        print(
            "[WARN] Raw output was:\n"
            f"{raw[:1000]}",
            flush=True,
        )
        return []

    if isinstance(
        parsed,
        dict,
    ):
        parsed = [parsed]

    if not isinstance(
        parsed,
        list,
    ):
        print(
            "[WARN] Parsed output is not "
            f"a list/dict | type={type(parsed)}. "
            "Skipping.",
            flush=True,
        )
        return []

    clean_items = []

    for question_item in parsed:
        if not isinstance(
            question_item,
            dict,
        ):
            print(
                "[WARN] Parsed question item "
                "is not a dictionary. "
                "Skipping one item.",
                flush=True,
            )
            continue

        question_text = question_item.get(
            "question"
        )

        gold_answer = question_item.get(
            "gold_answer"
        )

        if (
            not question_text
            or not gold_answer
        ):
            print(
                "[WARN] Question item is "
                "missing question or "
                "gold_answer. "
                "Skipping one item.",
                flush=True,
            )
            continue

        clean_items.append(
            question_item
        )

    return clean_items


def generate_one_question_batch(
    client,
    model_key,
    model_cfg,
    domain,
    difficulty,
    n,
    output_path,
    existing_question_ids=None,
):
    """
    Generate one batch:

        one model
        × one domain
        × one difficulty

    Parsed questions are appended immediately.
    """

    if existing_question_ids is None:
        existing_question_ids = set()

    print(
        "\n[BATCH] "
        f"model={model_key} / "
        f"{model_cfg['model_name']} | "
        f"domain={domain} | "
        f"difficulty={difficulty} | "
        f"n={n}",
        flush=True,
    )

    system_prompt, user_prompt = (
        build_question_generation_prompt(
            domain,
            difficulty,
            n,
        )
    )

    start_time = time.time()

    try:
        raw = client.generate(
            system_prompt,
            user_prompt,
        )

    except Exception as error:
        print(
            "[ERROR] API call failed | "
            f"model={model_key}, "
            f"domain={domain}, "
            f"difficulty={difficulty}: "
            f"{error}",
            flush=True,
        )
        return []

    elapsed = (
        time.time() - start_time
    )

    print(
        "[INFO] API call finished in "
        f"{elapsed:.2f}s",
        flush=True,
    )

    parsed_items = parse_question_batch(
        raw=raw,
        model_key=model_key,
        domain=domain,
        difficulty=difficulty,
    )

    # A provider may return more items than requested.
    # Save at most the requested remaining quantity.
    parsed_items = parsed_items[:n]

    print(
        "[INFO] Parsed valid questions "
        f"selected for saving: "
        f"{len(parsed_items)} / {n}",
        flush=True,
    )

    saved_items = []

    batch_id = make_generation_batch_id(
        model_key=model_key,
        domain=domain,
        difficulty=difficulty,
    )

    for question_item in parsed_items:
        question_id = question_item.get(
            "id"
        )

        if (
            not question_id
            or question_id
            in existing_question_ids
        ):
            question_id = (
                f"q_{uuid.uuid4().hex[:10]}"
            )

        question_item["id"] = (
            question_id
        )

        question_item[
            "question_generator"
        ] = model_key

        question_item[
            "question_source_model_name"
        ] = model_cfg["model_name"]

        question_item[
            "generation_batch_id"
        ] = batch_id

        # Force configured labels instead of
        # trusting potentially inconsistent
        # model-provided values.
        question_item[
            "domain"
        ] = domain

        question_item[
            "difficulty"
        ] = difficulty

        append_jsonl(
            question_item,
            output_path,
        )

        existing_question_ids.add(
            question_id
        )

        saved_items.append(
            question_item
        )

        print(
            f"[SAVED] {question_id} | "
            f"model={model_key} | "
            f"domain={domain} | "
            f"difficulty={difficulty} | "
            f"{question_item.get('question', '')[:80]}",
            flush=True,
        )

    if len(saved_items) < n:
        print(
            "[WARN] Batch is still incomplete | "
            f"model={model_key}, "
            f"domain={domain}, "
            f"difficulty={difficulty}, "
            f"requested={n}, "
            f"saved={len(saved_items)}. "
            "The next run will retry the "
            "remaining quantity.",
            flush=True,
        )

    return saved_items


def full_generate_questions(
    config_path="config/config.yaml",
    models_path="config/models.yaml",
    domains_path="config/domains.yaml",
    output_path=None,
):
    """
    Full multi-model question generation.

    Warning:
    This function clears output_path first.

    Normal runs should use generate_questions().
    """

    cfg = load_config(
        config_path
    )
    models = load_models(
        models_path
    )
    domains = load_domains(
        domains_path
    )

    output_path = get_output_path(
        cfg=cfg,
        output_path=output_path,
    )

    model_keys = (
        get_question_generator_models(
            cfg=cfg,
            models=models,
        )
    )

    generation_cfg = cfg.get(
        "generation",
        {},
    )

    difficulties = generation_cfg[
        "difficulties"
    ]

    n_per_batch = generation_cfg[
        "num_questions_per_domain"
    ]

    print(
        "=" * 80,
        flush=True,
    )
    print(
        "[FULL] Multi-model "
        "question generation",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )
    print(
        f"[INFO] Output path: "
        f"{output_path}",
        flush=True,
    )
    print(
        f"[INFO] Models: "
        f"{model_keys}",
        flush=True,
    )
    print(
        f"[INFO] Domains: "
        f"{domains}",
        flush=True,
    )
    print(
        f"[INFO] Difficulties: "
        f"{difficulties}",
        flush=True,
    )
    print(
        "[INFO] Questions per "
        "model/domain/difficulty: "
        f"{n_per_batch}",
        flush=True,
    )

    # Clear only when the caller explicitly
    # requests full regeneration.
    write_jsonl(
        [],
        output_path,
    )

    total_batches = (
        len(model_keys)
        * len(domains)
        * len(difficulties)
    )

    batch_index = 0
    all_items = []
    clients = {}
    existing_question_ids = set()

    for model_key in model_keys:
        model_cfg = models[
            model_key
        ]

        clients[model_key] = (
            create_model_client(
                cfg=cfg,
                model_cfg=model_cfg,
            )
        )

    for model_key in model_keys:
        model_cfg = models[
            model_key
        ]
        client = clients[
            model_key
        ]

        for domain in domains:
            for difficulty in difficulties:
                batch_index += 1

                print(
                    "\n[CALL "
                    f"{batch_index}/"
                    f"{total_batches}] "
                    f"model={model_key}, "
                    f"domain={domain}, "
                    f"difficulty={difficulty}",
                    flush=True,
                )

                saved_items = (
                    generate_one_question_batch(
                        client=client,
                        model_key=model_key,
                        model_cfg=model_cfg,
                        domain=domain,
                        difficulty=difficulty,
                        n=n_per_batch,
                        output_path=output_path,
                        existing_question_ids=(
                            existing_question_ids
                        ),
                    )
                )

                all_items.extend(
                    saved_items
                )

    print(
        "\n[DONE] Generated total "
        f"questions: {len(all_items)}",
        flush=True,
    )
    print(
        f"[DONE] Saved to: "
        f"{output_path}",
        flush=True,
    )

    return all_items


def generate_missing_questions(
    config_path="config/config.yaml",
    models_path="config/models.yaml",
    domains_path="config/domains.yaml",
    output_path=None,
):
    """
    Generate only missing/incomplete question batches.

    This function never clears questions_raw.jsonl.
    """

    cfg = load_config(
        config_path
    )
    models = load_models(
        models_path
    )

    output_path = get_output_path(
        cfg=cfg,
        output_path=output_path,
    )

    missing_batches = (
        check_missing_question_batches(
            config_path=config_path,
            models_path=models_path,
            domains_path=domains_path,
            output_path=output_path,
        )
    )

    if not missing_batches:
        print(
            "[DONE] Nothing missing. "
            "No question generation needed.",
            flush=True,
        )
        return []

    existing_items = read_jsonl(
        output_path
    )

    existing_question_ids = (
        get_existing_question_ids(
            existing_items
        )
    )

    clients = {}
    new_items = []

    print(
        "=" * 80,
        flush=True,
    )
    print(
        "[MISSING] Generating "
        "missing question batches",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    for index, batch in enumerate(
        missing_batches,
        start=1,
    ):
        model_key = batch[
            "model_key"
        ]
        domain = batch[
            "domain"
        ]
        difficulty = batch[
            "difficulty"
        ]
        remaining_n = batch[
            "remaining_n"
        ]

        model_cfg = models[
            model_key
        ]

        if model_key not in clients:
            clients[model_key] = (
                create_model_client(
                    cfg=cfg,
                    model_cfg=model_cfg,
                )
            )

        print(
            "\n[MISSING BATCH "
            f"{index}/"
            f"{len(missing_batches)}] "
            f"model={model_key} | "
            f"domain={domain} | "
            f"difficulty={difficulty} | "
            f"remaining_n={remaining_n}",
            flush=True,
        )

        saved_items = (
            generate_one_question_batch(
                client=clients[model_key],
                model_key=model_key,
                model_cfg=model_cfg,
                domain=domain,
                difficulty=difficulty,
                n=remaining_n,
                output_path=output_path,
                existing_question_ids=(
                    existing_question_ids
                ),
            )
        )

        new_items.extend(
            saved_items
        )

    print(
        "\n[DONE] Newly generated "
        "missing questions: "
        f"{len(new_items)}",
        flush=True,
    )
    print(
        f"[DONE] Appended to: "
        f"{output_path}",
        flush=True,
    )

    return new_items


def generate_questions(
    config_path="config/config.yaml",
    models_path="config/models.yaml",
    domains_path="config/domains.yaml",
    output_path=None,
    force_full_regenerate=False,
):
    """
    Safe auto mode for multi-model question generation.

    Default behavior:
    - preserve all existing records;
    - identify completed and incomplete batches;
    - generate only missing quantities;
    - never clear the output automatically.

    Full regeneration occurs only when:
        force_full_regenerate=True
    """

    cfg = load_config(
        config_path
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
        "[AUTO] Multi-model "
        "question generation",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )
    print(
        f"[AUTO] Output path: "
        f"{output_path}",
        flush=True,
    )
    print(
        "[AUTO] Force full regenerate: "
        f"{force_full_regenerate}",
        flush=True,
    )

    existing_items = read_jsonl(
        output_path
    )

    print(
        "[AUTO] Existing question "
        f"rows detected: {len(existing_items)}",
        flush=True,
    )

    if force_full_regenerate:
        print(
            "[AUTO] Full regeneration was "
            "explicitly requested. "
            "The existing output file "
            "will be cleared.",
            flush=True,
        )

        return full_generate_questions(
            config_path=config_path,
            models_path=models_path,
            domains_path=domains_path,
            output_path=output_path,
        )

    print(
        "[AUTO] Checking every "
        "model/domain/difficulty batch. "
        "Only missing quantities "
        "will be generated.",
        flush=True,
    )

    return generate_missing_questions(
        config_path=config_path,
        models_path=models_path,
        domains_path=domains_path,
        output_path=output_path,
    )