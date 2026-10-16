"""FaultyVerifierBench Step 08 adjudication execution engine."""

from __future__ import annotations

import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import yaml

from prompts.adjudication_prompts import build_adjudication_prompt
from src.api.client_factory import get_client
from src.config.taxonomy_loader import load_benchmark_taxonomies
from src.dataset.io_utils import append_jsonl, read_jsonl, write_jsonl
from src.evaluation.adjudication_validation import finalize_adjudication_record
from src.utils.json_utils import safe_json_loads


_THREAD_LOCAL = threading.local()
SCHEMA_VERSION = "2.0"
MAX_IN_FLIGHT_MULTIPLIER = 2


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping in {config_path}.")
    return value


def load_models(models_path="config/models.yaml"):
    with open(models_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    models = data.get("models")
    if not isinstance(models, dict):
        raise TypeError(f"Expected a models mapping in {models_path}.")
    return models


def get_evaluation_setting(cfg, key, default=None):
    return cfg.get("evaluation", {}).get(key, default)


def get_models_path(cfg, models_path=None):
    return models_path or get_evaluation_setting(
        cfg,
        "models_path",
        "config/models.yaml",
    )


def get_split(cfg, split=None):
    return split or get_evaluation_setting(cfg, "default_split", "test")


def get_target_model_keys(cfg, models, model_key=None):
    if model_key is not None:
        if model_key not in models:
            raise ValueError(
                f"Model key {model_key!r} is not defined in models.yaml. "
                f"Available models: {list(models)}"
            )
        return [model_key]

    target_models = get_evaluation_setting(cfg, "target_models")
    if target_models:
        missing = [key for key in target_models if key not in models]
        if missing:
            raise ValueError(
                f"Target models missing from models.yaml: {missing}. "
                f"Available models: {list(models)}"
            )
        return list(target_models)

    default_model = get_evaluation_setting(cfg, "default_model")
    if default_model:
        if default_model not in models:
            raise ValueError(
                f"Default model {default_model!r} is not in models.yaml."
            )
        return [default_model]

    return list(models)


def get_default_input_path(cfg, split):
    split_dir = cfg.get("paths", {}).get("split_dir", "data/splits")
    return f"{split_dir}/{split}.jsonl"


def get_default_output_path(cfg, model_key, split):
    result_dir = cfg.get("paths", {}).get("result_dir", "data/results")
    return f"{result_dir}/model_outputs_{model_key}_{split}.jsonl"


def get_missing_output_path(cfg, model_key, split):
    result_dir = cfg.get("paths", {}).get("result_dir", "data/results")
    return f"{result_dir}/missing_outputs_{model_key}_{split}.jsonl"


def get_api_settings(cfg, provider=None):
    """Return provider-aware timeouts and shared retry settings.

    This signature intentionally accepts ``provider``. The previous one-argument
    function caused every worker to fail before creating a client.
    """

    api_cfg = cfg.get("api", {})
    generation_cfg = cfg.get("generation", {})
    normalized_provider = (
        str(provider).strip().lower() if provider is not None else ""
    )
    provider_cfg = api_cfg.get("provider_timeouts", {}).get(
        normalized_provider,
        {},
    )

    return {
        "connect_timeout": float(
            provider_cfg.get(
                "connect_timeout_seconds",
                api_cfg.get("connect_timeout_seconds", 20),
            )
        ),
        "read_timeout": float(
            provider_cfg.get(
                "read_timeout_seconds",
                api_cfg.get("read_timeout_seconds", 240),
            )
        ),
        "max_retries": max(
            1,
            int(generation_cfg.get("max_retries", 3)),
        ),
        "retry_base_delay": float(
            generation_cfg.get("retry_base_delay_seconds", 5)
        ),
    }


def get_workers_per_model(cfg, workers_per_model=None):
    if workers_per_model is None:
        workers_per_model = get_evaluation_setting(
            cfg,
            "evaluation_workers_per_model",
            1,
        )
    return max(1, int(workers_per_model))


def create_model_client(cfg, model_cfg):
    """Create one provider client using one shared output-token limit."""

    evaluation = cfg.get("evaluation", {})
    generation = cfg.get("generation", {})
    provider = str(model_cfg["provider"]).strip().lower()
    model_name = str(model_cfg["model_name"]).strip()
    temperature = float(
        evaluation.get(
            "temperature",
            generation.get("temperature", 0.7),
        )
    )
    max_output_tokens = int(evaluation.get("max_output_tokens", 2048))

    if max_output_tokens <= 0:
        raise ValueError(
            "evaluation.max_output_tokens must be greater than zero."
        )

    api_settings = get_api_settings(cfg=cfg, provider=provider)

    print(
        "[CLIENT] Generation configuration | "
        f"provider={provider} | model_name={model_name} | "
        f"max_output_tokens={max_output_tokens} | "
        f"connect_timeout={api_settings['connect_timeout']}s | "
        f"read_timeout={api_settings['read_timeout']}s | "
        f"max_retries={api_settings['max_retries']}",
        flush=True,
    )

    return get_client(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        gemini_thinking_level=str(
            evaluation.get("gemini_thinking_level", "high")
        ),
        gemini_include_thoughts=bool(
            evaluation.get("gemini_include_thoughts", False)
        ),
        deepseek_thinking_enabled=bool(
            evaluation.get("deepseek_thinking_enabled", True)
        ),
        deepseek_reasoning_effort=str(
            evaluation.get("deepseek_reasoning_effort", "high")
        ),
        local_do_sample=bool(
            evaluation.get("local_do_sample", False)
        ),
        local_enable_thinking=bool(
            evaluation.get("local_enable_thinking", False)
        ),
        local_trust_remote_code=bool(
            evaluation.get("local_trust_remote_code", True)
        ),
        **api_settings,
    )


def get_thread_client(cfg, model_key, model_cfg):
    client_cache = getattr(_THREAD_LOCAL, "client_cache", None)
    if client_cache is None:
        client_cache = {}
        _THREAD_LOCAL.client_cache = client_cache

    if model_key not in client_cache:
        print(
            "[CLIENT] Initializing thread-local client | "
            f"model={model_key} | thread={threading.current_thread().name}",
            flush=True,
        )
        client_cache[model_key] = create_model_client(
            cfg=cfg,
            model_cfg=model_cfg,
        )

    return client_cache[model_key]


def _parse_adjudication_output(raw, item_id, model_key):
    if raw is None:
        print(
            f"API returned None | model={model_key} | item_id={item_id}",
            flush=True,
        )
        return None
    if not isinstance(raw, str):
        print(
            f"API returned non-string | model={model_key} | "
            f"item_id={item_id} | type={type(raw).__name__}",
            flush=True,
        )
        return None
    if not raw.strip():
        print(
            f"API returned empty text | model={model_key} | "
            f"item_id={item_id}",
            flush=True,
        )
        return None

    print(
        f"[DEBUG] Adjudication output | model={model_key} | "
        f"item_id={item_id} | preview={raw[:900]}",
        flush=True,
    )

    parsed = safe_json_loads(raw)
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else None
    if not isinstance(parsed, dict):
        print(
            f"[WARN] Could not parse adjudication JSON | model={model_key} | "
            f"item_id={item_id}",
            flush=True,
        )
        return None

    for field in (
        "valid_verifier_points",
        "faulty_verifier_points",
        "fallacy_detected",
        "fallacy_explanations",
        "hallucination_detected",
        "hallucination_explanations",
    ):
        parsed.setdefault(field, [])
    parsed.setdefault("justification", "")
    return parsed


def _build_output_record(item, model_key, model_cfg, raw, parsed):
    return {
        "id": item["id"],
        "target_model": model_key,
        "target_model_name": model_cfg["model_name"],
        "raw_output": raw,
        "parsed_output": parsed,
        "adjudication_schema_version": parsed.get(
            "adjudication_schema_version",
            SCHEMA_VERSION,
        ),
        "adjudication_parse_success": parsed.get(
            "adjudication_parse_success",
            False,
        ),
    }


def generate_one_adjudication_output(
    item,
    model_key,
    model_cfg,
    client,
    allowed_fallacies,
    allowed_hallucinations,
):
    item_id = item.get("id")
    print(
        "[CALL] Running adjudication | "
        f"model={model_key} | item_id={item_id} | "
        f"domain={item.get('domain', '')} | "
        f"mode={item.get('verifier_generation_mode', item.get('verifier_correctness', ''))}",
        flush=True,
    )

    try:
        system_prompt, user_prompt = build_adjudication_prompt(item)
    except Exception as error:
        print(
            "Could not build adjudication prompt | "
            f"model={model_key} | item_id={item_id} | "
            f"error_type={type(error).__name__} | error={error}",
            flush=True,
        )
        return None

    started = time.time()
    try:
        raw = client.generate(system_prompt, user_prompt)
    except Exception as error:
        print(
            "Adjudication API call failed | "
            f"model={model_key} | item_id={item_id} | "
            f"elapsed={time.time() - started:.2f}s | "
            f"error_type={type(error).__name__} | error={error}",
            flush=True,
        )
        return None

    print(
        "Adjudication API call completed | "
        f"model={model_key} | item_id={item_id} | "
        f"elapsed={time.time() - started:.2f}s",
        flush=True,
    )

    parsed = _parse_adjudication_output(raw, item_id, model_key)
    if parsed is None:
        return None

    validated = finalize_adjudication_record(
        parsed=parsed,
        allowed_fallacies=allowed_fallacies,
        allowed_hallucinations=allowed_hallucinations,
    )
    if not validated["adjudication_parse_success"]:
        print(
            "[WARN] Adjudication schema validation failed | "
            f"model={model_key} | item_id={item_id} | "
            f"errors={validated['adjudication_validation_errors']}",
            flush=True,
        )
        return None

    return _build_output_record(
        item=item,
        model_key=model_key,
        model_cfg=model_cfg,
        raw=raw,
        parsed=validated,
    )


def get_completed_ids(output_path):
    outputs = read_jsonl(output_path)
    completed_ids = {
        output.get("id")
        for output in outputs
        if output.get("id")
        and (
            output.get("adjudication_parse_success") is True
            or output.get("parsed_output", {}).get(
                "adjudication_parse_success"
            ) is True
        )
        and (
            output.get("adjudication_schema_version") == SCHEMA_VERSION
            or output.get("parsed_output", {}).get(
                "adjudication_schema_version"
            ) == SCHEMA_VERSION
        )
    }
    counts = Counter(
        output.get("id") for output in outputs if output.get("id")
    )
    duplicates = [
        {"id": item_id, "count": count}
        for item_id, count in counts.items()
        if count > 1
    ]
    return outputs, completed_ids, duplicates


def build_model_task_plans(
    cfg,
    target_model_keys,
    split,
    input_items,
    output_path=None,
):
    if output_path is not None and len(target_model_keys) > 1:
        print(
            "[WARN] --output is ignored in multi-model mode.",
            flush=True,
        )

    task_plans = {}
    output_paths = {}

    for model_key in target_model_keys:
        current_output_path = (
            output_path
            if output_path is not None and len(target_model_keys) == 1
            else get_default_output_path(cfg, model_key, split)
        )
        output_paths[model_key] = current_output_path
        existing, completed_ids, duplicate_ids = get_completed_ids(
            current_output_path
        )
        missing = [
            item
            for item in input_items
            if item.get("id") and item.get("id") not in completed_ids
        ]
        task_plans[model_key] = missing
        missing_output_path = get_missing_output_path(cfg, model_key, split)
        write_jsonl(missing, missing_output_path)

        print("=" * 80, flush=True)
        print(f"Model: {model_key}", flush=True)
        print(f"Existing rows: {len(existing)}", flush=True)
        print(
            f"Reusable schema-v2 IDs: {len(completed_ids)}",
            flush=True,
        )
        print(f"Duplicate IDs: {len(duplicate_ids)}", flush=True)
        print(f"Missing items: {len(missing)}", flush=True)
        print(f"Output path: {current_output_path}", flush=True)
        print(f"Missing-list path: {missing_output_path}", flush=True)

    return task_plans, output_paths


def clear_model_output_files(
    cfg,
    target_model_keys,
    split,
    output_path=None,
):
    for model_key in target_model_keys:
        current_output_path = (
            output_path
            if output_path is not None and len(target_model_keys) == 1
            else get_default_output_path(cfg, model_key, split)
        )
        Path(current_output_path).parent.mkdir(parents=True, exist_ok=True)
        write_jsonl([], current_output_path)
        print(
            f"Cleared output | model={model_key} | "
            f"path={current_output_path}",
            flush=True,
        )


def run_models_concurrently(
    task_plans,
    output_paths,
    models,
    cfg,
    workers_per_model,
    allowed_fallacies,
    allowed_hallucinations,
):
    active_models = [key for key, items in task_plans.items() if items]
    if not active_models:
        print("No missing evaluation tasks.", flush=True)
        return {key: [] for key in task_plans}

    total_tasks = sum(len(task_plans[key]) for key in active_models)
    print("=" * 80, flush=True)
    print("Starting independent model executors", flush=True)
    print(f"Active models: {active_models}", flush=True)
    print(f"Workers per model: {workers_per_model}", flush=True)
    print(
        f"Maximum total concurrency: "
        f"{len(active_models) * workers_per_model}",
        flush=True,
    )
    print(f"Total missing tasks: {total_tasks}", flush=True)
    print("Only the main thread writes JSONL.", flush=True)

    executors = {
        key: ThreadPoolExecutor(
            max_workers=workers_per_model,
            thread_name_prefix=f"evaluation-{key}",
        )
        for key in active_models
    }
    item_iterators = {
        key: iter(task_plans[key]) for key in active_models
    }
    futures: dict[Future, dict[str, Any]] = {}
    max_in_flight = max(1, workers_per_model * MAX_IN_FLIGHT_MULTIPLIER)

    completed_total = 0
    saved_total = 0
    failed_total = 0
    completed_by_model = Counter()
    saved_by_model = Counter()
    failed_by_model = Counter()
    outputs_by_model = {key: [] for key in task_plans}

    def submit_one(model_key):
        try:
            item = next(item_iterators[model_key])
        except StopIteration:
            return False

        def run_item():
            model_cfg = models[model_key]
            client = get_thread_client(cfg, model_key, model_cfg)
            return generate_one_adjudication_output(
                item=item,
                model_key=model_key,
                model_cfg=model_cfg,
                client=client,
                allowed_fallacies=allowed_fallacies,
                allowed_hallucinations=allowed_hallucinations,
            )

        future = executors[model_key].submit(run_item)
        futures[future] = {
            "model_key": model_key,
            "item_id": item.get("id"),
        }
        return True

    try:
        for model_key in active_models:
            for _ in range(max_in_flight):
                if not submit_one(model_key):
                    break

        while futures:
            done, _ = wait(
                futures,
                timeout=0.5,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue

            for future in done:
                metadata = futures.pop(future)
                model_key = metadata["model_key"]
                item_id = metadata["item_id"]
                completed_total += 1
                completed_by_model[model_key] += 1

                try:
                    output_record = future.result()
                except Exception as error:
                    output_record = None
                    print(
                        "Unexpected evaluation worker failure | "
                        f"model={model_key} | item_id={item_id} | "
                        f"error_type={type(error).__name__} | error={error}",
                        flush=True,
                    )

                if output_record is not None:
                    append_jsonl(output_record, output_paths[model_key])
                    outputs_by_model[model_key].append(output_record)
                    saved_total += 1
                    saved_by_model[model_key] += 1
                    parsed = output_record.get("parsed_output", {})
                    print(
                        f"model={model_key} | item_id={item_id} | "
                        f"judgment={parsed.get('verifier_judgment', '')}",
                        flush=True,
                    )
                else:
                    failed_total += 1
                    failed_by_model[model_key] += 1

                print(
                    f"total_completed={completed_total}/{total_tasks} | "
                    f"saved={saved_total} | failed={failed_total} | "
                    f"remaining={total_tasks - completed_total}",
                    flush=True,
                )
                print(
                    "[MODEL PROGRESS] "
                    + " | ".join(
                        f"{key}: completed={completed_by_model[key]}/"
                        f"{len(task_plans[key])}, saved={saved_by_model[key]}, "
                        f"failed={failed_by_model[key]}"
                        for key in active_models
                    ),
                    flush=True,
                )

                submit_one(model_key)

    except KeyboardInterrupt:
        print(
            "Cancelling queued tasks. Completed outputs are saved.",
            flush=True,
        )
        for future in futures:
            future.cancel()
        raise
    finally:
        for executor in executors.values():
            executor.shutdown(wait=False, cancel_futures=True)

    print("=" * 80, flush=True)
    print("Concurrent multi-model evaluation finished", flush=True)
    print(f"Completed tasks: {completed_total}/{total_tasks}", flush=True)
    print(f"Saved outputs: {saved_total}", flush=True)
    print(f"Failed or invalid outputs: {failed_total}", flush=True)
    return outputs_by_model


def auto_run_adjudication(
    model_key=None,
    split=None,
    models_path=None,
    config_path="config/config.yaml",
    input_path=None,
    output_path=None,
    force_full_regenerate=False,
    workers_per_model=None,
):
    cfg = load_config(config_path)
    models_path = get_models_path(cfg, models_path)
    models = load_models(models_path)
    split = get_split(cfg, split)
    target_model_keys = get_target_model_keys(cfg, models, model_key)
    if input_path is None:
        input_path = get_default_input_path(cfg, split)
    workers_per_model = get_workers_per_model(cfg, workers_per_model)

    project_root = Path(config_path).resolve().parents[1]
    allowed_fallacies, allowed_hallucinations = load_benchmark_taxonomies(
        project_root
    )

    print("=" * 80, flush=True)
    print("Independent concurrent model evaluation", flush=True)
    print(f"Target models: {target_model_keys}", flush=True)
    print(f"Split: {split}", flush=True)
    print(f"Input path: {input_path}", flush=True)
    print(f"Models path: {models_path}", flush=True)
    print(f"Workers per model: {workers_per_model}", flush=True)
    print(
        f"Shared max_output_tokens: "
        f"{get_evaluation_setting(cfg, 'max_output_tokens', 2048)}",
        flush=True,
    )
    print(
        f"Taxonomies: fallacies={len(allowed_fallacies)} | "
        f"hallucinations={len(allowed_hallucinations)}",
        flush=True,
    )

    input_items = read_jsonl(input_path)
    if not input_items:
        raise ValueError(f"No evaluation items found at {input_path}")

    if force_full_regenerate:
        clear_model_output_files(
            cfg,
            target_model_keys,
            split,
            output_path,
        )

    task_plans, output_paths = build_model_task_plans(
        cfg,
        target_model_keys,
        split,
        input_items,
        output_path,
    )

    return run_models_concurrently(
        task_plans=task_plans,
        output_paths=output_paths,
        models=models,
        cfg=cfg,
        workers_per_model=workers_per_model,
        allowed_fallacies=allowed_fallacies,
        allowed_hallucinations=allowed_hallucinations,
    )
