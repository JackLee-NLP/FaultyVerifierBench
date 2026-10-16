#!/usr/bin/env python3
"""Isolated local-model launcher for the unchanged project Step 08 runner.

Place at:
    objectively_verify/scripts_qwen38/08_run_evaluation.py

All generated files remain inside scripts_qwen38. Files under scripts/, src/,
and config/ are read but never modified.
"""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ORIGINAL_RUNNER = ROOT / "scripts" / "08_run_evaluation.py"
RUNTIME_RUNNER = HERE / ".runtime_08_run_evaluation.py"
RUNTIME_CONFIG = HERE / ".runtime_config.yaml"
RUNTIME_MODELS = HERE / ".runtime_models.yaml"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from memory_guard import MemoryGuard
from qwen38_client import Qwen38LinuxClient


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def private_model_definition() -> tuple[str, dict[str, Any]]:
    document = read_yaml(HERE / "models.yaml")
    models = document.get("models", document)
    if not isinstance(models, dict) or not models:
        raise RuntimeError("scripts_qwen38/models.yaml contains no model definition")

    requested = os.getenv("LOCAL_MODEL_KEY", "").strip()
    if requested:
        if requested not in models:
            raise KeyError(
                f"LOCAL_MODEL_KEY={requested!r} is not in scripts_qwen38/models.yaml"
            )
        model_key = requested
    elif len(models) == 1:
        model_key = next(iter(models))
    else:
        raise RuntimeError(
            "scripts_qwen38/models.yaml contains multiple models. Set "
            "LOCAL_MODEL_KEY to the one Step 08 should run."
        )

    model_config = models[model_key]
    if not isinstance(model_config, dict):
        raise TypeError(f"Model configuration must be a mapping: {model_key}")
    return str(model_key), model_config


def model_path_from_config(model_key: str, model_config: dict[str, Any]) -> Path:
    model_name = model_config.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise TypeError(f"{model_key}.model_name must be a non-empty string")
    path = Path(model_name).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {path}")
    return path


def prepare_runtime_files(model_key: str) -> Path:
    base_config = read_yaml(ROOT / "config" / "config.yaml")
    private_config = read_yaml(HERE / "config_override.yaml")
    merged_config = deep_merge(base_config, private_config)

    base_models_document = read_yaml(ROOT / "config" / "models.yaml")
    private_models_document = read_yaml(HERE / "models.yaml")
    base_models = base_models_document.get("models", base_models_document)
    private_models = private_models_document.get("models", private_models_document)
    if not isinstance(base_models, dict) or not isinstance(private_models, dict):
        raise TypeError("Both model configuration files must contain mappings")

    runtime_models = {"models": deep_merge(base_models, private_models)}
    with RUNTIME_MODELS.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(runtime_models, handle, sort_keys=False, allow_unicode=True)

    evaluation = merged_config.setdefault("evaluation", {})
    if not isinstance(evaluation, dict):
        raise TypeError("config.yaml evaluation section must be a mapping")
    evaluation["models_path"] = str(RUNTIME_MODELS)
    evaluation["target_models"] = [model_key]
    evaluation["evaluation_workers_per_model"] = 1
    evaluation["evaluation_max_concurrent_per_model"] = 1
    evaluation["max_output_tokens"] = 2048

    with RUNTIME_CONFIG.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(merged_config, handle, sort_keys=False, allow_unicode=True)
    return RUNTIME_CONFIG


def load_index_weight_keys(model_path: Path) -> list[str]:
    keys: set[str] = set()
    for index_path in sorted(model_path.glob("*.safetensors.index.json")):
        with index_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        weight_map = data.get("weight_map", {})
        if isinstance(weight_map, dict):
            keys.update(str(key) for key in weight_map)
    return sorted(keys)


def load_shard_weight_keys(model_path: Path) -> list[str]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required for checkpoint inspection") from error

    keys: set[str] = set()
    for shard_path in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            keys.update(str(key) for key in handle.keys())
    return sorted(keys)


def load_checkpoint_keys(model_path: Path) -> list[str]:
    keys = load_index_weight_keys(model_path)
    if not keys:
        keys = load_shard_weight_keys(model_path)
    if not keys:
        raise RuntimeError(f"No checkpoint tensor keys found: {model_path}")
    return keys


def find_unique_tensor_key(
    keys: list[str],
    exact_candidates: tuple[str, ...],
    suffix_candidates: tuple[str, ...],
    description: str,
) -> str:
    for candidate in exact_candidates:
        if candidate in keys:
            return candidate

    matches = sorted(
        {
            key
            for key in keys
            if any(key.endswith(suffix) for suffix in suffix_candidates)
        }
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(
            f"No {description} tensor found. Checked suffixes={suffix_candidates}"
        )
    raise RuntimeError(
        f"Multiple {description} tensor candidates found; refusing to guess: {matches}"
    )


def find_embedding_key(keys: list[str]) -> str:
    return find_unique_tensor_key(
        keys=keys,
        exact_candidates=(
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "model.language_model.model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
        ),
        suffix_candidates=(".embed_tokens.weight",),
        description="embedding",
    )


def find_output_head_key(keys: list[str]) -> str:
    # 1) Real, untied output head (lm_head / output / unembed)
    try:
        return find_unique_tensor_key(
            keys=keys,
            exact_candidates=(
                "lm_head.weight",
                "model.lm_head.weight",
                "language_model.lm_head.weight",
                "language_model.model.lm_head.weight",
                "model.language_model.lm_head.weight",
                "model.language_model.model.lm_head.weight",
            ),
            suffix_candidates=(
                ".lm_head.weight",
                ".output.weight",
                ".output_layer.weight",
                ".unembed.weight",
            ),
            description="output-head",
        )
    except RuntimeError:
        # 2) Tied embeddings (e.g. Gemma4Unified: tie_word_embeddings=True).
        #    The lm_head is the embedding matrix, so reuse it as the output head.
        return find_unique_tensor_key(
            keys=keys,
            exact_candidates=(
                "embed_tokens.weight",
                "model.embed_tokens.weight",
                "language_model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
                "model.language_model.model.embed_tokens.weight",
            ),
            suffix_candidates=(
                ".embed_tokens.weight",
            ),
            description="output-head (tied to embed_tokens)",
        )


def prepare_runtime_runner(
    actual_embedding_key: str,
    actual_output_head_key: str,
) -> Path:
    if not ORIGINAL_RUNNER.exists():
        raise FileNotFoundError(f"Original Step 08 runner not found: {ORIGINAL_RUNNER}")

    source = ORIGINAL_RUNNER.read_text(encoding="utf-8")
    expected_embedding_key = "model.embed_tokens.weight"
    expected_output_head_key = "lm_head.weight"
    embedding_count = source.count(expected_embedding_key)
    output_head_count = source.count(expected_output_head_key)

    if embedding_count == 0:
        raise RuntimeError(
            f"Validator key {expected_embedding_key!r} was not found in {ORIGINAL_RUNNER}"
        )
    if output_head_count == 0:
        raise RuntimeError(
            f"Validator key {expected_output_head_key!r} was not found in {ORIGINAL_RUNNER}"
        )

    runtime_source = source.replace(expected_embedding_key, actual_embedding_key)
    runtime_source = runtime_source.replace(
        expected_output_head_key,
        actual_output_head_key,
    )
    header = (
        "# AUTO-GENERATED BY scripts_qwen38/08_run_evaluation.py\n"
        "# The original scripts/08_run_evaluation.py remains unchanged.\n"
        f"# Embedding tensor: {actual_embedding_key}\n"
        f"# Output-head tensor: {actual_output_head_key}\n\n"
    )
    RUNTIME_RUNNER.write_text(header + runtime_source, encoding="utf-8")
    print(
        "[LOCAL MODEL] Private validator adapter | "
        f"embedding_key={actual_embedding_key} | "
        f"output_head_key={actual_output_head_key} | "
        f"embedding_replacements={embedding_count} | "
        f"output_head_replacements={output_head_count} | "
        f"runtime_runner={RUNTIME_RUNNER}",
        flush=True,
    )
    return RUNTIME_RUNNER


def install_factory_override(selected_model_path: Path) -> None:
    import src.api.client_factory as factory

    original_get_client = factory.get_client
    selected_resolved = str(selected_model_path)

    def isolated_get_client(*args: Any, **kwargs: Any) -> Any:
        provider = str(kwargs.get("provider", args[0] if args else "")).lower()
        model_name = str(
            kwargs.get("model_name", args[1] if len(args) > 1 else "")
        )
        try:
            candidate = str(Path(model_name).expanduser().resolve())
        except (OSError, RuntimeError, ValueError):
            candidate = model_name

        if provider == "huggingface" and candidate == selected_resolved:
            return Qwen38LinuxClient(*args, **kwargs)
        return original_get_client(*args, **kwargs)

    factory.get_client = isolated_get_client


def normalized_argv(
    config_path: Path,
    runner_path: Path,
    model_key: str,
) -> list[str]:
    user_args = sys.argv[1:]
    result = [str(runner_path), *user_args]
    if "--config" not in user_args:
        result.extend(["--config", str(config_path)])
    if "--model" not in user_args and "--models" not in user_args:
        result.extend(["--model", model_key])
    if "--workers-per-model" not in user_args:
        result.extend(["--workers-per-model", "1"])
    return result


def main() -> None:
    guard = MemoryGuard(
        hard_floor_gib=float(os.getenv("QWEN38_RAM_HARD_FLOOR_GIB", "2")),
        warning_floor_gib=float(os.getenv("QWEN38_RAM_RESERVE_GIB", "3")),
    )
    guard.start()
    try:
        model_key, model_config = private_model_definition()
        model_path = model_path_from_config(model_key, model_config)
        config_path = prepare_runtime_files(model_key)

        checkpoint_keys = load_checkpoint_keys(model_path)
        embedding_key = find_embedding_key(checkpoint_keys)
        output_head_key = find_output_head_key(checkpoint_keys)
        runner_path = prepare_runtime_runner(embedding_key, output_head_key)

        install_factory_override(model_path)
        sys.argv = normalized_argv(config_path, runner_path, model_key)
        runpy.run_path(str(runner_path), run_name="__main__")
    finally:
        guard.stop()


if __name__ == "__main__":
    main()
