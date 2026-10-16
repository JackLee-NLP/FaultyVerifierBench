#!/usr/bin/env python3
"""Isolated Qwen3.8 launcher for the unchanged project Step 08 runner.

Place this file at:
    objectively_verify/scripts_qwen38/08_run_evaluation.py

It creates runtime-only files inside scripts_qwen38 and never modifies files
under scripts/, src/, or config/.
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
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def prepare_runtime_files() -> Path:
    base_config = read_yaml(ROOT / "config" / "config.yaml")
    private_config = read_yaml(HERE / "config_override.yaml")
    merged_config = deep_merge(base_config, private_config)

    base_models = read_yaml(ROOT / "config" / "models.yaml")
    private_models = read_yaml(HERE / "models.yaml")
    base_model_map = base_models.get("models", base_models)
    private_model_map = private_models.get("models", private_models)
    runtime_model_map = deep_merge(base_model_map, private_model_map)

    with RUNTIME_MODELS.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"models": runtime_model_map},
            handle,
            sort_keys=False,
            allow_unicode=True,
        )

    evaluation = merged_config.setdefault("evaluation", {})
    evaluation["models_path"] = str(RUNTIME_MODELS)
    evaluation["target_models"] = ["Ministral-3-14B"]
    evaluation["evaluation_workers_per_model"] = 1
    evaluation["evaluation_max_concurrent_per_model"] = 1
    evaluation["max_output_tokens"] = 2048

    with RUNTIME_CONFIG.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            merged_config,
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    return RUNTIME_CONFIG


def get_qwen_model_path() -> Path:
    private_models = read_yaml(HERE / "models.yaml")
    model_map = private_models.get("models", private_models)
    model_config = model_map.get("Ministral-3-14B")
    if not isinstance(model_config, dict):
        raise KeyError(
            "Ministral-3-14B is missing from scripts_qwen38/models.yaml"
        )
    model_name = model_config.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise TypeError("Ministral-3-14B.model_name must be a non-empty string")
    model_path = Path(model_name).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Qwen3.8 model path does not exist: {model_path}")
    return model_path


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


def find_embedding_key(model_path: Path) -> str:
    keys = load_index_weight_keys(model_path)
    if not keys:
        keys = load_shard_weight_keys(model_path)
    if not keys:
        raise RuntimeError(f"No checkpoint tensor keys found: {model_path}")

    exact_candidates = (
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
    )
    for candidate in exact_candidates:
        if candidate in keys:
            return candidate

    matches = [key for key in keys if key.endswith(".embed_tokens.weight")]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        similar = [
            key for key in keys
            if "embed" in key.lower() and key.endswith(".weight")
        ]
        raise RuntimeError(
            "No embed_tokens.weight tensor found. Similar keys: "
            f"{similar[:30]}"
        )
    raise RuntimeError(
        "Multiple embedding tensor candidates found; refusing to guess: "
        f"{matches}"
    )


def prepare_runtime_runner(actual_embedding_key: str) -> Path:
    if not ORIGINAL_RUNNER.exists():
        raise FileNotFoundError(f"Original Step 08 runner not found: {ORIGINAL_RUNNER}")

    source = ORIGINAL_RUNNER.read_text(encoding="utf-8")
    expected_key = "model.embed_tokens.weight"
    count = source.count(expected_key)
    if count == 0:
        raise RuntimeError(
            "Could not find the original checkpoint validator key "
            f"{expected_key!r} in {ORIGINAL_RUNNER}"
        )

    runtime_source = source.replace(expected_key, actual_embedding_key)
    header = (
        "# AUTO-GENERATED BY scripts_qwen38/08_run_evaluation.py\n"
        "# The original scripts/08_run_evaluation.py remains unchanged.\n"
        f"# Qwen3.8 embedding key: {actual_embedding_key}\n\n"
    )
    RUNTIME_RUNNER.write_text(header + runtime_source, encoding="utf-8")
    print(
        "[QWEN3.8] Private validator adapter | "
        f"expected_key={expected_key} | actual_key={actual_embedding_key} | "
        f"replacements={count} | runtime_runner={RUNTIME_RUNNER}",
        flush=True,
    )
    return RUNTIME_RUNNER


def install_factory_override() -> None:
    import src.api.client_factory as factory

    original_get_client = factory.get_client

    def isolated_get_client(*args: Any, **kwargs: Any) -> Any:
        provider = str(
            kwargs.get("provider", args[0] if args else "")
        ).lower()
        model_name = str(
            kwargs.get("model_name", args[1] if len(args) > 1 else "")
        )
        normalized_name = model_name.lower().replace("_", "").replace("-", "")
        if provider == "huggingface" and "qwen3.827b" in normalized_name:
            return Qwen38LinuxClient(*args, **kwargs)
        return original_get_client(*args, **kwargs)

    factory.get_client = isolated_get_client


def normalized_argv(config_path: Path, runner_path: Path) -> list[str]:
    user_args = sys.argv[1:]
    result = [str(runner_path), *user_args]
    if "--config" not in user_args:
        result.extend(["--config", str(config_path)])
    if "--model" not in user_args and "--models" not in user_args:
        result.extend(["--model", "Ministral-3-14B"])
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
        config_path = prepare_runtime_files()
        model_path = get_qwen_model_path()
        embedding_key = find_embedding_key(model_path)
        runner_path = prepare_runtime_runner(embedding_key)
        install_factory_override()
        sys.argv = normalized_argv(config_path, runner_path)
        runpy.run_path(str(runner_path), run_name="__main__")
    finally:
        guard.stop()


if __name__ == "__main__":
    main()
