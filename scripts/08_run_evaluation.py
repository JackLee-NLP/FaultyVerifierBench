#!/usr/bin/env python3
"""FaultyVerifierBench evaluation runner."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import yaml
from safetensors import safe_open

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from src.evaluation.run_adjudication import auto_run_adjudication

LOCAL_MODEL_PROVIDERS = {"huggingface"}

def _yaml(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f: value = yaml.safe_load(f) or {}
    if not isinstance(value, dict): raise TypeError(f"Expected YAML mapping in {path}")
    return value

def _json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f: value = json.load(f)
    if not isinstance(value, dict): raise TypeError(f"Expected JSON object in {path}")
    return value

def _path(value: str | Path, base: Path) -> Path:
    p = Path(value).expanduser(); return (p if p.is_absolute() else base / p).resolve()

def _models_path(config_path: Path, explicit: Optional[str]) -> Path:
    return _path(explicit, Path.cwd()) if explicit else _path(_yaml(config_path).get("evaluation", {}).get("models_path", "config/models.yaml"), config_path.parent.parent)

def _models(path: Path):
    models = _yaml(path).get("models")
    if not isinstance(models, dict): raise TypeError(f"Expected models mapping in {path}")
    return models

def _active(config_path: Path, models, explicit: Optional[str]) -> list[str]:
    if explicit: keys = [explicit.strip()]
    else:
        evaluation = _yaml(config_path).get("evaluation", {})
        keys = list(evaluation.get("target_models") or [evaluation.get("default_model")])
    missing = [key for key in keys if key not in models]
    if missing: raise ValueError(f"Unknown models: {missing}")
    return keys

def _local(active: Iterable[str], models):
    return {key: _path(models[key]["model_name"], PROJECT_ROOT) for key in active if str(models[key].get("provider", "")).lower() in LOCAL_MODEL_PROVIDERS}

def _inspect(path: Path):
    found = {}; targets = {"lm_head.weight", "model.embed_tokens.weight"}
    shards = sorted(path.glob("*.safetensors"))
    if not shards: raise FileNotFoundError(f"No safetensors files in {path}")
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as h:
            for key in targets & set(h.keys()):
                if key not in found: found[key] = list(h.get_tensor(key).shape)
    return found

def _check(key: str, path: Path):
    if not path.is_dir(): raise FileNotFoundError(path)
    config = _json(path / "config.json"); found = _inspect(path)
    if "model.embed_tokens.weight" not in found: raise RuntimeError("model.embed_tokens.weight is missing")
    if "lm_head.weight" not in found and not bool(config.get("tie_word_embeddings", False)):
        raise RuntimeError("lm_head.weight is missing while tie_word_embeddings=False")
    print(f"[CHECKPOINT] model={key} validated at {path}", flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model"); p.add_argument("--split"); p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--models"); p.add_argument("--input"); p.add_argument("--output")
    p.add_argument("--workers-per-model", type=int); p.add_argument("--force-full", action="store_true")
    a = p.parse_args()
    try:
        config_path = _path(a.config, Path.cwd()); models_path = _models_path(config_path, a.models)
        models = _models(models_path); active = _active(config_path, models, a.model); local = _local(active, models)
        for key, path in local.items(): _check(key, path)
        e = _yaml(config_path).get("evaluation", {})
        print(f"[SCRIPT] Active models: {active}", flush=True)
        print(f"[SCRIPT] Local models: {list(local)}", flush=True)
        print(f"[SCRIPT] Local batch size: {e.get('local_batch_size', 4)}", flush=True)
        print(f"[SCRIPT] Local max batch tokens: {e.get('local_max_batch_tokens', 16384)}", flush=True)
    except Exception as error:
        p.error(f"Configuration validation failed: {type(error).__name__}: {error}")
    auto_run_adjudication(model_key=a.model, split=a.split, models_path=str(models_path), config_path=str(config_path), input_path=a.input, output_path=a.output, force_full_regenerate=a.force_full, workers_per_model=a.workers_per_model)

if __name__ == "__main__": main()
