from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


def safe_divide(a: Any, b: Any) -> float | None:
    return None if b == 0 else float(a) / float(b)


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def normalize_boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return result
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            result.append(value)
    return result


def _atomic_write(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: str | Path, value: Any) -> None:
    _atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    _atomic_write(path, "\n".join(lines) + ("\n" if lines else ""))


def index_by_id(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("id", "")).strip()
        if not item_id:
            raise ValueError("Encountered a row without an id")
        if item_id in result:
            raise ValueError(f"Duplicate id: {item_id}")
        result[item_id] = row
    return result


def compare_id_sets(
    gold_ids: set[str],
    output_ids: set[str],
) -> dict[str, Any]:
    matched = gold_ids & output_ids
    missing = gold_ids - output_ids
    unexpected = output_ids - gold_ids
    return {
        "n_gold": len(gold_ids),
        "n_outputs": len(output_ids),
        "n_matched": len(matched),
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "coverage_rate": safe_divide(len(matched), len(gold_ids)),
        "missing_ids": sorted(missing),
        "unexpected_ids": sorted(unexpected),
    }


def common_ids(
    gold_ids: set[str],
    output_ids_by_model: Mapping[str, set[str]],
) -> set[str]:
    common = set(gold_ids)
    for model_ids in output_ids_by_model.values():
        common.intersection_update(model_ids)
    return common
