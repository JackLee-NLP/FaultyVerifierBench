from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.dataset.io_utils import read_jsonl, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "verifier_raw.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "verifier_validated.jsonl"
DEFAULT_REJECTED_PATH = PROJECT_ROOT / "data" / "processed" / "verifier_rejected.jsonl"
DEFAULT_FALLACIES_PATH = PROJECT_ROOT / "config" / "fallacies.yaml"
DEFAULT_HALLUCINATIONS_PATH = PROJECT_ROOT / "config" / "hallucinations.yaml"

ALLOWED_VERIFIER_CORRECTNESS = {"correct", "faulty", "mixed"}
ALLOWED_FAULT_CATEGORIES = {"fallacy", "hallucination"}
EMPTY_LABELS = {"", "none", "null", "n/a", "na", "not_applicable", "no_fault"}

# Keys that describe YAML structure rather than taxonomy labels.
STRUCTURAL_YAML_KEYS = {
    "name", "label", "id", "key", "type", "types", "items", "values", "labels",
    "taxonomy", "taxonomies", "fallacies", "hallucinations", "categories", "definitions",
    "description", "definition", "examples", "example", "aliases", "metadata", "version",
}


def _normalise_label(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _as_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        output: list[str] = []
        for part in value:
            output.extend(_as_labels(part))
        return list(dict.fromkeys(output))
    if isinstance(value, dict):
        output = []
        for key, enabled in value.items():
            if enabled is True:
                output.append(_normalise_label(key))
            elif isinstance(enabled, (str, list, tuple, set, dict)):
                output.extend(_as_labels(enabled))
        return list(dict.fromkeys(x for x in output if x not in EMPTY_LABELS))

    text = str(value).strip()
    if not text:
        return []
    parts = text.split("|") if "|" in text else text.split(",") if "," in text else [text]
    return list(
        dict.fromkeys(
            label for part in parts
            if (label := _normalise_label(part)) not in EMPTY_LABELS
        )
    )


def _extract_taxonomy_labels(node: Any) -> set[str]:
    """Support lists, mappings keyed by label, and lists of name/id records."""
    labels: set[str] = set()

    if isinstance(node, str):
        label = _normalise_label(node)
        if label not in EMPTY_LABELS:
            labels.add(label)
        return labels

    if isinstance(node, list):
        for item in node:
            if isinstance(item, dict):
                explicit = next(
                    (item.get(key) for key in ("name", "label", "id", "key", "type") if item.get(key)),
                    None,
                )
                if explicit is not None:
                    labels.update(_as_labels(explicit))
                else:
                    labels.update(_extract_taxonomy_labels(item))
            else:
                labels.update(_extract_taxonomy_labels(item))
        return labels

    if isinstance(node, dict):
        # Prefer well-known taxonomy containers when present.
        container_keys = ("fallacies", "hallucinations", "taxonomy", "types", "items", "values", "labels")
        found_container = False
        for key in container_keys:
            if key in node:
                labels.update(_extract_taxonomy_labels(node[key]))
                found_container = True
        if found_container:
            return labels

        # Otherwise treat non-structural mapping keys as taxonomy labels.
        for key, value in node.items():
            normalised_key = _normalise_label(key)
            if normalised_key not in STRUCTURAL_YAML_KEYS:
                labels.add(normalised_key)
            elif isinstance(value, (dict, list)):
                labels.update(_extract_taxonomy_labels(value))
        return labels

    return labels


def _load_taxonomy(path: Path, taxonomy_name: str) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"{taxonomy_name} taxonomy file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    labels = _extract_taxonomy_labels(data)
    if not labels:
        raise ValueError(f"No {taxonomy_name} labels could be extracted from: {path}")
    return labels


def _categorise_labels(item: dict[str, Any]) -> tuple[set[str], set[str], list[str]]:
    categories = set(_as_labels(item.get("fault_category")))
    generic_fault_types = set(_as_labels(item.get("fault_type")))
    fallacy_types = set(_as_labels(item.get("fallacy_type")))
    hallucination_types = set(_as_labels(item.get("hallucination_type")))
    errors: list[str] = []

    unknown_categories = categories - ALLOWED_FAULT_CATEGORIES
    if unknown_categories:
        errors.append(f"unknown fault_category: {sorted(unknown_categories)}")

    # Current schema stores fallacy labels in fault_type. For hallucination records,
    # accept either fault_type or hallucination_type. Mixed records may use both.
    if "fallacy" in categories:
        fallacy_types.update(generic_fault_types)
    if "hallucination" in categories:
        hallucination_types.update(generic_fault_types)

    return fallacy_types, hallucination_types, errors


def _validate_item(
    item: dict[str, Any],
    allowed_fallacies: set[str],
    allowed_hallucinations: set[str],
) -> list[str]:
    errors: list[str] = []

    if not str(item.get("verifier_response", "")).strip():
        errors.append("missing verifier_response")

    correctness = _normalise_label(item.get("verifier_correctness", ""))
    if correctness not in ALLOWED_VERIFIER_CORRECTNESS:
        errors.append(
            "invalid verifier_correctness: "
            f"{item.get('verifier_correctness')!r}; allowed={sorted(ALLOWED_VERIFIER_CORRECTNESS)}"
        )

    categories = set(_as_labels(item.get("fault_category")))
    fallacy_types, hallucination_types, category_errors = _categorise_labels(item)
    errors.extend(category_errors)

    invalid_fallacies = fallacy_types - allowed_fallacies
    invalid_hallucinations = hallucination_types - allowed_hallucinations
    if invalid_fallacies:
        errors.append(f"unrecognised fallacy label(s): {sorted(invalid_fallacies)}")
    if invalid_hallucinations:
        errors.append(f"unrecognised hallucination label(s): {sorted(invalid_hallucinations)}")

    if correctness == "correct" and (categories or fallacy_types or hallucination_types):
        errors.append("correct verifier must not contain fault labels")

    if correctness in {"faulty", "mixed"}:
        if not categories:
            errors.append("faulty/mixed verifier requires fault_category")
        if "fallacy" in categories and not fallacy_types:
            errors.append("fallacy category requires at least one fallacy label")
        if "hallucination" in categories and not hallucination_types:
            errors.append("hallucination category requires at least one hallucination label")

    return errors


def validate_verifier(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    rejected_path: str | Path = DEFAULT_REJECTED_PATH,
    fallacies_path: str | Path = DEFAULT_FALLACIES_PATH,
    hallucinations_path: str | Path = DEFAULT_HALLUCINATIONS_PATH,
) -> list[dict[str, Any]]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    rejected_path = Path(rejected_path)
    fallacies_path = Path(fallacies_path)
    hallucinations_path = Path(hallucinations_path)

    allowed_fallacies = _load_taxonomy(fallacies_path, "fallacy")
    allowed_hallucinations = _load_taxonomy(hallucinations_path, "hallucination")

    validated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    total = 0

    for source_item in read_jsonl(input_path):
        total += 1
        item = dict(source_item)
        errors = _validate_item(item, allowed_fallacies, allowed_hallucinations)

        if errors:
            item["verifier_validation_status"] = "rejected"
            item["verifier_validation_errors"] = errors
            rejected.append(item)
            rejection_reasons.update(errors)
        else:
            item["verifier_validation_status"] = "validated"
            item.pop("verifier_validation_errors", None)
            validated.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(validated, output_path)
    write_jsonl(rejected, rejected_path)

    print(f"[VALIDATION] Input: {input_path}", flush=True)
    print(f"[VALIDATION] Allowed fallacies: {len(allowed_fallacies)}", flush=True)
    print(f"[VALIDATION] Allowed hallucinations: {len(allowed_hallucinations)}", flush=True)
    print(f"[VALIDATION] Total items: {total}", flush=True)
    print(f"[VALIDATION] Validated items: {len(validated)}", flush=True)
    print(f"[VALIDATION] Rejected items: {len(rejected)}", flush=True)
    print(f"[VALIDATION] Validated output: {output_path}", flush=True)
    print(f"[VALIDATION] Rejected output: {rejected_path}", flush=True)

    if rejection_reasons:
        print("[VALIDATION] Rejection reasons:", flush=True)
        for reason, count in rejection_reasons.most_common():
            print(f"  {count:8d}  {reason}", flush=True)

    return validated
