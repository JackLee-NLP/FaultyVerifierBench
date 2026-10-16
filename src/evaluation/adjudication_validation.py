"""Strict Step 08 parsed-output validation."""

from __future__ import annotations

from typing import Any, Iterable

REQUIRED_KEYS = {
    "verifier_judgment",
    "valid_verifier_points",
    "faulty_verifier_points",
    "fallacy_detected",
    "fallacy_explanations",
    "hallucination_detected",
    "hallucination_explanations",
    "should_revise_solver_answer",
    "final_answer",
    "justification",
}
VALID_JUDGMENTS = {"accept", "reject", "partially_accept"}


def _validate_explanations(
    values: Any,
    selected_labels: set[str],
    field_name: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(values, list):
        return [f"{field_name} must be a list"]
    for index, value in enumerate(values):
        if not isinstance(value, dict) or set(value) != {"type", "explanation"}:
            errors.append(
                f"{field_name}[{index}] must contain exactly type and explanation"
            )
            continue
        if value["type"] not in selected_labels:
            errors.append(
                f"{field_name}[{index}].type must also appear in its detected array"
            )
        if not isinstance(value["explanation"], str):
            errors.append(f"{field_name}[{index}].explanation must be a string")
    return errors


def validate_adjudication(
    parsed: dict[str, Any],
    allowed_fallacies: Iterable[str],
    allowed_hallucinations: Iterable[str],
) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_KEYS - set(parsed))
    if missing:
        errors.append(f"Missing required keys: {missing}")

    if parsed.get("verifier_judgment") not in VALID_JUDGMENTS:
        errors.append("Invalid verifier_judgment")
    if not isinstance(parsed.get("should_revise_solver_answer"), bool):
        errors.append("should_revise_solver_answer must be Boolean")
    if not isinstance(parsed.get("final_answer"), str):
        errors.append("final_answer must be a string")

    fallacies = parsed.get("fallacy_detected", [])
    hallucinations = parsed.get("hallucination_detected", [])
    if not isinstance(fallacies, list) or not all(isinstance(x, str) for x in fallacies):
        errors.append("fallacy_detected must be a list of strings")
        fallacies = []
    if not isinstance(hallucinations, list) or not all(
        isinstance(x, str) for x in hallucinations
    ):
        errors.append("hallucination_detected must be a list of strings")
        hallucinations = []

    unknown_fallacies = sorted(set(fallacies) - set(allowed_fallacies))
    unknown_hallucinations = sorted(set(hallucinations) - set(allowed_hallucinations))
    if unknown_fallacies:
        errors.append(f"Unknown fallacy labels: {unknown_fallacies}")
    if unknown_hallucinations:
        errors.append(f"Unknown hallucination labels: {unknown_hallucinations}")

    errors.extend(
        _validate_explanations(
            parsed.get("fallacy_explanations", []),
            set(fallacies),
            "fallacy_explanations",
        )
    )
    errors.extend(
        _validate_explanations(
            parsed.get("hallucination_explanations", []),
            set(hallucinations),
            "hallucination_explanations",
        )
    )
    return errors


def finalize_adjudication_record(
    parsed: dict[str, Any],
    allowed_fallacies: Iterable[str],
    allowed_hallucinations: Iterable[str],
) -> dict[str, Any]:
    """Attach validation fields before Step 08 writes the output record."""
    record = dict(parsed)
    errors = validate_adjudication(
        record,
        allowed_fallacies,
        allowed_hallucinations,
    )
    record["adjudication_parse_success"] = not errors
    record["adjudication_validation_errors"] = errors
    record["adjudication_schema_version"] = "2.0"
    return record
