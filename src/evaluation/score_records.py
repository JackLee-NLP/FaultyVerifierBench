"""Canonical Step 08 to Step 09 score-row conversion (evaluation schema 3.1)."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Iterable

from .metric_utils import normalize_boolean, normalize_label
from .score_fault_detection import score_fault_detection
from .score_final_answer import score_final_answer
from .score_verifier_judgment import normalize_judgment, score_verifier_judgment

CONTAINERS = (
    "parsed_output",
    "parsed_response",
    "parsed_adjudication",
    "adjudication",
    "prediction",
    "response",
    "result",
    "output",
    "model_output",
)
ALIASES = {
    "hallucinated_citation": "fabricated_citation",
    "fabricated_reference": "fabricated_citation",
    "fabricated_source": "fabricated_citation",
    "solver_response_misrepresentation": "misquoted_solver_response",
    "misquoting_solver": "misquoted_solver_response",
    "fabricated_constraint": "unsupported_constraint",
    "fabricated_observation": "fabricated_environment_observation",
    "arithmetic_error": "incorrect_calculation",
}
CASE_BY_CORRECTNESS = {
    ("correct", "correct"): "C1",
    ("correct", "faulty"): "C2",
    ("incorrect", "correct"): "C3",
    ("incorrect", "faulty"): "C4",
}
BASE_ID_KEYS = (
    "base_question_id",
    "source_question_id",
    "question_id",
    "base_id",
    "source_id",
    "parent_id",
)
BASE_ID_PATTERNS = (
    r"^(.*?)(?:_C[1-4])(?:_|$)",
    r"^(.*?)(?:_case[1-4])(?:_|$)",
    r"^(.*?)(?:__(?:C[1-4]))(?:__|$)",
)
EMBEDDED_TEXT_KEYS = (
    "raw_response",
    "response_text",
    "raw_output",
    "completion",
    "content",
    "generated_text",
)
ANSWER_FIELDS = (
    "final_answer",
    "predicted_final_answer",
    "answer",
    "adjudicated_final_answer",
)
JUDGMENT_FIELDS = (
    "verifier_judgment",
    "verifier_assessment",
    "judgment",
    "verifier_decision",
    "decision",
)
FALLACY_FIELDS = (
    "fallacy_detected",
    "fallacies_detected",
    "detected_fallacies",
    "fallacy_types",
    "pred_fallacy_types",
)
HALLUCINATION_FIELDS = (
    "hallucination_detected",
    "hallucinations_detected",
    "detected_hallucinations",
    "hallucination_types",
    "pred_hallucination_types",
)
REVISION_FIELDS = (
    "should_revise_solver_answer",
    "should_revise",
    "revise_solver_answer",
    "revision_required",
    "needs_revision",
)
PARSE_FIELDS = (
    "adjudication_parse_success",
    "parse_success",
    "parsed_successfully",
    "parsing_success",
)
JUSTIFICATION_FIELDS = ("justification", "reasoning", "explanation")
GOLD_REVISION_FIELDS = (
    "should_revise_solver_answer",
    "gold_should_revise_solver_answer",
    "expected_should_revise_solver_answer",
    "revision_required",
)
FINAL_ANSWER_FIELDS = (
    "status",
    "method",
    "reason",
    "normalized_prediction",
    "normalized_gold",
    "token_precision",
    "token_recall",
    "token_f1",
    "numeric_prediction",
    "numeric_gold",
    "semantic_similarity",
    "semantic_similarity_threshold",
    "material_order_reversal",
    "possible_negation_conflict",
    "gold_token_order_preserved",
    "gold_token_multiplicity_preserved",
    "extra_prediction_tokens",
    "alternative_answer_flag",
    "both_polarities",
    "matcher_guards",
    "matcher_results",
    "needs_semantic_review",
    "numeric_reason",
)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _lookup(record, path):
    value = record
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _paths(*names):
    nested = [f"{container}.{name}" for container in CONTAINERS for name in names]
    return list(names) + nested


def _first_path(record, paths, default=None):
    for path in paths:
        value = _lookup(record, path)
        if value is not None:
            return value, path
    return default, None


def _first(record, paths, default=None):
    return _first_path(record, paths, default)[0]


def _embedded(record):
    expanded = dict(record)
    for key in EMBEDDED_TEXT_KEYS:
        value = record.get(key)
        if not isinstance(value, str):
            continue
        text = value.strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else None
        if isinstance(parsed, dict):
            expanded.setdefault("embedded", parsed)
    return expanded


def _canonical(value):
    label = normalize_label(value)
    return ALIASES.get(label, label)


def _split_label(item, allowed):
    if not isinstance(item, str):
        return [item]

    parts = [part.strip() for part in re.split(r"[,;|\n]+", item) if part.strip()]
    if len(parts) != 1 or not re.search(r"\s+(?:and|&)\s+", item, re.I):
        return parts

    candidates = [part.strip() for part in re.split(r"\s+(?:and|&)\s+", item, flags=re.I) if part.strip()]
    if candidates and all(_canonical(part) in allowed for part in candidates):
        return candidates
    return parts


def _tax(value, allowed=()):
    allowed = set(allowed)
    tokens = []
    aliases = []
    for item in _as_list(value):
        if item is None:
            continue
        for part in _split_label(item, allowed):
            original = normalize_label(part)
            mapped = ALIASES.get(original, original)
            if not original or not mapped:
                continue
            tokens.append(mapped)
            if mapped != original:
                aliases.append({"raw_normalized": original, "canonical": mapped})
    return sorted(set(tokens)), aliases


def _case(solver, verifier):
    return CASE_BY_CORRECTNESS.get((solver, verifier), "unknown")


def _gold_judgment(verifier):
    if verifier == "correct":
        return "accept"
    if verifier == "faulty":
        return "reject"
    raise ValueError(f"Unsupported verifier_correctness: {verifier!r}")


def _base_id(gold_item):
    for key in BASE_ID_KEYS:
        value = gold_item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip(), False

    item_id = str(gold_item.get("id", "")).strip()
    for pattern in BASE_ID_PATTERNS:
        match = re.match(pattern, item_id, flags=re.I)
        if match and match.group(1):
            return match.group(1), True
    return item_id, True


def _source_container(selected_path):
    if selected_path and "." in selected_path:
        return selected_path.split(".", 1)[0]
    return "flat_record"


def extract_step08_prediction(record, allowed_fallacies=(), allowed_hallucinations=()):
    record = _embedded(record)
    answer, answer_path = _first_path(
        record,
        _paths(*ANSWER_FIELDS) + ["embedded.final_answer", "embedded.answer"],
        "",
    )
    judgment, judgment_path = _first_path(
        record,
        _paths(*JUDGMENT_FIELDS) + ["embedded.verifier_judgment", "embedded.judgment"],
        "",
    )
    raw_fallacies, fallacy_path = _first_path(record, _paths(*FALLACY_FIELDS), [])
    raw_hallucinations, hallucination_path = _first_path(record, _paths(*HALLUCINATION_FIELDS), [])
    fallacies, fallacy_aliases = _tax(raw_fallacies, allowed_fallacies)
    hallucinations, hallucination_aliases = _tax(raw_hallucinations, allowed_hallucinations)

    revision = normalize_boolean(_first(record, _paths(*REVISION_FIELDS), None))
    success = normalize_boolean(_first(record, _paths(*PARSE_FIELDS), None))
    if success is None:
        success = bool(str(answer).strip() or str(judgment).strip() or fallacies or hallucinations)

    selected = judgment_path or answer_path or fallacy_path or hallucination_path
    container = _source_container(selected)
    return {
        "final_answer": answer,
        "judgment": judgment,
        "raw_fallacies": _as_list(raw_fallacies),
        "fallacies": fallacies,
        "raw_hallucinations": _as_list(raw_hallucinations),
        "hallucinations": hallucinations,
        "taxonomy_aliases_applied": {
            "fallacies": fallacy_aliases,
            "hallucinations": hallucination_aliases,
        },
        "revision": revision,
        "justification": str(_first(record, _paths(*JUSTIFICATION_FIELDS), "") or ""),
        "parse_success": success,
        "schema_variant": "embedded_json" if container == "embedded" else "structured_record",
        "source_container": container,
    }


def _known_labels(values, allowed):
    return sorted(label for label in values if label in allowed)


def _unknown_labels(values, allowed):
    return sorted(label for label in values if label not in allowed)


def build_score_row(
    gold_item,
    model_output,
    model_key,
    target_model_name=None,
    allowed_fallacies: Iterable[str] = (),
    allowed_hallucinations: Iterable[str] = (),
):
    if str(gold_item["id"]) != str(model_output.get("id", "")):
        raise ValueError("Mismatched IDs")

    fallacy_labels = set(allowed_fallacies)
    hallucination_labels = set(allowed_hallucinations)
    all_labels = fallacy_labels | hallucination_labels
    domain = normalize_label(gold_item.get("domain")) or "unknown"
    solver = normalize_label(gold_item.get("solver_correctness")) or "unknown"
    verifier = normalize_label(gold_item.get("verifier_correctness")) or "unknown"
    case = str(gold_item.get("case_type") or _case(solver, verifier)).upper()
    derived_case = _case(solver, verifier)
    if derived_case != "unknown" and case != derived_case:
        raise ValueError(
            f"Case mismatch for {gold_item['id']}: stored={case}, derived={derived_case}"
        )

    prediction = extract_step08_prediction(model_output, fallacy_labels, hallucination_labels)
    gold_answer = gold_item.get("expected_final_answer", gold_item.get("gold_answer", ""))
    stored_judgment = gold_item.get(
        "expected_verifier_judgment",
        gold_item.get("gold_verifier_judgment", gold_item.get("verifier_judgment", "")),
    )
    stored_judgment_norm = normalize_judgment(stored_judgment)
    gold_judgment = _gold_judgment(verifier)
    gold_disagreement = stored_judgment_norm != "invalid" and stored_judgment_norm != gold_judgment

    raw_faults = _as_list(gold_item.get("fault_type", []))
    raw_fallacies = _as_list(gold_item.get("fallacy_type", gold_item.get("fallacy_types", [])))
    raw_hallucinations = _as_list(gold_item.get("hallucination_type", gold_item.get("hallucination_types", [])))
    normalized_faults, _ = _tax(raw_faults, all_labels)
    normalized_fallacies, _ = _tax(raw_fallacies, fallacy_labels)
    normalized_hallucinations, _ = _tax(raw_hallucinations, hallucination_labels)

    gold_fallacies = sorted({label for label in normalized_faults + normalized_fallacies if label in fallacy_labels})
    gold_hallucinations = sorted({
        label for label in normalized_faults + normalized_hallucinations if label in hallucination_labels
    })
    unknown_gold = sorted({
        label for label in normalized_faults + normalized_fallacies + normalized_hallucinations
        if label not in all_labels
    })
    predicted_fallacies = _known_labels(prediction["fallacies"], fallacy_labels)
    predicted_hallucinations = _known_labels(prediction["hallucinations"], hallucination_labels)
    unknown_fallacies = _unknown_labels(prediction["fallacies"], fallacy_labels)
    unknown_hallucinations = _unknown_labels(prediction["hallucinations"], hallucination_labels)

    final = score_final_answer(prediction["final_answer"], gold_answer, domain, gold_item.get("question", ""))
    judgment = score_verifier_judgment(prediction["judgment"], gold_judgment)
    fault = score_fault_detection(predicted_fallacies, predicted_hallucinations, gold_fallacies, gold_hallucinations)
    parsed = bool(prediction["parse_success"])
    if not parsed:
        final.update(
            correct=False,
            status="incorrect",
            method="adjudication_parse_failure",
            reason="No recoverable parsed adjudication.",
        )
        judgment["correct"] = False
        fault["exact_fault_detection_correct"] = False

    gold_revision = normalize_boolean(_first(gold_item, GOLD_REVISION_FIELDS, None))
    if gold_revision is None and solver in {"correct", "incorrect"}:
        gold_revision = solver == "incorrect"
    predicted_revision = prediction["revision"]
    base_id, base_fallback = _base_id(gold_item)

    row = {
        "evaluation_schema_version": "3.1",
        "id": str(gold_item["id"]),
        "base_question_id": base_id,
        "base_question_id_fallback": base_fallback,
        "target_model": model_key,
        "target_model_name": target_model_name or model_key,
        "domain": domain,
        "case_type": case,
        "question": str(gold_item.get("question", "")),
        "solver_correctness": solver,
        "verifier_correctness": verifier,
        "fault_type": gold_fallacies,
        "hallucination_type": gold_hallucinations,
        "unknown_gold_taxonomy_labels": unknown_gold,
        "raw_pred_fallacy_types": prediction["raw_fallacies"],
        "raw_pred_hallucination_types": prediction["raw_hallucinations"],
        "taxonomy_aliases_applied": prediction["taxonomy_aliases_applied"],
        "schema_variant": prediction["schema_variant"],
        "source_container": prediction["source_container"],
        "pred_justification": prediction["justification"],
        "pred_final_answer": "" if prediction["final_answer"] is None else str(prediction["final_answer"]),
        "gold_answer": "" if gold_answer is None else str(gold_answer),
        "pred_verifier_judgment": judgment["predicted_label"],
        "gold_verifier_judgment": gold_judgment,
        "gold_verifier_judgment_source": "derived_from_verifier_correctness",
        "stored_gold_verifier_judgment": stored_judgment_norm,
        "raw_stored_gold_verifier_judgment": "" if stored_judgment is None else str(stored_judgment),
        "gold_verifier_judgment_disagreement": gold_disagreement,
        "raw_pred_verifier_judgment": judgment["raw_predicted_label"],
        "raw_gold_verifier_judgment": gold_judgment,
        "pred_fallacy_types": fault["predicted_fallacies"],
        "pred_hallucination_types": fault["predicted_hallucinations"],
        "unknown_predicted_fallacy_labels": unknown_fallacies,
        "unknown_predicted_hallucination_labels": unknown_hallucinations,
        "pred_should_revise_solver_answer": predicted_revision,
        "gold_should_revise_solver_answer": gold_revision,
        "solver_revision_correct": (
            predicted_revision is not None and gold_revision is not None and predicted_revision == gold_revision
        ),
        "gold_has_any_fault": bool(gold_fallacies or gold_hallucinations),
        "pred_has_any_fault": bool(predicted_fallacies or predicted_hallucinations) if parsed else False,
        "gold_has_fallacy": bool(gold_fallacies),
        "pred_has_fallacy": bool(predicted_fallacies) if parsed else False,
        "gold_has_hallucination": bool(gold_hallucinations),
        "pred_has_hallucination": bool(predicted_hallucinations) if parsed else False,
        "final_answer_correct": bool(final["correct"]),
        "verifier_judgment_correct": bool(judgment["correct"]),
        "fault_detection_correct": bool(fault["exact_fault_detection_correct"]),
        "adjudication_parse_success": parsed,
        "unsupported_judgment_prediction": judgment["predicted_label"] not in {"accept", "reject"},
    }
    for key in FINAL_ANSWER_FIELDS:
        row["final_answer_" + key] = final[key]
    for policy, result in final["matcher_results"].items():
        row[f"final_answer_{policy}_matched"] = bool(result["matched"])
        row[f"final_answer_{policy}_outcome"] = result["outcome"]
    return row
