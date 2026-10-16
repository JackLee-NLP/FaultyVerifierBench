from __future__ import annotations

from collections.abc import Iterable

from .metric_utils import normalize_label

IGNORED_LABELS = {"", "none", "no_fault"}


def normalize_taxonomy_values(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Iterable) or isinstance(value, (bytes, dict)):
        value = [value]

    labels = {normalize_label(item) for item in value}
    labels -= IGNORED_LABELS
    return sorted(labels)


def score_fault_detection(predicted_fallacies, predicted_hallucinations, gold_fallacies, gold_hallucinations):
    predicted_fallacies = normalize_taxonomy_values(predicted_fallacies)
    predicted_hallucinations = normalize_taxonomy_values(predicted_hallucinations)
    gold_fallacies = normalize_taxonomy_values(gold_fallacies)
    gold_hallucinations = normalize_taxonomy_values(gold_hallucinations)

    fallacy_exact = set(predicted_fallacies) == set(gold_fallacies)
    hallucination_exact = set(predicted_hallucinations) == set(gold_hallucinations)
    return {
        "exact_fault_detection_correct": fallacy_exact and hallucination_exact,
        "fallacy_exact": fallacy_exact,
        "hallucination_exact": hallucination_exact,
        "predicted_fallacies": predicted_fallacies,
        "predicted_hallucinations": predicted_hallucinations,
        "gold_fallacies": gold_fallacies,
        "gold_hallucinations": gold_hallucinations,
    }
