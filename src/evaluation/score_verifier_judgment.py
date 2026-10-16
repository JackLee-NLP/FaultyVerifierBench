from __future__ import annotations

from typing import Any

from .metric_utils import normalize_label

ALIASES = {
    "accept": "accept",
    "accepted": "accept",
    "agree": "accept",
    "accept_verifier": "accept",
    "accept_feedback": "accept",
    "accept_verifier_feedback": "accept",
    "verifier_correct": "accept",
    "correct": "accept",
    "reject_solver": "accept",
    "revise_solver": "accept",
    "revise_solver_answer": "accept",
    "should_revise": "accept",
    "reject": "reject",
    "rejected": "reject",
    "disagree": "reject",
    "reject_verifier": "reject",
    "reject_feedback": "reject",
    "reject_verifier_feedback": "reject",
    "verifier_faulty": "reject",
    "verifier_incorrect": "reject",
    "faulty": "reject",
    "keep_solver": "reject",
    "maintain_solver": "reject",
    "maintain_solver_answer": "reject",
    "do_not_revise": "reject",
    "partially_accept": "partially_accept",
    "partial_accept": "partially_accept",
    "partially_agree": "partially_accept",
    "mixed": "partially_accept",
    "mixed_feedback": "partially_accept",
}


def normalize_judgment(value: Any):
    return ALIASES.get(normalize_label(value), "invalid")


def score_verifier_judgment(predicted_judgment, gold_judgment):
    predicted = normalize_judgment(predicted_judgment)
    gold = normalize_judgment(gold_judgment)
    return {
        "correct": predicted == gold and predicted != "invalid",
        "predicted_label": predicted,
        "gold_label": gold,
        "raw_predicted_label": "" if predicted_judgment is None else str(predicted_judgment),
        "raw_gold_label": "" if gold_judgment is None else str(gold_judgment),
    }
