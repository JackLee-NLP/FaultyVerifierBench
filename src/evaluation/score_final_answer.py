from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

NUMERIC_DOMAINS = {"math_reasoning", "data_interpretation", "code_reasoning"}
MATCHER_POLICIES = (
    "M0_exact",
    "M1_numeric",
    "M2_ordered_containment",
    "M3_guarded_containment",
    "M4_operational",
)
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
ALTERNATIVE_PATTERN = re.compile(r"\b(?:or|either|alternatively|instead|but|however)\b")
NEGATION_PATTERN = re.compile(r"\b(?:not|never|cannot|can't)\b")
UNIT_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
OPPOSITES = {"yes": "no", "no": "yes"}


def norm(value):
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,;:!?")


def toks(value):
    return TOKEN_PATTERN.findall(value)


def one_num(value):
    matches = NUMBER_PATTERN.findall(value)
    if len(matches) != 1:
        return None
    try:
        return float(matches[0].replace(",", ""))
    except ValueError:
        return None


def _is_subsequence(need, have):
    position = 0
    for token in have:
        if position < len(need) and token == need[position]:
            position += 1
    return position == len(need)


def _is_contiguous(need, have):
    if not need:
        return False
    width = len(need)
    return any(have[i:i + width] == need for i in range(len(have) - width + 1))


def _answer_risks(pred_tokens, gold_tokens):
    gold_set = set(gold_tokens)
    pred_set = set(pred_tokens)
    gold_head = gold_tokens[0] if gold_tokens else ""
    text = " ".join(pred_tokens)

    polarity_conflict = gold_head in OPPOSITES and OPPOSITES[gold_head] in pred_set
    alternatives = bool(ALTERNATIVE_PATTERN.search(text))
    both_polarities = "yes" in pred_set and "no" in pred_set
    negated = bool(NEGATION_PATTERN.search(text)) and gold_head in {"yes", "no"}
    negation_conflict = polarity_conflict or both_polarities or negated
    return {
        "gold_token_order_preserved": _is_subsequence(gold_tokens, pred_tokens) if gold_tokens else False,
        "gold_token_multiplicity_preserved": not (Counter(gold_tokens) - Counter(pred_tokens)),
        "extra_prediction_tokens": sorted(pred_set - gold_set),
        "possible_negation_conflict": negation_conflict,
        "alternative_answer_flag": alternatives,
        "both_polarities": both_polarities,
    }


def _numeric_match(pred_tokens, gold_tokens, domain):
    if domain not in NUMERIC_DOMAINS:
        return False, None, None, "not_numeric_domain"

    prediction = one_num(" ".join(pred_tokens))
    gold = one_num(" ".join(gold_tokens))
    if prediction is None or gold is None:
        return False, prediction, gold, "requires_one_number_each"
    if not math.isclose(prediction, gold, rel_tol=1e-6, abs_tol=1e-6):
        return False, prediction, gold, "numeric_values_differ"

    gold_units = [token for token in gold_tokens if not UNIT_PATTERN.fullmatch(token)]
    if not all(token in pred_tokens for token in gold_units):
        return False, prediction, gold, "incompatible_or_missing_units"
    return True, prediction, gold, "numeric_equivalence"


def _guard_names(pred_tokens, gold_tokens, risks):
    guards = []
    if risks["alternative_answer_flag"]:
        guards.append("alternative_answer")
    if risks["both_polarities"]:
        guards.append("both_polarities")
    if risks["possible_negation_conflict"]:
        guards.append("polarity_conflict")
    if not risks["gold_token_multiplicity_preserved"]:
        guards.append("gold_multiplicity_missing")

    pred_set = set(pred_tokens)
    tokens_present = bool(gold_tokens) and all(token in pred_set for token in gold_tokens)
    if tokens_present and not risks["gold_token_order_preserved"]:
        guards.append("gold_order_reversed")
    return guards


def _policy_outcome(matched, definite=False, guards=(), high_overlap=False):
    if definite and matched:
        return "definitely_matched"
    if matched or guards or high_overlap:
        return "unresolved"
    return "definitely_unmatched"


def _method_name(exact, numeric, special, special_name):
    if exact:
        return "normalized_exact"
    if numeric:
        return "numeric"
    if special:
        return special_name
    return "unmatched"


def _matcher_entry(matched, definite, method, guards=(), high_overlap=False):
    return {
        "matched": matched,
        "outcome": _policy_outcome(matched, definite, guards, high_overlap=high_overlap),
        "method": method,
        "guards": list(guards),
    }


def _token_overlap(pred_tokens, gold_tokens):
    if not pred_tokens or not gold_tokens:
        overlap = 0
    else:
        overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    precision = overlap / len(pred_tokens) if pred_tokens else None
    recall = overlap / len(gold_tokens) if gold_tokens else None
    if precision is None or recall is None or not precision + recall:
        f1 = 0.0 if precision is not None else None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score_final_answer(predicted_answer, gold_answer, domain, question):
    del question
    prediction = norm(predicted_answer)
    gold = norm(gold_answer)
    pred_tokens = toks(prediction)
    gold_tokens = toks(gold)
    pred_set = set(pred_tokens)

    precision, recall, f1 = _token_overlap(pred_tokens, gold_tokens)
    risks = _answer_risks(pred_tokens, gold_tokens)
    guards = _guard_names(pred_tokens, gold_tokens, risks)
    numeric, numeric_prediction, numeric_gold, numeric_reason = _numeric_match(pred_tokens, gold_tokens, domain)

    exact = bool(prediction and gold and prediction == gold)
    short_gold = 1 <= len(gold_tokens) <= 6
    ordered = short_gold and _is_contiguous(gold_tokens, pred_tokens)
    contained = short_gold and all(token in pred_set for token in gold_tokens)
    guarded = contained and not guards
    high_overlap = f1 is not None and f1 >= 0.6
    definite = exact or numeric

    matcher_results = {
        "M0_exact": _matcher_entry(
            exact,
            exact,
            "normalized_exact" if exact else "unmatched",
            high_overlap=high_overlap and not exact,
        ),
        "M1_numeric": _matcher_entry(
            exact or numeric,
            definite,
            _method_name(exact, numeric, False, "unmatched"),
            [] if definite else [numeric_reason],
            high_overlap=high_overlap and not definite,
        ),
        "M2_ordered_containment": _matcher_entry(
            exact or numeric or ordered,
            definite,
            "ordered_containment" if ordered and not definite else "inherited",
            guards,
            high_overlap=high_overlap and not (definite or ordered),
        ),
        "M3_guarded_containment": _matcher_entry(
            exact or numeric or guarded,
            definite,
            "guarded_containment" if guarded and not definite else "inherited",
            guards,
            high_overlap=high_overlap and not definite,
        ),
        "M4_operational": _matcher_entry(
            exact or numeric or contained,
            definite,
            "gold_token_containment" if contained and not definite else "inherited",
            guards,
            high_overlap=high_overlap and not definite,
        ),
    }
    if not prediction or not gold:
        for result in matcher_results.values():
            result.update(matched=False, outcome="definitely_unmatched", method="missing_answer")

    operational = matcher_results["M4_operational"]
    result = {
        "correct": operational["matched"],
        "status": "correct" if operational["matched"] else "incorrect",
        "method": operational["method"],
        "reason": "Deterministic M4 operational matching.",
        "normalized_prediction": prediction,
        "normalized_gold": gold,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
        "numeric_prediction": numeric_prediction,
        "numeric_gold": numeric_gold,
        "semantic_similarity": None,
        "semantic_similarity_threshold": None,
        "material_order_reversal": bool(pred_tokens and gold_tokens and not risks["gold_token_order_preserved"] and contained),
        "possible_negation_conflict": risks["possible_negation_conflict"],
        "gold_token_order_preserved": risks["gold_token_order_preserved"],
        "gold_token_multiplicity_preserved": risks["gold_token_multiplicity_preserved"],
        "extra_prediction_tokens": risks["extra_prediction_tokens"],
        "alternative_answer_flag": risks["alternative_answer_flag"],
        "both_polarities": risks["both_polarities"],
        "matcher_guards": guards,
        "matcher_results": matcher_results,
        "needs_semantic_review": bool(f1 and f1 >= 0.6),
        "numeric_reason": numeric_reason,
    }
    if not prediction or not gold:
        result.update(method="missing_answer", reason="Prediction or gold is empty.")
    return result
