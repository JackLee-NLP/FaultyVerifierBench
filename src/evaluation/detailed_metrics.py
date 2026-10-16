"""Pure aggregation over canonical FaultyVerifierBench Step 09 score rows."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

from .metric_utils import safe_divide

JUDGMENTS = ("accept", "reject")
MATCHER_POLICIES = (
    "M0_exact",
    "M1_numeric",
    "M2_ordered_containment",
    "M3_guarded_containment",
    "M4_operational",
)
OUTCOME_STATUSES = ("definitely_matched", "definitely_unmatched", "unresolved")
CASE_BY_CORRECTNESS = {
    ("correct", "correct"): "C1",
    ("correct", "faulty"): "C2",
    ("incorrect", "correct"): "C3",
    ("incorrect", "faulty"): "C4",
}


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return safe_divide(2.0 * precision * recall, precision + recall)


def _rate(rows: Sequence[dict[str, Any]], predicate) -> float | None:
    return safe_divide(sum(bool(predicate(row)) for row in rows), len(rows))


def _label_report(rows, labels, gold_field, pred_field):
    labels = tuple(labels)
    per_label = {}
    tp_all = fp_all = fn_all = 0
    macro_p, macro_r, macro_f1 = [], [], []
    exact = sum(
        row.get("adjudication_parse_success") is True
        and set(row.get(gold_field) or []) == set(row.get(pred_field) or [])
        for row in rows
    )
    positive = [row for row in rows if row.get(gold_field)]
    positive_exact = sum(
        row.get("adjudication_parse_success") is True
        and set(row.get(gold_field) or []) == set(row.get(pred_field) or [])
        for row in positive
    )
    for label in labels:
        tp = sum(label in (row.get(gold_field) or []) and label in (row.get(pred_field) or []) for row in rows)
        fp = sum(label not in (row.get(gold_field) or []) and label in (row.get(pred_field) or []) for row in rows)
        fn = sum(label in (row.get(gold_field) or []) and label not in (row.get(pred_field) or []) for row in rows)
        tn = len(rows) - tp - fp - fn
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = _f1(precision, recall)
        support = tp + fn
        per_label[label] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "support": support,
            "precision": precision,
            "recall": recall,
            "F1": f1,
        }
        tp_all += tp
        fp_all += fp
        fn_all += fn
        if support:
            macro_p.append(0.0 if precision is None else precision)
            macro_r.append(0.0 if recall is None else recall)
            macro_f1.append(0.0 if f1 is None else f1)
    precision = safe_divide(tp_all, tp_all + fp_all)
    recall = safe_divide(tp_all, tp_all + fn_all)
    return {
        "per_label": per_label,
        "summary": {
            "exact_set_match": safe_divide(exact, len(rows)),
            "positive_support_exact_set_match": safe_divide(positive_exact, len(positive)),
            "positive_support_N": len(positive),
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_F1": _f1(precision, recall),
            "macro_precision": safe_divide(sum(macro_p), len(macro_p)),
            "macro_recall": safe_divide(sum(macro_r), len(macro_r)),
            "macro_F1": safe_divide(sum(macro_f1), len(macro_f1)),
            "macro_supported_labels": len(macro_f1),
        },
    }


def _single_class(rows, labels, gold, pred):
    labels = tuple(labels)
    all_labels = labels + ("unsupported", "invalid")
    matrix = {gold_label: {pred_label: 0 for pred_label in all_labels} for gold_label in all_labels}
    for row in rows:
        gold_value = row.get(gold, "invalid")
        pred_value = row.get(pred, "invalid")
        if gold_value not in matrix:
            gold_value = "unsupported"
        if pred_value not in matrix[gold_value]:
            pred_value = "unsupported"
        matrix[gold_value][pred_value] += 1

    per_label = {}
    precisions, recalls, f1s = [], [], []
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in matrix if other != label)
        fn = sum(matrix[label][other] for other in matrix[label] if other != label)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = _f1(precision, recall)
        support = sum(matrix[label].values())
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "support": support,
        }
        if support:
            precisions.append(0.0 if precision is None else precision)
            recalls.append(0.0 if recall is None else recall)
            f1s.append(0.0 if f1 is None else f1)

    rates = {}
    for gold_label, counts in matrix.items():
        total = sum(counts.values())
        rates[gold_label] = {pred_label: safe_divide(count, total) for pred_label, count in counts.items()}
    return {
        "accuracy": safe_divide(sum(matrix[label][label] for label in labels), len(rows)),
        "macro_precision": safe_divide(sum(precisions), len(precisions)),
        "macro_recall": safe_divide(sum(recalls), len(recalls)),
        "macro_F1": safe_divide(sum(f1s), len(f1s)),
        "macro_supported_classes": len(f1s),
        "per_label": per_label,
        "confusion_matrix": {"counts": matrix, "row_normalized_rates": rates},
    }


def _binary_report(rows, gold_field, pred_field):
    tp = sum(bool(row.get(gold_field)) and bool(row.get(pred_field)) for row in rows)
    fp = sum((not bool(row.get(gold_field))) and bool(row.get(pred_field)) for row in rows)
    fn = sum(bool(row.get(gold_field)) and (not bool(row.get(pred_field))) for row in rows)
    tn = len(rows) - tp - fp - fn
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "accuracy": safe_divide(tp + tn, len(rows)),
        "precision": precision,
        "recall": recall,
        "F1": _f1(precision, recall),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "positive_support": tp + fn,
        "negative_support": tn + fp,
    }


def _joint(rows):
    judgment_and_answer = sum(
        bool(row.get("verifier_judgment_correct")) and bool(row.get("final_answer_correct"))
        for row in rows
    )
    judgment_only = sum(
        bool(row.get("verifier_judgment_correct")) and not bool(row.get("final_answer_correct"))
        for row in rows
    )
    answer_only = sum(
        (not bool(row.get("verifier_judgment_correct"))) and bool(row.get("final_answer_correct"))
        for row in rows
    )
    neither = len(rows) - judgment_and_answer - judgment_only - answer_only
    judgment_correct = judgment_and_answer + judgment_only
    judgment_incorrect = answer_only + neither
    return {
        "N": len(rows),
        "ARS": safe_divide(judgment_and_answer, len(rows)),
        "ACR": safe_divide(judgment_and_answer, judgment_correct),
        "RG": safe_divide(judgment_only, len(rows)),
        "CRG": safe_divide(judgment_only, judgment_correct),
        "MCR": safe_divide(answer_only, judgment_incorrect),
        "outcomes": {
            "judgment_correct__answer_correct": {
                "count": judgment_and_answer,
                "rate": safe_divide(judgment_and_answer, len(rows)),
            },
            "judgment_correct__answer_incorrect": {
                "count": judgment_only,
                "rate": safe_divide(judgment_only, len(rows)),
            },
            "judgment_incorrect__answer_correct": {
                "count": answer_only,
                "rate": safe_divide(answer_only, len(rows)),
            },
            "judgment_incorrect__answer_incorrect": {
                "count": neither,
                "rate": safe_divide(neither, len(rows)),
            },
        },
    }


def _recovery(rows):
    solver_correct = [row for row in rows if row.get("solver_correctness") == "correct"]
    solver_incorrect = [row for row in rows if row.get("solver_correctness") == "incorrect"]
    c3 = [row for row in rows if row.get("case_type") == "C3"]
    c4 = [row for row in rows if row.get("case_type") == "C4"]
    return {
        "CAPR": _rate(solver_correct, lambda row: row.get("final_answer_correct")),
        "TRR": _rate(solver_incorrect, lambda row: row.get("final_answer_correct")),
        "TRR_assisted": _rate(c3, lambda row: row.get("final_answer_correct")),
        "TRR_independent": _rate(c4, lambda row: row.get("final_answer_correct")),
        "support": {
            "solver_correct": len(solver_correct),
            "solver_incorrect": len(solver_incorrect),
            "C3": len(c3),
            "C4": len(c4),
        },
    }


def _performance(rows):
    faulty = [row for row in rows if row.get("verifier_correctness") == "faulty"]
    correct = [row for row in rows if row.get("verifier_correctness") == "correct"]
    revision = [
        row for row in rows
        if row.get("gold_should_revise_solver_answer") is not None
        and row.get("pred_should_revise_solver_answer") is not None
    ]
    joint = _joint(rows)
    recovery = _recovery(rows)
    vfar = _rate(correct, lambda row: row.get("pred_verifier_judgment") == "accept")
    ffrr = _rate(faulty, lambda row: row.get("pred_verifier_judgment") == "reject")
    oar = _rate(faulty, lambda row: row.get("pred_verifier_judgment") == "accept")
    orr = _rate(correct, lambda row: row.get("pred_verifier_judgment") == "reject")
    return {
        "N": len(rows),
        "FAA": _rate(rows, lambda row: row.get("final_answer_correct")),
        "SRA": _rate(revision, lambda row: row.get("solver_revision_correct")),
        "VJA": _rate(rows, lambda row: row.get("verifier_judgment_correct")),
        "VFAR": vfar,
        "FFRR": ffrr,
        "OAR": oar,
        "ORR": orr,
        "faulty_feedback_non_rejection_rate": None if ffrr is None else 1.0 - ffrr,
        "valid_feedback_non_acceptance_rate": None if vfar is None else 1.0 - vfar,
        **{key: recovery[key] for key in ("CAPR", "TRR", "TRR_assisted", "TRR_independent")},
        **{key: joint[key] for key in ("ARS", "ACR", "RG", "CRG", "MCR")},
    }


def _groups(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unknown"))].append(row)
    return {key: _performance(group) for key, group in sorted(groups.items())}


def _conditional_diagnosis(rows, labels, gold_field, pred_field):
    eligible = [
        row for row in rows
        if row.get(gold_field) and row.get("pred_verifier_judgment") == "reject"
    ]
    if not eligible:
        return {
            "N": 0,
            "DCR_exact": None,
            "DG_exact": None,
            "micro_precision": None,
            "micro_recall": None,
            "micro_F1": None,
            "macro_F1": None,
        }
    report = _label_report(eligible, labels, gold_field, pred_field)
    dcr = report["summary"]["exact_set_match"]
    return {
        "N": len(eligible),
        "DCR_exact": dcr,
        "DG_exact": None if dcr is None else 1.0 - dcr,
        "micro_precision": report["summary"]["micro_precision"],
        "micro_recall": report["summary"]["micro_recall"],
        "micro_F1": report["summary"]["micro_F1"],
        "macro_F1": report["summary"]["macro_F1"],
    }


def _case_macro(by_case):
    result = {}
    supported = [case for case in ("C1", "C2", "C3", "C4") if case in by_case]
    for metric in ("FAA", "VJA", "ARS"):
        values = [by_case[case].get(metric) for case in supported if by_case[case].get(metric) is not None]
        result[metric] = safe_divide(sum(values), len(values))
    result["supported_cases"] = supported
    return result


def _sensitivity_bounds(rows, field, predicate=lambda row: True, joint=False):
    subset = [row for row in rows if predicate(row)]
    definite = 0
    unresolved = 0
    for row in subset:
        judgment_ok = (not joint) or bool(row.get("verifier_judgment_correct"))
        if not judgment_ok:
            continue
        if row.get(field) == "definitely_matched":
            definite += 1
        elif row.get(field) == "unresolved":
            unresolved += 1
    return {
        "L": safe_divide(definite, len(subset)),
        "U": safe_divide(definite + unresolved, len(subset)),
        "definitely_matched_N": definite,
        "unresolved_N": unresolved,
        "N": len(subset),
    }


def _bound_gap(left, right):
    lower = None if left["L"] is None or right["U"] is None else left["L"] - right["U"]
    upper = None if left["U"] is None or right["L"] is None else left["U"] - right["L"]
    return {"L": lower, "U": upper}


def _transition_matrix(rows, previous, current):
    previous_field = f"final_answer_{previous}_outcome"
    current_field = f"final_answer_{current}_outcome"
    matrix = {source: {target: 0 for target in OUTCOME_STATUSES} for source in OUTCOME_STATUSES}
    for row in rows:
        source = row.get(previous_field)
        target = row.get(current_field)
        if source in matrix and target in matrix[source]:
            matrix[source][target] += 1

    source_counts = {source: sum(matrix[source].values()) for source in OUTCOME_STATUSES}
    target_counts = {
        target: sum(matrix[source][target] for source in OUTCOME_STATUSES)
        for target in OUTCOME_STATUSES
    }
    source_expected = {
        source: sum(row.get(previous_field) == source for row in rows)
        for source in OUTCOME_STATUSES
    }
    target_expected = {
        target: sum(row.get(current_field) == target for row in rows)
        for target in OUTCOME_STATUSES
    }
    return {
        "counts": matrix,
        "invariants": {
            "cells_sum_to_N": sum(source_counts.values()) == len(rows),
            "row_marginals_match_source": source_counts == source_expected,
            "column_marginals_match_target": target_counts == target_expected,
        },
    }


def _policy_sensitivity(rows, policy):
    field = f"final_answer_{policy}_outcome"
    faa = _sensitivity_bounds(rows, field)
    trr = _sensitivity_bounds(rows, field, lambda row: row.get("solver_correctness") == "incorrect")
    preservation = _sensitivity_bounds(rows, field, lambda row: row.get("solver_correctness") == "correct")
    c3 = _sensitivity_bounds(rows, field, lambda row: row.get("case_type") == "C3")
    c4 = _sensitivity_bounds(rows, field, lambda row: row.get("case_type") == "C4")
    ars = _sensitivity_bounds(rows, field, joint=True)
    status_counts = {
        status: sum(row.get(field) == status for row in rows)
        for status in OUTCOME_STATUSES
    }
    entry = {
        "FAA": {"L": faa["L"], "U": faa["U"]},
        "TRR": {"L": trr["L"], "U": trr["U"]},
        "ARS": {"L": ars["L"], "U": ars["U"]},
        "preservation": {"L": preservation["L"], "U": preservation["U"]},
        "C3_recovery": {"L": c3["L"], "U": c3["U"]},
        "C4_recovery": {"L": c4["L"], "U": c4["U"]},
        "C3_minus_C4": _bound_gap(c3, c4),
        "status_counts": {**status_counts, "missing": 0},
        "invariants": {"matched_statuses_equal_N_matched": sum(status_counts.values()) == len(rows)},
        "unresolved": {
            "N": status_counts["unresolved"],
            "percentage": safe_divide(status_counts["unresolved"], len(rows)),
        },
        "supports": {"solver_incorrect": trr["N"], "C3": c3["N"], "C4": c4["N"]},
    }
    if policy == "M4_operational":
        operational = sum(row.get("final_answer_M4_operational_matched") is True for row in rows)
        entry["operational"] = {
            "operational_match_rate": safe_divide(operational, len(rows)),
            "definitely_matched_rate": safe_divide(status_counts["definitely_matched"], len(rows)),
            "definitely_unmatched_rate": safe_divide(status_counts["definitely_unmatched"], len(rows)),
            "unresolved_rate": safe_divide(status_counts["unresolved"], len(rows)),
            "lower_bound": faa["L"],
            "upper_bound": faa["U"],
        }
    return entry


def _answer_sensitivity(rows):
    transitions = {}
    for previous, current in zip(MATCHER_POLICIES, MATCHER_POLICIES[1:]):
        transitions[f"{previous}_to_{current}"] = _transition_matrix(rows, previous, current)
    result = {policy: _policy_sensitivity(rows, policy) for policy in MATCHER_POLICIES}
    result["adjacent_transitions"] = transitions
    return result


def _gold_case(gold):
    stored = str(gold.get("case_type") or "").upper()
    if stored:
        return stored
    solver = str(gold.get("solver_correctness") or "").lower()
    verifier = str(gold.get("verifier_correctness") or "").lower()
    return CASE_BY_CORRECTNESS.get((solver, verifier), "")


def _coverage_records(gold_rows, row_by_id, field):
    records = []
    for gold in gold_rows:
        row = row_by_id.get(str(gold.get("id")))
        outcome = "missing" if row is None else row.get(field, "missing")
        records.append((gold, row, outcome))
    return records


def _coverage_bound(records, predicate, joint=False):
    subset = [record for record in records if predicate(record[0])]
    definite = 0
    unresolved = 0
    for _, row, outcome in subset:
        judgment_ok = (not joint) or bool(row and row.get("verifier_judgment_correct"))
        if not judgment_ok:
            continue
        if outcome == "definitely_matched":
            definite += 1
        elif outcome == "unresolved":
            unresolved += 1
    return {
        "L": safe_divide(definite, len(subset)),
        "U": safe_divide(definite + unresolved, len(subset)),
        "N": len(subset),
    }


def coverage_adjusted_answer_sensitivity(rows, gold_rows):
    row_by_id = {str(row.get("id")): row for row in rows}
    counted_statuses = ("definitely_matched", "definitely_unmatched", "unresolved", "missing")
    result = {}
    for policy in MATCHER_POLICIES:
        field = f"final_answer_{policy}_outcome"
        records = _coverage_records(gold_rows, row_by_id, field)
        status_counts = {
            status: sum(outcome == status for _, _, outcome in records)
            for status in counted_statuses
        }
        faa = _coverage_bound(records, lambda gold: True)
        trr = _coverage_bound(records, lambda gold: gold.get("solver_correctness") == "incorrect")
        preservation = _coverage_bound(records, lambda gold: gold.get("solver_correctness") == "correct")
        c3 = _coverage_bound(records, lambda gold: _gold_case(gold) == "C3")
        c4 = _coverage_bound(records, lambda gold: _gold_case(gold) == "C4")
        ars = _coverage_bound(records, lambda gold: True, joint=True)
        matched = sum(status_counts[status] for status in OUTCOME_STATUSES)
        entry = {
            "FAA": {"L": faa["L"], "U": faa["U"]},
            "TRR": {"L": trr["L"], "U": trr["U"]},
            "ARS": {"L": ars["L"], "U": ars["U"]},
            "preservation": {"L": preservation["L"], "U": preservation["U"]},
            "C3_recovery": {"L": c3["L"], "U": c3["U"]},
            "C4_recovery": {"L": c4["L"], "U": c4["U"]},
            "C3_minus_C4": _bound_gap(c3, c4),
            "status_counts": status_counts,
            "invariants": {
                "matched_statuses_equal_N_matched": matched == len(rows),
                "matched_plus_missing_equal_N_gold": sum(status_counts.values()) == len(gold_rows),
            },
        }
        if policy == "M4_operational":
            operational = sum(
                row is not None and row.get("final_answer_M4_operational_matched") is True
                for _, row, _ in records
            )
            n_gold = len(gold_rows)
            entry["operational"] = {
                "operational_match_rate": safe_divide(operational, n_gold),
                "definitely_matched_rate": safe_divide(status_counts["definitely_matched"], n_gold),
                "definitely_unmatched_rate": safe_divide(status_counts["definitely_unmatched"], n_gold),
                "unresolved_rate": safe_divide(status_counts["unresolved"], n_gold),
                "lower_bound": faa["L"],
                "upper_bound": faa["U"],
            }
        result[policy] = entry
    return result


def compute_detailed_metrics(
    rows: list[dict[str, Any]],
    fallacy_labels: Iterable[str],
    hallucination_labels: Iterable[str],
) -> dict[str, Any]:
    fallacy_labels = tuple(fallacy_labels)
    hallucination_labels = tuple(hallucination_labels)
    hallucination = _label_report(rows, hallucination_labels, "hallucination_type", "pred_hallucination_types")
    fallacy = _label_report(rows, fallacy_labels, "fault_type", "pred_fallacy_types")
    judgment = _single_class(rows, JUDGMENTS, "gold_verifier_judgment", "pred_verifier_judgment")
    recovery = _recovery(rows)
    joint = _joint(rows)
    overall = _performance(rows)
    binary = {
        "any_fault": _binary_report(rows, "gold_has_any_fault", "pred_has_any_fault"),
        "fallacy": _binary_report(rows, "gold_has_fallacy", "pred_has_fallacy"),
        "hallucination": _binary_report(rows, "gold_has_hallucination", "pred_has_hallucination"),
    }
    overall.update({
        "VJ_macro_precision": judgment["macro_precision"],
        "VJ_macro_recall": judgment["macro_recall"],
        "VJ_macro_F1": judgment["macro_F1"],
        "hallucination_micro_F1": hallucination["summary"]["micro_F1"],
        "fallacy_micro_F1": fallacy["summary"]["micro_F1"],
        "binary_any_fault_F1": binary["any_fault"]["F1"],
        "binary_fallacy_F1": binary["fallacy"]["F1"],
        "binary_hallucination_F1": binary["hallucination"]["F1"],
    })
    by_case = _groups(rows, "case_type")
    return {
        "evaluation_schema_version": "3.0",
        "overall": overall,
        "by_domain": _groups(rows, "domain"),
        "by_case": by_case,
        "by_solver_correctness": _groups(rows, "solver_correctness"),
        "by_verifier_correctness": _groups(rows, "verifier_correctness"),
        "verifier_judgment": judgment,
        "answer_recovery": recovery,
        "joint_adjudication_recovery": joint,
        "recovery_consistency": joint["outcomes"],
        "hallucination_detection": hallucination,
        "fallacy_detection": fallacy,
        "binary_fault_presence": binary,
        "conditional_diagnosis": {
            "fallacy": _conditional_diagnosis(rows, fallacy_labels, "fault_type", "pred_fallacy_types"),
            "hallucination": _conditional_diagnosis(
                rows, hallucination_labels, "hallucination_type", "pred_hallucination_types"
            ),
        },
        "case_macro": _case_macro(by_case),
        "answer_sensitivity": _answer_sensitivity(rows),
    }
