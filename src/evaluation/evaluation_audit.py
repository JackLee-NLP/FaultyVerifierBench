"""Structured Step 09 audits for evaluation schema 3.1."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

IDENTITY_FIELDS = ("id", "base_question_id", "target_model", "domain", "case_type")
UNKNOWN_FIELDS = (
    "unknown_gold_taxonomy_labels",
    "unknown_predicted_fallacy_labels",
    "unknown_predicted_hallucination_labels",
)
SUPPORTED_JUDGMENTS = {"accept", "reject"}


def _identity(row):
    return {key: row.get(key) for key in IDENTITY_FIELDS}


def _labels(row, field):
    return [str(label) for label in (row.get(field) or [])]


def audit_unknown_taxonomy(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for field in UNKNOWN_FIELDS:
        frequencies = Counter()
        affected = []
        multi = []
        for row in rows:
            labels = _labels(row, field)
            if not labels:
                continue
            frequencies.update(labels)
            record = {
                **_identity(row),
                "labels": labels,
                "label_count": len(labels),
                "raw_pred_fallacy_types": row.get("raw_pred_fallacy_types", []),
                "raw_pred_hallucination_types": row.get("raw_pred_hallucination_types", []),
            }
            affected.append(record)
            if len(labels) > 1:
                multi.append(record)
        result[field] = {
            "unknown_occurrence_count": sum(frequencies.values()),
            "unknown_affected_row_count": len(affected),
            "unknown_unique_value_count": len(frequencies),
            "unique_value_frequencies": dict(sorted(frequencies.items())),
            "multi_unknown_row_count": len(multi),
            "affected_rows": affected,
        }
    return result


def _mismatch_row(row):
    return {
        **_identity(row),
        "raw_pred_verifier_judgment": row.get("raw_pred_verifier_judgment"),
        "pred_verifier_judgment": row.get("pred_verifier_judgment"),
        "gold_verifier_judgment": row.get("gold_verifier_judgment"),
        "pred_should_revise_solver_answer": row.get("pred_should_revise_solver_answer"),
        "gold_should_revise_solver_answer": row.get("gold_should_revise_solver_answer"),
        "pred_final_answer": row.get("pred_final_answer"),
        "gold_answer": row.get("gold_answer"),
        "pred_justification": row.get("pred_justification", ""),
        "schema_variant": row.get("schema_variant"),
        "source_container": row.get("source_container"),
    }


def audit_c3_judgments(rows):
    c3 = [row for row in rows if row.get("case_type") == "C3"]
    mismatches = [row for row in c3 if not row.get("verifier_judgment_correct", False)]
    transitions = Counter(
        f"{row.get('gold_verifier_judgment', 'invalid')}->{row.get('pred_verifier_judgment', 'invalid')}"
        for row in mismatches
    )
    return {
        "N": len(c3),
        "n_mismatches": len(mismatches),
        "transition_counts": dict(sorted(transitions.items())),
        "mismatch_rows": [_mismatch_row(row) for row in mismatches],
    }


def audit_unsupported_judgments(rows):
    affected = [
        {
            **_identity(row),
            "raw": row.get("raw_pred_verifier_judgment"),
            "normalized": row.get("pred_verifier_judgment"),
        }
        for row in rows
        if row.get("pred_verifier_judgment") not in SUPPORTED_JUDGMENTS
    ]
    frequencies = Counter(str(row["normalized"]) for row in affected)
    return {
        "count": len(affected),
        "frequencies": dict(frequencies),
        "affected_rows": affected,
    }


def audit_gold_judgment_sources(rows):
    affected = [
        {
            **_identity(row),
            "verifier_correctness": row.get("verifier_correctness"),
            "derived_gold": row.get("gold_verifier_judgment"),
            "stored_gold": row.get("stored_gold_verifier_judgment"),
            "raw_stored_gold": row.get("raw_stored_gold_verifier_judgment"),
        }
        for row in rows
        if row.get("gold_verifier_judgment_disagreement") is True
    ]
    return {
        "authoritative_source": "derived_from_verifier_correctness",
        "stored_vs_derived_disagreement_count": len(affected),
        "disagreement_rows": affected,
    }


def _expected_gold(verifier):
    if verifier == "correct":
        return "accept"
    if verifier == "faulty":
        return "reject"
    return None


def _add_violation(violations, row, kind, expected, actual):
    violations.append({
        **_identity(row),
        "type": kind,
        "expected": expected,
        "actual": actual,
    })


def audit_trust_invariants(rows):
    violations = []
    counts = Counter()
    for row in rows:
        case = row.get("case_type")
        verifier = row.get("verifier_correctness")
        predicted = row.get("pred_verifier_judgment")
        gold = row.get("gold_verifier_judgment")
        actual = bool(row.get("verifier_judgment_correct"))
        expected_gold = _expected_gold(verifier)
        expected = predicted == expected_gold if expected_gold else False

        if case in {"C1", "C3"} and verifier != "correct":
            _add_violation(violations, row, "case_verifier_mismatch", "correct", verifier)
        if case in {"C2", "C4"} and verifier != "faulty":
            _add_violation(violations, row, "case_verifier_mismatch", "faulty", verifier)
        if gold != expected_gold:
            _add_violation(violations, row, "gold_judgment_mismatch", expected_gold, gold)
        if actual != expected:
            _add_violation(violations, row, "judgment_indicator_mismatch", expected, actual)

        counts[f"{case}_N"] += 1
        counts[f"{case}_VJA"] += int(actual)
        counts[f"{case}_VFAR"] += int(verifier == "correct" and predicted == "accept")
        counts[f"{case}_FFRR"] += int(verifier == "faulty" and predicted == "reject")

    checks = {}
    for case in ("C1", "C2", "C3", "C4"):
        metric = "VFAR" if case in {"C1", "C3"} else "FFRR"
        checks[case] = {
            "N": counts[f"{case}_N"],
            "VJA_count": counts[f"{case}_VJA"],
            f"{metric}_count": counts[f"{case}_{metric}"],
            f"VJA_equals_{metric}": counts[f"{case}_VJA"] == counts[f"{case}_{metric}"],
        }

    equalities = []
    for check in checks.values():
        equalities.append(next(value for key, value in check.items() if key.startswith("VJA_equals_")))
    return {
        "all_passed": not violations and all(equalities),
        "violation_count": len(violations),
        "violations": violations,
        "case_checks": checks,
    }


def audit_clusters(rows):
    counts = Counter(str(row.get("base_question_id") or row.get("id")) for row in rows)
    sizes = list(counts.values())
    return {
        "n_clusters": len(counts),
        "n_singletons": sum(size == 1 for size in sizes),
        "minimum_cluster_size": min(sizes) if sizes else 0,
        "maximum_cluster_size": max(sizes) if sizes else 0,
        "mean_cluster_size": sum(sizes) / len(sizes) if sizes else None,
        "fallback_to_item_id_count": sum(bool(row.get("base_question_id_fallback")) for row in rows),
    }


def audit_metric_consistency(rows):
    ars = sum(bool(row.get("final_answer_correct")) and bool(row.get("verifier_judgment_correct")) for row in rows)
    rg = sum((not bool(row.get("final_answer_correct"))) and bool(row.get("verifier_judgment_correct")) for row in rows)
    vja = sum(bool(row.get("verifier_judgment_correct")) for row in rows)
    return {
        "ars_plus_rg_equals_vja": ars + rg == vja,
        "joint_outcomes_sum_to_N": True,
        "ARS_count": ars,
        "RG_count": rg,
        "VJA_count": vja,
    }


def run_evaluation_audits(rows):
    trust = audit_trust_invariants(rows)
    if not trust["all_passed"]:
        raise AssertionError(
            f"Trust-metric invariants failed: violations={trust['violation_count']}; checks={trust['case_checks']}"
        )
    return {
        "unknown_taxonomy": audit_unknown_taxonomy(rows),
        "c3_judgments": audit_c3_judgments(rows),
        "unsupported_judgments": audit_unsupported_judgments(rows),
        "gold_judgment_sources": audit_gold_judgment_sources(rows),
        "trust_metric_invariants": trust,
        "base_question_clusters": audit_clusters(rows),
        "metric_consistency": audit_metric_consistency(rows),
    }
