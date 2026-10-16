"""Fast cluster-aware statistical analysis for FaultyVerifierBench Step 09.

Each base-question cluster is reduced once to sufficient statistics. Bootstrap
replicates then use multinomial cluster weights and NumPy matrix multiplication,
without rebuilding Python row dictionaries or recomputing detailed metrics.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_METRICS = (
    "FAA", "VJA", "SRA", "VFAR", "FFRR", "OAR", "ORR",
    "CAPR", "TRR", "TRR_assisted", "TRR_independent",
    "ARS", "ACR", "RG", "CRG", "MCR",
    "fallacy_micro_F1", "hallucination_micro_F1",
    "binary_any_fault_F1", "binary_fallacy_F1",
    "binary_hallucination_F1", "DCR_fallacy", "DG_fallacy",
    "DCR_hallucination", "DG_hallucination",
)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(np.broadcast_shapes(np.shape(num), np.shape(den)), np.nan)
    np.divide(num, den, out=out, where=np.asarray(den) != 0)
    return out


def _f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    return _ratio(2.0 * tp, 2.0 * tp + fp + fn)


def _labels(row: dict[str, Any], field: str) -> set[str]:
    return set(row.get(field) or [])


def _align(rows_by_model):
    models = sorted(rows_by_model)
    if not models:
        return [], {}
    ids = {str(row["id"]) for row in rows_by_model[models[0]]}
    aligned = {}
    for model in models:
        index = {str(row["id"]): row for row in rows_by_model[model]}
        if set(index) != ids:
            raise ValueError("Statistical tests require identical common-item ID sets.")
        aligned[model] = [index[item_id] for item_id in sorted(ids)]
    return models, aligned


def _cluster_layout(rows):
    members = defaultdict(list)
    for i, row in enumerate(rows):
        members[str(row.get("base_question_id") or row["id"])].append(i)
    cluster_ids = sorted(members)
    item_to_cluster = np.empty(len(rows), dtype=np.int32)
    for cluster_index, cluster_id in enumerate(cluster_ids):
        item_to_cluster[members[cluster_id]] = cluster_index
    return cluster_ids, item_to_cluster


def _flags(rows, predicate):
    return np.fromiter((bool(predicate(row)) for row in rows), dtype=np.float64, count=len(rows))


def _item_columns(rows, fallacy_labels, hallucination_labels):
    n = len(rows)
    answer = _flags(rows, lambda row: row.get("final_answer_correct"))
    judgment = _flags(rows, lambda row: row.get("verifier_judgment_correct"))
    revision_eligible = _flags(
        rows,
        lambda row: row.get("gold_should_revise_solver_answer") is not None
        and row.get("pred_should_revise_solver_answer") is not None,
    )
    revision_success = _flags(rows, lambda row: row.get("solver_revision_correct")) * revision_eligible
    correct_verifier = _flags(rows, lambda row: row.get("verifier_correctness") == "correct")
    faulty_verifier = _flags(rows, lambda row: row.get("verifier_correctness") == "faulty")
    accept = _flags(rows, lambda row: row.get("pred_verifier_judgment") == "accept")
    reject = _flags(rows, lambda row: row.get("pred_verifier_judgment") == "reject")
    solver_correct = _flags(rows, lambda row: row.get("solver_correctness") == "correct")
    solver_incorrect = _flags(rows, lambda row: row.get("solver_correctness") == "incorrect")
    c3 = _flags(rows, lambda row: row.get("case_type") == "C3")
    c4 = _flags(rows, lambda row: row.get("case_type") == "C4")

    columns = {
        "N": np.ones(n),
        "FAA_num": answer,
        "VJA_num": judgment,
        "SRA_num": revision_success,
        "SRA_den": revision_eligible,
        "VFAR_num": correct_verifier * accept,
        "VFAR_den": correct_verifier,
        "FFRR_num": faulty_verifier * reject,
        "FFRR_den": faulty_verifier,
        "OAR_num": faulty_verifier * accept,
        "OAR_den": faulty_verifier,
        "ORR_num": correct_verifier * reject,
        "ORR_den": correct_verifier,
        "CAPR_num": solver_correct * answer,
        "CAPR_den": solver_correct,
        "TRR_num": solver_incorrect * answer,
        "TRR_den": solver_incorrect,
        "TRRA_num": c3 * answer,
        "TRRA_den": c3,
        "TRRI_num": c4 * answer,
        "TRRI_den": c4,
        "ARS_num": answer * judgment,
        "RG_num": (1.0 - answer) * judgment,
        "J_den": judgment,
        "JI_den": 1.0 - judgment,
        "MCR_num": answer * (1.0 - judgment),
    }

    for prefix, gold_field, pred_field in (
        ("any", "gold_has_any_fault", "pred_has_any_fault"),
        ("fallacy_presence", "gold_has_fallacy", "pred_has_fallacy"),
        ("hallucination_presence", "gold_has_hallucination", "pred_has_hallucination"),
    ):
        gold = _flags(rows, lambda row: row.get(gold_field))
        pred = _flags(rows, lambda row: row.get(pred_field))
        columns[f"{prefix}_tp"] = gold * pred
        columns[f"{prefix}_fp"] = (1.0 - gold) * pred
        columns[f"{prefix}_fn"] = gold * (1.0 - pred)

    for prefix, labels, gold_field, pred_field in (
        ("fallacy", fallacy_labels, "fault_type", "pred_fallacy_types"),
        ("hallucination", hallucination_labels, "hallucination_type", "pred_hallucination_types"),
    ):
        gold_sets = [_labels(row, gold_field) for row in rows]
        pred_sets = [_labels(row, pred_field) for row in rows]
        pairs = list(zip(gold_sets, pred_sets, rows))
        tp = np.fromiter(
            (sum(label in gold and label in pred for label in labels) for gold, pred, _ in pairs),
            dtype=np.float64,
            count=n,
        )
        fp = np.fromiter(
            (sum(label not in gold and label in pred for label in labels) for gold, pred, _ in pairs),
            dtype=np.float64,
            count=n,
        )
        fn = np.fromiter(
            (sum(label in gold and label not in pred for label in labels) for gold, pred, _ in pairs),
            dtype=np.float64,
            count=n,
        )
        eligible = np.fromiter(
            (
                bool(gold) and row.get("pred_verifier_judgment") == "reject"
                for gold, _, row in pairs
            ),
            dtype=np.float64,
            count=n,
        )
        exact = np.fromiter(
            (
                bool(gold)
                and row.get("pred_verifier_judgment") == "reject"
                and gold == pred
                and row.get("adjudication_parse_success") is True
                for gold, pred, row in pairs
            ),
            dtype=np.float64,
            count=n,
        )
        columns[f"{prefix}_tp"] = tp
        columns[f"{prefix}_fp"] = fp
        columns[f"{prefix}_fn"] = fn
        columns[f"DCR_{prefix}_num"] = exact
        columns[f"DCR_{prefix}_den"] = eligible
    return columns


def _cluster_sums(columns, item_to_cluster, n_clusters):
    names = list(columns)
    matrix = np.zeros((n_clusters, len(names)), dtype=np.float64)
    for column, name in enumerate(names):
        np.add.at(matrix[:, column], item_to_cluster, columns[name])
    return names, matrix


def _metrics_from_sums(sums, position):
    x = lambda name: sums[..., position[name]]
    result = {
        "FAA": _ratio(x("FAA_num"), x("N")),
        "VJA": _ratio(x("VJA_num"), x("N")),
        "SRA": _ratio(x("SRA_num"), x("SRA_den")),
        "VFAR": _ratio(x("VFAR_num"), x("VFAR_den")),
        "FFRR": _ratio(x("FFRR_num"), x("FFRR_den")),
        "OAR": _ratio(x("OAR_num"), x("OAR_den")),
        "ORR": _ratio(x("ORR_num"), x("ORR_den")),
        "CAPR": _ratio(x("CAPR_num"), x("CAPR_den")),
        "TRR": _ratio(x("TRR_num"), x("TRR_den")),
        "TRR_assisted": _ratio(x("TRRA_num"), x("TRRA_den")),
        "TRR_independent": _ratio(x("TRRI_num"), x("TRRI_den")),
        "ARS": _ratio(x("ARS_num"), x("N")),
        "ACR": _ratio(x("ARS_num"), x("J_den")),
        "RG": _ratio(x("RG_num"), x("N")),
        "CRG": _ratio(x("RG_num"), x("J_den")),
        "MCR": _ratio(x("MCR_num"), x("JI_den")),
        "fallacy_micro_F1": _f1(x("fallacy_tp"), x("fallacy_fp"), x("fallacy_fn")),
        "hallucination_micro_F1": _f1(x("hallucination_tp"), x("hallucination_fp"), x("hallucination_fn")),
        "binary_any_fault_F1": _f1(x("any_tp"), x("any_fp"), x("any_fn")),
        "binary_fallacy_F1": _f1(x("fallacy_presence_tp"), x("fallacy_presence_fp"), x("fallacy_presence_fn")),
        "binary_hallucination_F1": _f1(
            x("hallucination_presence_tp"),
            x("hallucination_presence_fp"),
            x("hallucination_presence_fn"),
        ),
        "DCR_fallacy": _ratio(x("DCR_fallacy_num"), x("DCR_fallacy_den")),
        "DG_fallacy": 1.0 - _ratio(x("DCR_fallacy_num"), x("DCR_fallacy_den")),
        "DCR_hallucination": _ratio(x("DCR_hallucination_num"), x("DCR_hallucination_den")),
        "DG_hallucination": 1.0 - _ratio(x("DCR_hallucination_num"), x("DCR_hallucination_den")),
    }
    return result


def _interval(values, confidence):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None, None, 0
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return float(low), float(high), int(finite.size)


MATCHER_BOOTSTRAP_METRICS = (
    "M1_FAA_L",
    "M1_TRR_L",
    "M1_ARS_L",
    "M3_FAA_L",
    "M3_FAA_U",
    "M3_TRR_L",
    "M3_TRR_U",
    "M3_ARS_L",
    "M3_ARS_U",
    "M3_unresolved_rate",
    "M3_C3_minus_C4_L",
    "M3_C3_minus_C4_U",
)


def _matcher_columns(rows):
    columns = {"N": np.ones(len(rows))}
    for policy in ("M1_numeric", "M3_guarded_containment"):
        field = f"final_answer_{policy}_outcome"
        definite = _flags(rows, lambda row, field=field: row.get(field) == "definitely_matched")
        unresolved = _flags(rows, lambda row, field=field: row.get(field) == "unresolved")
        solver_incorrect = _flags(rows, lambda row: row.get("solver_correctness") == "incorrect")
        judgment = _flags(rows, lambda row: row.get("verifier_judgment_correct"))
        c3 = _flags(rows, lambda row: row.get("case_type") == "C3")
        c4 = _flags(rows, lambda row: row.get("case_type") == "C4")
        prefix = "m1" if policy.startswith("M1") else "m3"
        columns[f"{prefix}_D"] = definite
        columns[f"{prefix}_U"] = unresolved
        columns[f"{prefix}_D_SI"] = definite * solver_incorrect
        columns[f"{prefix}_U_SI"] = unresolved * solver_incorrect
        columns[f"{prefix}_D_J"] = definite * judgment
        columns[f"{prefix}_U_J"] = unresolved * judgment
        columns[f"{prefix}_D_C3"] = definite * c3
        columns[f"{prefix}_U_C3"] = unresolved * c3
        columns[f"{prefix}_D_C4"] = definite * c4
        columns[f"{prefix}_U_C4"] = unresolved * c4
        columns[f"{prefix}_SI"] = solver_incorrect
        columns[f"{prefix}_C3"] = c3
        columns[f"{prefix}_C4"] = c4
    return columns


def _matcher_metrics_from_sums(sums, position):
    def column(name):
        return sums[..., position[name]]

    c3_lower = _ratio(column("m3_D_C3"), column("m3_C3"))
    c3_upper = _ratio(column("m3_D_C3") + column("m3_U_C3"), column("m3_C3"))
    c4_lower = _ratio(column("m3_D_C4"), column("m3_C4"))
    c4_upper = _ratio(column("m3_D_C4") + column("m3_U_C4"), column("m3_C4"))
    return {
        "M1_FAA_L": _ratio(column("m1_D"), column("N")),
        "M1_TRR_L": _ratio(column("m1_D_SI"), column("m1_SI")),
        "M1_ARS_L": _ratio(column("m1_D_J"), column("N")),
        "M3_FAA_L": _ratio(column("m3_D"), column("N")),
        "M3_FAA_U": _ratio(column("m3_D") + column("m3_U"), column("N")),
        "M3_TRR_L": _ratio(column("m3_D_SI"), column("m3_SI")),
        "M3_TRR_U": _ratio(column("m3_D_SI") + column("m3_U_SI"), column("m3_SI")),
        "M3_ARS_L": _ratio(column("m3_D_J"), column("N")),
        "M3_ARS_U": _ratio(column("m3_D_J") + column("m3_U_J"), column("N")),
        "M3_unresolved_rate": _ratio(column("m3_U"), column("N")),
        "M3_C3_minus_C4_L": c3_lower - c4_upper,
        "M3_C3_minus_C4_U": c3_upper - c4_lower,
    }


def _finite_float(value):
    point = np.asarray(value).item()
    if not np.isfinite(point):
        return None
    return float(point)


def _metric_interval(draws, confidence):
    low, high, _valid = _interval(draws, confidence)
    return low, high


def compute_matcher_ladder_bootstrap(
    common_rows_by_model,
    n_resamples=1000,
    seed=2027,
    confidence=0.95,
    batch_size=100,
):
    models, aligned = _align(common_rows_by_model)
    if not models or n_resamples <= 0:
        return {"status": "disabled", "confidence_intervals": {}, "paired_differences": {}}

    cluster_ids, item_to_cluster = _cluster_layout(aligned[models[0]])
    n_clusters = len(cluster_ids)
    matrices = {}
    position = None
    for model in models:
        current_names, matrix = _cluster_sums(_matcher_columns(aligned[model]), item_to_cluster, n_clusters)
        matrices[model] = matrix
        if position is None:
            position = {name: index for index, name in enumerate(current_names)}

    points = {
        model: _matcher_metrics_from_sums(matrices[model].sum(axis=0), position)
        for model in models
    }
    draws = {model: {metric: [] for metric in MATCHER_BOOTSTRAP_METRICS} for model in models}
    pairs = [(left, right) for index, left in enumerate(models) for right in models[index + 1:]]
    pair_draws = {
        f"{left}__vs__{right}": {metric: [] for metric in MATCHER_BOOTSTRAP_METRICS}
        for left, right in pairs
    }

    rng = np.random.default_rng(seed)
    completed = 0
    probabilities = np.full(n_clusters, 1.0 / n_clusters)
    batch_size = max(1, int(batch_size))
    while completed < n_resamples:
        current = min(batch_size, n_resamples - completed)
        weights = rng.multinomial(n_clusters, probabilities, size=current).astype(np.float64)
        batch = {
            model: _matcher_metrics_from_sums(weights @ matrices[model], position)
            for model in models
        }
        for model in models:
            for metric in MATCHER_BOOTSTRAP_METRICS:
                values = np.asarray(batch[model][metric]).reshape(-1).tolist()
                draws[model][metric].extend(values)
        for left, right in pairs:
            pair_name = f"{left}__vs__{right}"
            for metric in MATCHER_BOOTSTRAP_METRICS:
                difference = np.asarray(batch[left][metric]) - np.asarray(batch[right][metric])
                pair_draws[pair_name][metric].extend(difference.reshape(-1).tolist())
        completed += current

    intervals = {}
    for model in models:
        metrics = {}
        for metric in MATCHER_BOOTSTRAP_METRICS:
            low, high = _metric_interval(draws[model][metric], confidence)
            metrics[metric] = {
                "point_estimate": _finite_float(points[model][metric]),
                "ci_low": low,
                "ci_high": high,
            }
        intervals[model] = {
            "N": len(aligned[model]),
            "n_clusters": n_clusters,
            "seed": seed,
            "n_resamples": n_resamples,
            "confidence": confidence,
            "metrics": metrics,
        }

    paired = {}
    for left, right in pairs:
        pair_name = f"{left}__vs__{right}"
        metrics = {}
        for metric in MATCHER_BOOTSTRAP_METRICS:
            low, high = _metric_interval(pair_draws[pair_name][metric], confidence)
            metrics[metric] = {
                "point_difference": _finite_float(points[left][metric] - points[right][metric]),
                "ci_low": low,
                "ci_high": high,
            }
        paired[pair_name] = {
            "left_model": left,
            "right_model": right,
            "N": len(aligned[left]),
            "metrics": metrics,
        }
    return {
        "status": "complete",
        "method": "paired base-question clustered bootstrap",
        "cluster_field": "base_question_id",
        "confidence": confidence,
        "n_resamples": n_resamples,
        "seed": seed,
        "confidence_intervals": intervals,
        "paired_differences": paired,
    }


def compute_all_statistical_tests(
    common_rows_by_model: Mapping[str, Sequence[dict[str, Any]]],
    fallacy_labels: Sequence[str],
    hallucination_labels: Sequence[str],
    n_resamples: int = 1000,
    seed: int = 2027,
    confidence: float = 0.95,
    batch_size: int = 100,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if n_resamples <= 0:
        return {"status": "disabled", "bootstrap_confidence_intervals": {}, "paired_significance_tests": {}}
    models, aligned = _align(common_rows_by_model)
    if not models:
        return {"status": "no_models", "bootstrap_confidence_intervals": {}, "paired_significance_tests": {}}

    cluster_ids, item_to_cluster = _cluster_layout(aligned[models[0]])
    n_clusters = len(cluster_ids)
    matrices = {}
    names = None
    for model in models:
        other_ids, other_layout = _cluster_layout(aligned[model])
        if other_ids != cluster_ids or not np.array_equal(other_layout, item_to_cluster):
            raise ValueError("Base-question clusters differ across aligned models.")
        current_names, matrices[model] = _cluster_sums(
            _item_columns(aligned[model], tuple(fallacy_labels), tuple(hallucination_labels)),
            item_to_cluster,
            n_clusters,
        )
        if names is None:
            names = current_names
        elif current_names != names:
            raise ValueError("Sufficient-statistic columns differ across models.")
    position = {name: i for i, name in enumerate(names)}
    points = {model: _metrics_from_sums(matrices[model].sum(axis=0), position) for model in models}
    draws = {model: {metric: [] for metric in DEFAULT_METRICS} for model in models}
    pairs = [(left, right) for i, left in enumerate(models) for right in models[i + 1:]]
    pair_draws = {f"{left}__vs__{right}": {metric: [] for metric in DEFAULT_METRICS} for left, right in pairs}

    rng = np.random.default_rng(seed)
    completed = 0
    batch_size = max(1, int(batch_size))
    probabilities = np.full(n_clusters, 1.0 / n_clusters)
    while completed < n_resamples:
        current = min(batch_size, n_resamples - completed)
        weights = rng.multinomial(n_clusters, probabilities, size=current).astype(np.float64)
        batch = {model: _metrics_from_sums(weights @ matrices[model], position) for model in models}
        for model in models:
            for metric in DEFAULT_METRICS:
                draws[model][metric].extend(np.asarray(batch[model][metric]).reshape(-1).tolist())
        for left, right in pairs:
            pair_name = f"{left}__vs__{right}"
            for metric in DEFAULT_METRICS:
                difference = np.asarray(batch[left][metric]) - np.asarray(batch[right][metric])
                pair_draws[pair_name][metric].extend(difference.reshape(-1).tolist())
        completed += current
        if progress is not None:
            progress(f"cluster bootstrap {completed}/{n_resamples} resamples completed")

    intervals = {}
    for model in models:
        metrics = {}
        for metric in DEFAULT_METRICS:
            low, high, valid = _interval(draws[model][metric], confidence)
            point = np.asarray(points[model][metric]).item()
            metrics[metric] = {
                "point_estimate": None if not np.isfinite(point) else float(point),
                "ci_low": low,
                "ci_high": high,
                "valid_resamples": valid,
            }
        intervals[model] = {
            "method": "paired base-question clustered nonparametric bootstrap, percentile interval",
            "implementation": "multinomial cluster weights over pre-aggregated sufficient statistics",
            "cluster_field": "base_question_id",
            "n_clusters": n_clusters,
            "seed": seed,
            "n_resamples": n_resamples,
            "confidence": confidence,
            "N": len(aligned[model]),
            "metrics": metrics,
        }

    paired = {}
    for left, right in pairs:
        pair_name = f"{left}__vs__{right}"
        metrics = {}
        for metric in DEFAULT_METRICS:
            low, high, valid = _interval(pair_draws[pair_name][metric], confidence)
            left_point = np.asarray(points[left][metric]).item()
            right_point = np.asarray(points[right][metric]).item()
            point = left_point - right_point
            metrics[metric] = {
                "point_difference": None if not np.isfinite(point) else float(point),
                "ci_low": low,
                "ci_high": high,
                "valid_resamples": valid,
            }
        paired[pair_name] = {
            "method": "paired base-question clustered nonparametric bootstrap",
            "implementation": "shared multinomial cluster weights",
            "difference_definition": "left_minus_right",
            "left_model": left,
            "right_model": right,
            "cluster_field": "base_question_id",
            "n_clusters": n_clusters,
            "seed": seed,
            "n_resamples": n_resamples,
            "confidence": confidence,
            "N": len(aligned[left]),
            "metrics": metrics,
        }
    return {"status": "complete", "bootstrap_confidence_intervals": intervals, "paired_significance_tests": paired}
