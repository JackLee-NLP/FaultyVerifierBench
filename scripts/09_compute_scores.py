"""FaultyVerifierBench Step 09 scoring, auditing, and clustered statistics.

This is the sole Step 09 executable. It rebuilds the evaluation results
directly from the current Step 08 outputs.
"""
from __future__ import annotations

import argparse
import atexit
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.config.taxonomy_loader import load_benchmark_taxonomies
from src.evaluation.detailed_metrics import (
    compute_detailed_metrics,
    coverage_adjusted_answer_sensitivity,
)
from src.evaluation.evaluation_audit import run_evaluation_audits
from src.evaluation.evaluation_subsets import (
    build_common_item_summary,
    filter_rows_by_ids,
    get_common_item_ids,
)
from src.evaluation.metric_utils import (
    compare_id_sets,
    index_by_id,
    load_jsonl,
    safe_divide,
    write_json,
    write_jsonl_atomic,
)
from src.evaluation.score_records import build_score_row
from src.evaluation.statistical_tests import (
    compute_all_statistical_tests,
    compute_matcher_ladder_bootstrap,
)

MATCHER_POLICIES = (
    "M0_exact",
    "M1_numeric",
    "M2_ordered_containment",
    "M3_guarded_containment",
    "M4_operational",
)
MATCHER_METRICS = ("FAA", "TRR", "ARS", "preservation", "C3_recovery", "C4_recovery")
TABLE_FIELDS = (
    "N", "FAA", "SRA", "VJA", "VFAR", "FFRR", "OAR", "ORR", "CAPR",
    "TRR", "TRR_assisted", "TRR_independent", "ARS", "ACR", "RG", "CRG", "MCR",
)
UNKNOWN_KEYS = (
    "unknown_occurrence_count",
    "unknown_affected_row_count",
    "unknown_unique_value_count",
    "multi_unknown_row_count",
)


class Tee:
    """Write to the terminal and a log file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(stream.isatty() for stream in self.streams if hasattr(stream, "isatty"))


def start_log(result_dir, split):
    path = result_dir / f"step_09_compute_scores_{split}.log"
    handle = path.open("w", encoding="utf-8")
    terminal_out, terminal_err = sys.stdout, sys.stderr
    sys.stdout = Tee(terminal_out, handle)
    sys.stderr = Tee(terminal_err, handle)

    def close():
        sys.stdout.flush()
        sys.stderr.flush()
        handle.close()
        sys.stdout, sys.stderr = terminal_out, terminal_err

    atexit.register(close)
    return path


def log(message):
    print(message, flush=True)


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return value


def load_models(path):
    with open(path, encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    models = value.get("models")
    if not isinstance(models, dict):
        raise TypeError(f"Expected models mapping in {path}")
    return models


def setting(config, key, default=None):
    return config.get("evaluation", {}).get(key, default)


def selected_models(config, models, requested):
    selected = requested or setting(config, "target_models")
    if not selected:
        default = setting(config, "default_model")
        selected = [default] if default else list(models)
    missing = [name for name in selected if name not in models]
    if missing:
        raise ValueError(f"Models missing from models.yaml: {missing}")
    return list(selected)


def output_paths(config, model, split):
    paths = config.get("paths", {})
    split_dir = Path(paths.get("split_dir", "data/splits"))
    result_dir = Path(paths.get("result_dir", "data/results"))
    return {
        "gold": split_dir / f"{split}.jsonl",
        "outputs": result_dir / f"model_outputs_{model}_{split}.jsonl",
        "rows": result_dir / f"score_rows_{model}_{split}.jsonl",
        "scores": result_dir / f"scores_{model}_{split}.json",
        "detailed": result_dir / f"detailed_scores_{model}_{split}.json",
        "audit": result_dir / f"evaluation_audit_{model}_{split}.json",
        "common": result_dir / f"common_scores_{model}_{split}.json",
        "sensitivity": result_dir / f"full_gold_sensitivity_{model}_{split}.json",
        "answer_audit": result_dir / f"answer_audit_{model}_{split}.jsonl",
    }


def answer_audit_sample(rows, size, seed):
    if size <= 0:
        return []

    credited = [row for row in rows if row.get("final_answer_method") == "gold_token_containment"]
    rejected = [
        row for row in rows
        if not row.get("final_answer_correct")
        and row.get("final_answer_token_recall") is not None
        and row.get("final_answer_token_recall", 0) >= 0.6
    ]
    rng = random.Random(seed)
    rng.shuffle(credited)
    rng.shuffle(rejected)

    half = size // 2
    selected = credited[:half] + rejected[: size - half]
    rng.shuffle(selected)
    # id() is fine here: credited and rejected are disjoint, and selected is a slice of them.
    credited_ids = {id(row) for row in credited}
    return [
        {
            "audit_stratum": "credited_token_containment" if id(row) in credited_ids else "rejected_high_overlap",
            **row,
        }
        for row in selected
    ]


def summary(metrics):
    return {
        "evaluation_schema_version": "3.0",
        **metrics["overall"],
        "case_macro": metrics["case_macro"],
        "conditional_diagnosis": metrics["conditional_diagnosis"],
        "answer_sensitivity": metrics["answer_sensitivity"],
    }


def sensitivity(rows, n_gold):
    faa = sum(bool(row.get("final_answer_correct")) for row in rows)
    vja = sum(bool(row.get("verifier_judgment_correct")) for row in rows)
    ars = sum(
        bool(row.get("final_answer_correct")) and bool(row.get("verifier_judgment_correct"))
        for row in rows
    )
    return {
        "policy": "missing_outputs_count_as_incorrect",
        "N": n_gold,
        "n_matched": len(rows),
        "n_missing": n_gold - len(rows),
        "coverage_adjusted_FAA": safe_divide(faa, n_gold),
        "coverage_adjusted_VJA": safe_divide(vja, n_gold),
        "coverage_adjusted_ARS": safe_divide(ars, n_gold),
    }


def _bound_difference(left, right):
    lower = None if left["L"] is None or right["U"] is None else left["L"] - right["U"]
    upper = None if left["U"] is None or right["L"] is None else left["U"] - right["L"]
    return {"L": lower, "U": upper}


def matcher_pairwise(common_results):
    models = sorted(common_results)
    pairs = {}
    for i, left in enumerate(models):
        for right in models[i + 1:]:
            pair = {}
            for policy in MATCHER_POLICIES:
                left_result = common_results[left][policy]
                right_result = common_results[right][policy]
                values = {
                    metric: _bound_difference(left_result[metric], right_result[metric])
                    for metric in MATCHER_METRICS
                }
                values["unresolved_rate"] = {
                    "point_difference": (
                        left_result["unresolved"]["percentage"] - right_result["unresolved"]["percentage"]
                    )
                }
                pair[policy] = values
            pairs[f"{left}__vs__{right}"] = {
                "left_model": left,
                "right_model": right,
                "difference_definition": "left_minus_right",
                "metrics": pair,
            }
    return pairs


def compact_ids(value):
    compacted = {}
    for key, item in value.items():
        if key in {"missing_ids", "unexpected_ids"}:
            compacted[key] = len(item)
        else:
            compacted[key] = item
    return compacted


def _unknown_counts(unknown, label):
    source = unknown[label]
    return {key: source[key] for key in UNKNOWN_KEYS}


def audit_summary(accounting, audit, rows):
    unknown = audit["unknown_taxonomy"]
    parsed = sum(row.get("adjudication_parse_success") is True for row in rows)
    return {
        "evaluation_schema_version": "3.0",
        **accounting,
        "n_score_rows": len(rows),
        "n_parsed_rows": parsed,
        "unknown_gold_taxonomy": _unknown_counts(unknown, "unknown_gold_taxonomy_labels"),
        "unknown_predicted_fallacy": _unknown_counts(unknown, "unknown_predicted_fallacy_labels"),
        "unknown_predicted_hallucination": _unknown_counts(unknown, "unknown_predicted_hallucination_labels"),
        "unsupported_judgment_predictions": audit["unsupported_judgments"]["count"],
        "c3_judgment_mismatches": audit["c3_judgments"]["n_mismatches"],
        "base_question_clusters": audit["base_question_clusters"],
        "metric_consistency": audit["metric_consistency"],
    }


def display(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(number, title, groups):
    log(f"\n{number}. {title}")
    log(f"{'group':<25}" + "".join(f"{name:>10}" for name in TABLE_FIELDS))
    for group, values in groups.items():
        cells = "".join(f"{display(values.get(name)):>10}" for name in TABLE_FIELDS)
        log(f"{group:<25}{cells}")


def print_json(number, title, value):
    log(f"\n{number}. {title}")
    log(json.dumps(value, ensure_ascii=False, indent=2))


def print_report(model, accounting, metrics, common, sens, audit):
    log("\n" + "=" * 100)
    log(f"MODEL: {model}")
    log("=" * 100)
    print_json("1", "Run configuration and row accounting", compact_ids(accounting))
    print_table("2", "Overall adjudication and recovery", {"overall": metrics["overall"]})
    print_table("3", "Results by domain", metrics["by_domain"])
    print_table("4", "Results by case", metrics["by_case"])
    print_table("5", "Results by solver correctness", metrics["by_solver_correctness"])
    print_table("6", "Results by verifier correctness", metrics["by_verifier_correctness"])
    print_json("7", "Verifier judgment", metrics["verifier_judgment"])
    print_json("8", "Answer preservation and truth recovery", metrics["answer_recovery"])
    print_json("9", "Joint adjudication and recovery", metrics["joint_adjudication_recovery"])
    print_json("10", "Recovery-consistency matrix", metrics["recovery_consistency"])
    print_json("11", "Binary fault awareness", metrics["binary_fault_presence"])
    print_json("12", "Hallucination diagnosis", metrics["hallucination_detection"])
    print_json("13", "Fallacy diagnosis", metrics["fallacy_detection"])
    print_json("14", "Diagnosis given correct rejection", metrics["conditional_diagnosis"])
    print_table("15", "Common-item overall results", {"overall": common["overall"]})
    print_table("16", "Common-item results by domain", common["by_domain"])
    print_table("17", "Common-item results by case", common["by_case"])
    print_json("18", "Common-item recovery metrics", common["answer_recovery"])
    print_json("19", "Case-macro results", common["case_macro"])
    print_json("20", "Missing-output-as-incorrect sensitivity", sens)
    print_json("21", "Audit summary", compact_ids(audit))


def parse_args():
    parser = argparse.ArgumentParser(description="Score Step 08 outputs and write Step 09 reports.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--split")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("--bootstrap-batch-size", type=int)
    parser.add_argument("--answer-audit-size", type=int, default=0)
    parser.add_argument("--answer-audit-seed", type=int, default=2027)
    parser.add_argument("--skip-statistics", action="store_true")
    return parser.parse_args()


def load_outputs(config, model_keys, split, gold_ids):
    outputs = {}
    output_ids = {}
    accounting = {}
    for model in model_keys:
        path = output_paths(config, model, split)["outputs"]
        outputs[model] = index_by_id(load_jsonl(path))
        output_ids[model] = set(outputs[model])
        accounting[model] = compare_id_sets(gold_ids, output_ids[model])
        matched = accounting[model]["n_matched"]
        missing = accounting[model]["n_missing"]
        log(f"{model}: matched={matched} missing={missing}")
    return outputs, output_ids, accounting


def score_model(model, gold, outputs, accounting, common_ids, fallacies, hallucinations, model_name):
    matched = sorted(set(gold) & set(outputs))
    rows = []
    step = max(1000, len(matched) // 10)
    for i, item_id in enumerate(matched, 1):
        rows.append(build_score_row(
            gold[item_id],
            outputs[item_id],
            model,
            model_name,
            fallacies,
            hallucinations,
        ))
        if i % step == 0 or i == len(matched):
            log(f"[SCORE] {model}: {i}/{len(matched)}")

    common_rows = filter_rows_by_ids(rows, common_ids)
    metrics = compute_detailed_metrics(rows, fallacies, hallucinations)
    common = compute_detailed_metrics(common_rows, fallacies, hallucinations)
    audit = run_evaluation_audits(rows)
    sens = sensitivity(rows, len(gold))
    sens["available_output_matcher_ladder"] = metrics["answer_sensitivity"]
    sens["coverage_adjusted_matcher_ladder"] = coverage_adjusted_answer_sensitivity(rows, list(gold.values()))
    return {
        "rows": rows,
        "common_rows": common_rows,
        "metrics": metrics,
        "common": common,
        "audit": audit,
        "sensitivity": sens,
        "audit_summary": audit_summary(accounting, audit, rows),
    }


def write_model_outputs(config, model, split, scored, answer_audit_size, answer_audit_seed):
    result_dir = Path(config.get("paths", {}).get("result_dir", "data/results"))
    paths = output_paths(config, model, split)
    audit = scored["audit"]
    write_jsonl_atomic(paths["rows"], scored["rows"])
    write_json(paths["scores"], summary(scored["metrics"]))
    write_json(paths["detailed"], scored["metrics"])
    write_json(paths["common"], scored["common"])
    write_json(paths["sensitivity"], scored["sensitivity"])
    write_json(paths["audit"], {"summary": scored["audit_summary"], "details": audit})
    if answer_audit_size > 0:
        sample = answer_audit_sample(scored["rows"], answer_audit_size, answer_audit_seed)
        write_jsonl_atomic(paths["answer_audit"], sample)

    unknown_rows = []
    for group in audit["unknown_taxonomy"].values():
        unknown_rows.extend(group["affected_rows"])
    write_jsonl_atomic(result_dir / f"audit_unknown_taxonomy_{model}_{split}.jsonl", unknown_rows)
    write_jsonl_atomic(result_dir / f"audit_c3_semantic_{model}_{split}.jsonl", audit["c3_judgments"]["mismatch_rows"])
    write_jsonl_atomic(
        result_dir / f"audit_unsupported_judgments_{model}_{split}.jsonl",
        audit["unsupported_judgments"]["affected_rows"],
    )


def bootstrap_settings(args, config):
    if args.skip_statistics:
        n_resamples = 0
    elif args.bootstrap_resamples is not None:
        n_resamples = args.bootstrap_resamples
    else:
        n_resamples = int(setting(config, "bootstrap_resamples", 1000))

    seed = args.bootstrap_seed
    if seed is None:
        seed = int(setting(config, "bootstrap_seed", 2027))
    batch = args.bootstrap_batch_size
    if batch is None:
        batch = int(setting(config, "bootstrap_batch_size", 25))
    return n_resamples, seed, batch


def run_statistics(common_rows_by_model, fallacies, hallucinations, n_resamples, seed, batch):
    if n_resamples <= 0:
        stats = {
            "status": "disabled",
            "bootstrap_confidence_intervals": {},
            "paired_significance_tests": {},
        }
        matcher_stats = {
            "status": "disabled",
            "confidence_intervals": {},
            "paired_differences": {},
        }
        return stats, matcher_stats

    stats = compute_all_statistical_tests(
        common_rows_by_model,
        fallacies,
        hallucinations,
        n_resamples=n_resamples,
        seed=seed,
        confidence=0.95,
        batch_size=batch,
        progress=lambda message: log(f"[STATS] {message}"),
    )
    matcher_stats = compute_matcher_ladder_bootstrap(
        common_rows_by_model,
        n_resamples=n_resamples,
        seed=seed,
        confidence=0.95,
        batch_size=batch,
    )
    return stats, matcher_stats


def main():
    args = parse_args()
    started = time.perf_counter()
    log("=" * 100)
    log("FaultyVerifierBench Step 09")
    log("=" * 100)

    config = load_config(args.config)
    split = args.split or setting(config, "default_split", "test")
    models = load_models(setting(config, "models_path", "config/models.yaml"))
    model_keys = selected_models(config, models, args.models)

    result_dir = Path(config.get("paths", {}).get("result_dir", "data/results"))
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = start_log(result_dir, split)
    log(f"Terminal output is also written to {log_path}")

    fallacies, hallucinations = load_benchmark_taxonomies(ROOT)
    gold_path = output_paths(config, model_keys[0], split)["gold"]
    gold = index_by_id(load_jsonl(gold_path))
    gold_ids = set(gold)
    log(f"[LOAD] Gold rows: {len(gold)}")

    outputs, output_ids, accounting = load_outputs(config, model_keys, split, gold_ids)
    common_ids = get_common_item_ids(gold_ids, output_ids)
    write_json(result_dir / f"common_item_ids_{split}.json", build_common_item_summary(gold_ids, output_ids))
    log(f"Common rows: {len(common_ids)} rows")

    common_rows_by_model = {}
    all_results = {}
    for index, model in enumerate(model_keys, 1):
        log(f"[MODEL {index}/{len(model_keys)}] {model}")
        scored = score_model(
            model,
            gold,
            outputs[model],
            accounting[model],
            common_ids,
            fallacies,
            hallucinations,
            models[model].get("model_name", model),
        )
        write_model_outputs(config, model, split, scored, args.answer_audit_size, args.answer_audit_seed)
        common_rows_by_model[model] = scored["common_rows"]
        all_results[model] = scored
        print_report(
            model,
            accounting[model],
            scored["metrics"],
            scored["common"],
            scored["sensitivity"],
            scored["audit_summary"],
        )

    n_resamples, seed, batch = bootstrap_settings(args, config)
    stats, matcher_stats = run_statistics(
        common_rows_by_model, fallacies, hallucinations, n_resamples, seed, batch
    )
    common_matcher_sensitivity = {
        model: all_results[model]["common"]["answer_sensitivity"] for model in model_keys
    }
    matcher_paired = matcher_pairwise(common_matcher_sensitivity)

    write_json(
        result_dir / f"bootstrap_confidence_intervals_{split}.json",
        {**stats["bootstrap_confidence_intervals"], "matcher_ladder": matcher_stats["confidence_intervals"]},
    )
    write_json(
        result_dir / f"paired_significance_tests_{split}.json",
        {
            **stats["paired_significance_tests"],
            "matcher_ladder": matcher_paired,
            "matcher_ladder_bootstrap": matcher_stats["paired_differences"],
        },
    )
    if n_resamples > 0:
        for model in model_keys:
            print_json(
                "22",
                f"{model} clustered-bootstrap confidence intervals",
                stats["bootstrap_confidence_intervals"][model],
            )
        print_json("23", "Paired clustered-bootstrap differences", stats["paired_significance_tests"])
    print_json("24", "Common-item matcher sensitivity", common_matcher_sensitivity)
    print_json("25", "Matcher-ladder paired identification ranges", matcher_paired)
    print_json("26", "Matcher-ladder clustered bootstrap confidence intervals", matcher_stats["confidence_intervals"])
    print_json("27", "Matcher-ladder clustered bootstrap paired differences", matcher_stats["paired_differences"])

    model_summaries = {}
    for model in model_keys:
        result = all_results[model]
        model_summaries[model] = {
            "accounting": accounting[model],
            "available": summary(result["metrics"]),
            "common": summary(result["common"]),
            "sensitivity": result["sensitivity"],
            "audit_summary": result["audit_summary"],
        }
    write_json(
        result_dir / f"scores_summary_{split}.json",
        {
            "evaluation_schema_version": "3.0",
            "split": split,
            "common_item_N": len(common_ids),
            "common_matcher_sensitivity": common_matcher_sensitivity,
            "matcher_ladder_paired_differences": matcher_paired,
            "matcher_ladder_bootstrap": matcher_stats,
            "models": model_summaries,
            "statistics": stats,
        },
    )
    log(f"Step 09 complete in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
