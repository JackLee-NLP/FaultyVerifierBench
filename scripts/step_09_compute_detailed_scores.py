from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.detailed_metrics import (
    compute_detailed_metrics,
    load_json,
    load_jsonl,
    merge_score_rows,
    save_json,
    validate_overall_consistency,
)

DOMAIN_COLUMNS = (
    ("n", "N", 9),
    ("final_answer_accuracy", "FAA", 12),
    ("verifier_judgment_accuracy", "VJA", 12),
    ("fault_detection_accuracy", "FDA", 12),
    ("faulty_verifier_rejection_accuracy", "FVR", 12),
    ("over_acceptance_error_rate", "OAR", 12),
    ("correct_verifier_acceptance_accuracy", "CVA", 12),
    ("over_rejection_error_rate", "ORR", 12),
)
FAULT_COLUMNS = (
    ("n", "N", 9),
    ("final_answer_accuracy", "FAA", 11),
    ("verifier_judgment_accuracy", "VJA", 11),
    ("fault_detection_accuracy", "FDA", 11),
    ("faulty_verifier_rejection_accuracy", "FVR", 11),
    ("over_acceptance_error_rate", "OAR", 11),
)
DETECT_COLUMNS = (
    ("precision", "Precision", 13),
    ("recall", "Recall", 13),
    ("f1", "F1", 13),
)
CSV_FIELDS = [
    "category",
    "model",
    "n",
    "final_answer_accuracy",
    "verifier_judgment_accuracy",
    "fault_detection_accuracy",
    "faulty_verifier_rejection_accuracy",
    "over_acceptance_error_rate",
]
DETECT_FIELDS = ["precision", "recall", "f1", "tp", "fp", "fn", "tn"]


class Tee:
    """Write to the terminal and a report file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Build the detailed Step 09 comparison tables.")
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


def pct(value):
    if value is None:
        return "N/A"
    return f"{100 * float(value):.2f}%"


def cnt(value):
    return f"{int(value or 0):,}"


def get_group(reports, model, key, label):
    return reports.get(model, {}).get(key, {}).get(label, {})


def get_detect(reports, model, key, label):
    return reports.get(model, {}).get(key, {}).get("per_label", {}).get(label, {})


def categories(reports, key):
    labels = set()
    for report in reports.values():
        labels.update(report.get(key, {}))
    return sorted(labels)


def group_support(reports, models, group_key, label):
    return sum(int(get_group(reports, model, group_key, label).get("n", 0)) for model in models)


def format_cell(values, key):
    if key == "n":
        return cnt(values.get(key))
    return pct(values.get(key))


def print_header(title, width, columns, label_width, label_name):
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    header = f"{label_name:<{label_width}}{'Model':<14}"
    header += "".join(f"{name:>col_width}" for _, name, col_width in columns)
    print(header)
    print("-" * width)


def print_row(label, model, values, columns, label_width, show_label):
    name = label if show_label else ""
    cells = "".join(f"{format_cell(values, key):>{col_width}}" for key, _, col_width in columns)
    print(f"{name:<{label_width}}{model:<14}{cells}")


def print_domains(reports, models):
    print_header("RESULTS BY DOMAIN", 150, DOMAIN_COLUMNS, 30, "Domain")
    for label in categories(reports, "by_domain"):
        for i, model in enumerate(models):
            group = get_group(reports, model, "by_domain", label)
            print_row(label, model, group, DOMAIN_COLUMNS, 30, i == 0)
        print("-" * 150)


def print_faults(reports, models, group_key, detect_key, title):
    columns = FAULT_COLUMNS + DETECT_COLUMNS
    print_header(title, 170, columns, 40, "Type")
    shown = False
    for label in categories(reports, group_key):
        if group_support(reports, models, group_key, label) == 0:
            continue
        shown = True
        for i, model in enumerate(models):
            group = get_group(reports, model, group_key, label)
            detected = get_detect(reports, model, detect_key, label)
            print_row(label, model, {**group, **detected}, columns, 40, i == 0)
        print("-" * 170)
    if not shown:
        print("No nonzero-support subtype results found. Check annotation field mapping.")


def write_csv(path, reports, models, group_key, detect_key=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(CSV_FIELDS)
    if detect_key:
        fields += DETECT_FIELDS

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label in categories(reports, group_key):
            if group_support(reports, models, group_key, label) == 0:
                continue
            for model in models:
                group = get_group(reports, model, group_key, label)
                row = {
                    "category": label,
                    "model": model,
                    "n": group.get("n"),
                    "final_answer_accuracy": group.get("final_answer_accuracy"),
                    "verifier_judgment_accuracy": group.get("verifier_judgment_accuracy"),
                    "fault_detection_accuracy": group.get("fault_detection_accuracy"),
                    "faulty_verifier_rejection_accuracy": group.get("faulty_verifier_rejection_accuracy"),
                    "over_acceptance_error_rate": group.get("over_acceptance_error_rate"),
                }
                if detect_key:
                    detected = get_detect(reports, model, detect_key, label)
                    row.update({key: detected.get(key) for key in DETECT_FIELDS})
                writer.writerow(row)


def model_paths(split_dir, result_dir, model, split):
    return {
        "gold": split_dir / f"{split}.jsonl",
        "out": result_dir / f"model_outputs_{model}_{split}.jsonl",
        "rows": result_dir / f"score_rows_{model}_{split}.jsonl",
        "score": result_dir / f"scores_{model}_{split}.json",
    }


def load_model_report(model, split, split_dir, result_dir):
    print("\n" + "=" * 100, flush=True)
    print(f"Starting | model={model}", flush=True)
    print("=" * 100, flush=True)

    paths = model_paths(split_dir, result_dir, model, split)
    if any(not path.exists() for path in paths.values()):
        print(f"{model}: required files missing", flush=True)
        return None

    print("Gold records...", flush=True)
    gold = load_jsonl(paths["gold"])
    print(f"Gold={len(gold):,}", flush=True)

    print("Model outputs...", flush=True)
    outputs = load_jsonl(paths["out"])
    print(f"Outputs={len(outputs):,}", flush=True)

    print("Row scores...", flush=True)
    scores = load_jsonl(paths["rows"])
    print(f"Scores={len(scores):,}", flush=True)

    print("Joining by item ID...", flush=True)
    merged, meta = merge_score_rows(gold, outputs, scores)
    print(
        f"merged={len(merged):,} | missing_gold={meta['n_missing_gold']:,} | "
        f"missing_output={meta['n_missing_output']:,} | missing_score={meta['n_missing_score_row']:,}",
        flush=True,
    )

    report = compute_detailed_metrics(merged, model, split, meta)
    print("Validating overall metrics...", flush=True)
    validate_overall_consistency(report, load_json(paths["score"]))
    print("Passed.", flush=True)

    save_json(result_dir / f"detailed_scores_{model}_{split}.json", report)
    print(f"Complete | model={model} | rows={len(merged):,}", flush=True)
    return report


def write_text_report(path, reports, models):
    original = sys.stdout
    with path.open("w", encoding="utf-8") as handle:
        sys.stdout = Tee(original, handle)
        try:
            print_domains(reports, models)
            print_faults(
                reports,
                models,
                "by_hallucination_type",
                "hallucination_detection",
                "RESULTS BY HALLUCINATION TYPE",
            )
            print_faults(
                reports,
                models,
                "by_fallacy_type",
                "fallacy_detection",
                "RESULTS BY FALLACY TYPE",
            )
        finally:
            sys.stdout = original


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    models = [str(name) for name in config.get("evaluation", {}).get("target_models", [])]
    split = str(config.get("evaluation", {}).get("default_split", "test"))

    paths = config.get("paths", {})
    split_dir = Path(paths.get("split_dir", "data/splits"))
    result_dir = Path(paths.get("result_dir", "data/results"))
    result_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    loaded = []
    for model in models:
        report = load_model_report(model, split, split_dir, result_dir)
        if report is None:
            continue
        reports= report
        loaded.append(model)
    if not loaded:
        raise RuntimeError("No complete model reports")

    report_path = result_dir / f"detailed_experiment_results_{split}.txt"
    write_text_report(report_path, reports, loaded)

    csv_dir = result_dir / "detailed_experiment_csv"
    write_csv(csv_dir / f"domains_{split}.csv", reports, loaded, "by_domain")
    write_csv(
        csv_dir / f"hallucination_types_{split}.csv",
        reports,
        loaded,
        "by_hallucination_type",
        "hallucination_detection",
    )
    write_csv(
        csv_dir / f"fallacy_types_{split}.csv",
        reports,
        loaded,
        "by_fallacy_type",
        "fallacy_detection",
    )
    save_json(result_dir / f"detailed_experiment_results_{split}.json", reports)
    print(f"{report_path}")


if __name__ == "__main__":
    main()
