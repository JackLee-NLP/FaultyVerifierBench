#!/usr/bin/env python3
"""Stratify adjudicator performance by benchmark generator identity.

The script joins frozen test-split provenance to Step 09 score rows by instance
ID. It reports FAA, VJA, TRR, and ARS for generator strata and evaluates every
adjudicator on rows with no DeepSeek involvement in question, solver, or
verifier generation. It also fits shallow verifier-text baselines on the
development split and scores them on the test split.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

METRICS = ("FAA", "VJA", "TRR", "ARS")
PROVENANCE_FIELDS = (
    "question_generator",
    "solver_generator",
    "verifier_generator",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rate(rows: list[dict[str, Any]], predicate) -> float | None:
    return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else None


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    solver_incorrect = [row for row in rows if row.get("solver_correctness") == "incorrect"]
    return {
        "N": len(rows),
        "FAA": rate(rows, lambda row: row.get("final_answer_correct")),
        "VJA": rate(rows, lambda row: row.get("verifier_judgment_correct")),
        "TRR": rate(solver_incorrect, lambda row: row.get("final_answer_correct")),
        "TRR_support": len(solver_incorrect),
        "ARS": rate(
            rows,
            lambda row: row.get("final_answer_correct")
            and row.get("verifier_judgment_correct"),
        ),
    }


def join_rows(
    split_rows: Iterable[dict[str, Any]],
    score_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    provenance = {str(row["id"]): row for row in split_rows}
    joined = []
    for score in score_rows:
        item_id = str(score.get("id", ""))
        source = provenance.get(item_id)
        if source is None:
            continue
        joined.append({**score, **{field: source.get(field) for field in PROVENANCE_FIELDS}})
    return joined


def is_deepseek(value: Any) -> bool:
    return "deepseek" in str(value or "").strip().lower()


def verifier_label(row: dict[str, Any]) -> int:
    return int(row.get("verifier_correctness") == "correct")


def word_count(row: dict[str, Any]) -> int:
    return len(str(row.get("verifier_response") or "").split())


def classification_metrics(gold: list[int], predicted: list[int]) -> dict[str, Any]:
    positive = sum(gold)
    return {
        "N": len(gold),
        "VJA": accuracy_score(gold, predicted) if gold else None,
        "balanced_accuracy": balanced_accuracy_score(gold, predicted) if gold else None,
        "correct_class_F1": f1_score(gold, predicted, pos_label=1, zero_division=0) if gold else None,
        "majority_VJA": max(positive, len(gold) - positive) / len(gold) if gold else None,
    }


def fit_length_threshold(rows: list[dict[str, Any]]) -> int:
    gold = [verifier_label(row) for row in rows]
    counts = [word_count(row) for row in rows]
    best_accuracy = -1.0
    best_threshold = 1
    for threshold in range(1, max(counts, default=1) + 1):
        predicted = [int(count < threshold) for count in counts]
        accuracy = accuracy_score(gold, predicted)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold


def shallow_vja_baselines(
    dev_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    common_ids: set[str],
    model_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Fit surface baselines on dev and score them without refitting on test."""
    threshold = fit_length_threshold(dev_rows)
    classifier = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000),
        LogisticRegression(max_iter=200, C=1.0, solver="liblinear"),
    )
    classifier.fit(
        [str(row.get("verifier_response") or "") for row in dev_rows],
        [verifier_label(row) for row in dev_rows],
    )

    def score_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
        gold = [verifier_label(row) for row in rows]
        texts = [str(row.get("verifier_response") or "") for row in rows]
        length_pred = [int(word_count(row) < threshold) for row in rows]
        lexical_pred = classifier.predict(texts).tolist() if rows else []
        return {
            "majority_faulty": classification_metrics(gold, [0] * len(gold)),
            "message_length": classification_metrics(gold, length_pred),
            "tfidf_word_bigram_logistic_regression": classification_metrics(gold, lexical_pred),
        }

    common_rows = [row for row in test_rows if str(row.get("id")) in common_ids]
    return {
        "text": "verifier_response",
        "label": "verifier_correctness == correct",
        "fit_split": "dev",
        "question_overlap_with_test": len(
            {row.get("question_id") for row in dev_rows}
            & {row.get("question_id") for row in test_rows}
        ),
        "length_rule": f"predict correct if verifier_response has fewer than {threshold} words",
        "length_threshold_words": threshold,
        "dev": score_split(dev_rows),
        "test": score_split(test_rows),
        "common_completed_items": score_split(common_rows),
        "adjudicator_VJA_on_common_completed_items": {
            model: metrics(
                [row for row in rows if str(row.get("id")) in common_ids]
            )["VJA"]
            for model, rows in model_rows.items()
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--result-dir", type=Path, default=Path("data/results"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "deepseek",
            "qwen3_4b_local",
            "qwen3_8_flash",
            "qwen3_8b_local",
            "Ministral-3-14B",
            "gpt_5_6_luna",
        ],
    )
    parser.add_argument("--adjudicator", default="deepseek")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/q5_generator_stratification_test.json"),
    )
    args = parser.parse_args()

    split_rows = read_jsonl(args.split_dir / f"{args.split}.jsonl")
    dev_rows = read_jsonl(args.split_dir / f"{args.dev_split}.jsonl")
    by_stratum: dict[str, set[str]] = defaultdict(set)
    model_rows: dict[str, list[dict[str, Any]]] = {}

    for model in args.models:
        score_path = args.result_dir / f"score_rows_{model}_{args.split}.jsonl"
        if not score_path.is_file():
            raise FileNotFoundError(score_path)
        rows = join_rows(split_rows, read_jsonl(score_path))
        model_rows[model] = rows
        for field in PROVENANCE_FIELDS:
            for value in sorted({str(row.get(field) or "unknown") for row in rows}):
                by_stratum[f"{field}={value}"].update(
                    str(row["id"])
                    for row in rows
                    if str(row.get(field) or "unknown") == value
                )

    no_deepseek_ids = {
        str(row["id"])
        for row in split_rows
        if all(not is_deepseek(row.get(field)) for field in PROVENANCE_FIELDS)
    }
    no_deepseek_by_model = {
        model: [row for row in rows if str(row.get("id")) in no_deepseek_ids]
        for model, rows in model_rows.items()
    }
    shared_no_deepseek_ids = set.intersection(
        *(
            {str(row.get("id")) for row in rows}
            for rows in no_deepseek_by_model.values()
        )
    ) if no_deepseek_by_model else set()
    common_ids_path = args.result_dir / f"common_item_ids_{args.split}.json"
    common_ids = set(
        json.loads(common_ids_path.read_text(encoding="utf-8")).get("common_item_ids", [])
    ) if common_ids_path.is_file() else set()

    report = {
        "split": args.split,
        "models": args.models,
        "provenance_fields": list(PROVENANCE_FIELDS),
        "strata": {
            stratum: {
                model: metrics(
                    [row for row in model_rows[model] if str(row.get("id")) in rows]
                )
                for model in args.models
            }
            for stratum, rows in sorted(by_stratum.items())
        },
        "no_deepseek_generation_subset": {
            "adjudicator": args.adjudicator,
            "criterion": "question_generator, solver_generator, and verifier_generator are all non-DeepSeek",
            "N": len(no_deepseek_by_model[args.adjudicator]),
            "metrics": metrics(no_deepseek_by_model[args.adjudicator]),
            "by_adjudicator": {
                model: metrics(rows)
                for model, rows in no_deepseek_by_model.items()
            },
            "shared_by_all_adjudicators": {
                "N": len(shared_no_deepseek_ids),
                "by_adjudicator": {
                    model: metrics(
                        [row for row in rows if str(row.get("id")) in shared_no_deepseek_ids]
                    )
                    for model, rows in model_rows.items()
                },
            },
        },
        "shallow_vja_baselines": shallow_vja_baselines(
            dev_rows, split_rows, common_ids, model_rows
        ),
        "coverage": {
            model: len(rows)
            for model, rows in model_rows.items()
        },
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[DONE] Wrote {args.output}")


if __name__ == "__main__":
    main()
