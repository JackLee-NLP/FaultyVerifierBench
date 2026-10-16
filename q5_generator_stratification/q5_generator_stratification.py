#!/usr/bin/env python3
"""Stratify adjudicator performance by benchmark generator identity.

The script joins frozen test-split provenance to Step 09 score rows by instance
ID. It reports FAA, VJA, TRR, and ARS for generator strata and evaluates the
DeepSeek adjudicator on rows with no DeepSeek involvement in question, solver,
or verifier generation.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
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
    no_deepseek = [
        row for row in model_rows[args.adjudicator]
        if str(row.get("id")) in no_deepseek_ids
    ]

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
            "N": len(no_deepseek),
            "metrics": metrics(no_deepseek),
        },
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
