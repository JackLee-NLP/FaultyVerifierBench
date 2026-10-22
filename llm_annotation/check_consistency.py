#!/usr/bin/env python3
"""Compare LLM judge labels by answer_id, ignoring batch size and order.

A comparison is consistent only when every judge in that comparison has the
same correct/incorrect label for an answer. Any disagreement is inconsistent.
Missing answers are reported separately and are not counted as inconsistent.

Pairwise inconsistency files keep answer_id plus the original benchmark
instance_id mappings so a row can be matched back to benchmark results.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

VALID_LABELS = {"correct", "incorrect"}
JUDGE_ALIASES = {
    "deepseek": "deepseek",
    "gpt": "gpt",
    "gpt_5_6_luna": "gpt",
    "qwen": "qwen",
    "qwen3_8_flash": "qwen",
}
DEFAULT_INPUTS = {
    "deepseek": "results_deepseek/judgments_deepseek.jsonl",
    "gpt": "results_gpt/judgments_gpt_5_6_luna.jsonl",
    "qwen": "results_qwen/judgments_qwen3_8_flash.jsonl",
}


def canonical_judge(name: str) -> str:
    key = name.strip().lower()
    try:
        return JUDGE_ALIASES[key]
    except KeyError as exc:
        known = ", ".join(sorted(JUDGE_ALIASES))
        raise argparse.ArgumentTypeError(
            f"Unknown judge {name!r}. Expected one of: {known}"
        ) from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(record)
    return rows


def index_judgments(path: Path, judge: str) -> dict[str, dict[str, Any]]:
    """Collapse batch records into one judgment per answer_id.

    Later completed records replace earlier ones for the same answer_id.
    Failed records are ignored so a retry can still supply the label.
    """
    indexed: dict[str, dict[str, Any]] = {}
    skipped_failed = 0
    for line_number, record in enumerate(load_jsonl(path), 1):
        status = str(record.get("status") or "completed")
        if status != "completed":
            skipped_failed += 1
            continue
        entries = record.get("judgments")
        if not isinstance(entries, list):
            raise ValueError(f"{path}:{line_number}: completed record has no judgments list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{path}:{line_number}: judgment entry must be an object")
            answer_id = str(entry.get("answer_id") or "").strip()
            label = str(entry.get("judgment") or "").strip().lower()
            if not answer_id or label not in VALID_LABELS:
                raise ValueError(
                    f"{path}:{line_number}: invalid judgment for answer_id={answer_id!r}"
                )
            indexed[answer_id] = {
                "answer_id": answer_id,
                "judgment": label,
                "question_key": record.get("question_key"),
                "request_id": record.get("request_id"),
                "label": entry.get("label"),
                "original_mappings": entry.get("original_mappings") or [],
            }
    print(
        f"[LOAD] judge={judge} file={path} answers={len(indexed)} "
        f"skipped_failed_records={skipped_failed}",
        flush=True,
    )
    return indexed


def instance_ids(mappings: Any) -> list[str]:
    if not isinstance(mappings, list):
        return []
    ids: list[str] = []
    for item in mappings:
        if isinstance(item, dict) and item.get("instance_id"):
            ids.append(str(item["instance_id"]))
    return ids


def compare_pair(
    left_name: str,
    left: dict[str, dict[str, Any]],
    right_name: str,
    right: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    shared = sorted(set(left) & set(right))
    inconsistent: list[dict[str, Any]] = []
    consistent = 0
    for answer_id in shared:
        left_item = left[answer_id]
        right_item = right[answer_id]
        if left_item["judgment"] == right_item["judgment"]:
            consistent += 1
            continue
        mappings = left_item["original_mappings"] or right_item["original_mappings"]
        inconsistent.append({
            "answer_id": answer_id,
            "question_key": left_item.get("question_key") or right_item.get("question_key"),
            "instance_ids": instance_ids(mappings),
            "original_mappings": mappings,
            "judgments": {
                left_name: left_item["judgment"],
                right_name: right_item["judgment"],
            },
            "requests": {
                left_name: left_item.get("request_id"),
                right_name: right_item.get("request_id"),
            },
        })
    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    return {
        "judges": [left_name, right_name],
        "shared": len(shared),
        "consistent": consistent,
        "inconsistent": len(inconsistent),
        "missing": {
            f"only_{left_name}": len(only_left),
            f"only_{right_name}": len(only_right),
        },
        "rows": inconsistent,
        "missing_ids": {
            f"only_{left_name}": only_left,
            f"only_{right_name}": only_right,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_ids(path: Path, rows: list[dict[str, Any]]) -> None:
    """One benchmark instance_id per line, for matching result files."""
    seen: list[str] = []
    present: set[str] = set()
    for row in rows:
        for instance_id in row.get("instance_ids") or []:
            if instance_id not in present:
                present.add(instance_id)
                seen.append(instance_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{item}\n" for item in seen), encoding="utf-8")


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Input must be JUDGE=PATH")
    judge, raw_path = value.split("=", 1)
    return canonical_judge(judge), Path(raw_path)


def run(args: argparse.Namespace) -> None:
    here = Path(__file__).resolve().parent
    selected = [canonical_judge(name) for name in args.judges]
    supplied = {judge: path for judge, path in args.input}
    paths: dict[str, Path] = {}
    for judge in selected:
        raw = supplied.get(judge, here / DEFAULT_INPUTS[judge])
        path = raw if raw.is_absolute() else (Path.cwd() / raw if raw.exists() else here / raw)
        if not path.is_file():
            print(f"[SKIP] judge={judge} file not found: {path}", flush=True)
            continue
        paths[judge] = path
    if len(paths) < 2:
        raise SystemExit("Need at least two existing judgment files to compare.")

    indexed = {judge: index_judgments(path, judge) for judge, path in paths.items()}
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = here / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    judges = [judge for judge in selected if judge in indexed]
    summary: dict[str, Any] = {"judges_loaded": judges, "pairs": {}}
    for index, left_name in enumerate(judges):
        for right_name in judges[index + 1:]:
            pair_key = f"{left_name}_vs_{right_name}"
            result = compare_pair(left_name, indexed[left_name], right_name, indexed[right_name])
            detail_path = output_dir / f"inconsistent_{pair_key}.jsonl"
            ids_path = output_dir / f"inconsistent_{pair_key}_instance_ids.txt"
            missing_path = output_dir / f"missing_{pair_key}.json"
            write_jsonl(detail_path, result["rows"])
            write_ids(ids_path, result["rows"])
            missing_path.write_text(
                json.dumps(result["missing_ids"], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            pair_summary = {
                key: result[key]
                for key in ("judges", "shared", "consistent", "inconsistent", "missing")
            }
            pair_summary["agreement_rate"] = (
                round(result["consistent"] / result["shared"], 4) if result["shared"] else None
            )
            pair_summary["detail_file"] = str(detail_path)
            pair_summary["instance_ids_file"] = str(ids_path)
            summary["pairs"][pair_key] = pair_summary
            print(
                f"[PAIR] {pair_key} shared={result['shared']} "
                f"consistent={result['consistent']} inconsistent={result['inconsistent']} "
                f"missing={result['missing']}",
                flush=True,
            )

    if len(judges) >= 3:
        all_ids = set.intersection(*(set(indexed[judge]) for judge in judges))
        all_inconsistent: list[dict[str, Any]] = []
        label_counts: Counter[str] = Counter()
        for answer_id in sorted(all_ids):
            labels = {judge: indexed[judge][answer_id]["judgment"] for judge in judges}
            signature = "|".join(f"{judge}={labels[judge]}" for judge in judges)
            if len(set(labels.values())) == 1:
                label_counts["consistent"] += 1
                continue
            label_counts[signature] += 1
            sample = indexed[judges[0]][answer_id]
            mappings = sample["original_mappings"]
            all_inconsistent.append({
                "answer_id": answer_id,
                "question_key": sample.get("question_key"),
                "instance_ids": instance_ids(mappings),
                "original_mappings": mappings,
                "judgments": labels,
            })
        write_jsonl(output_dir / "inconsistent_all_three.jsonl", all_inconsistent)
        write_ids(output_dir / "inconsistent_all_three_instance_ids.txt", all_inconsistent)
        summary["all_three"] = {
            "shared": len(all_ids),
            "consistent": label_counts["consistent"],
            "inconsistent": len(all_inconsistent),
        }

    summary_path = output_dir / "consistency_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] summary={summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", type=parse_input, default=[], metavar="JUDGE=JSONL",
        help="Override a judge file. Repeat for each judge. Defaults are used otherwise.",
    )
    parser.add_argument(
        "--judge", dest="judges", action="append", default=None,
        help="Judges to include, in comparison order. Default: deepseek gpt qwen.",
    )
    parser.add_argument(
        "--output", default="consistency",
        help="Directory under llm_annotation for inconsistency files.",
    )
    args = parser.parse_args()
    if args.judges is None:
        args.judges = ["deepseek", "gpt", "qwen"]
    run(args)


if __name__ == "__main__":
    main()
