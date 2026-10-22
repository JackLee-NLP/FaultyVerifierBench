#!/usr/bin/env python3
"""Run independent, anonymous LLM correctness annotations with append-only resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.client_factory import get_client

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("PyYAML is required to load config/models.yaml") from exc

DEFAULT_JUDGES = ("gpt_5_6_luna", "deepseek", "qwen3_8_flash")
VALID_LABELS = {"correct", "incorrect"}
SYSTEM_PROMPT = """You are an independent correctness judge.
Judge each anonymous answer independently against the question and reference
answer. Return JSON only, with this exact shape:
{"judgments":[{"label":"A1","judgment":"correct"},{"label":"A2","judgment":"incorrect"}]}
Use only the labels supplied in the request. Do not merge, rank, or compare
answers. Mark an answer correct when it is substantively correct and answers
the question; do not require wording identical to the reference."""


def parse_input_spec(value: str) -> tuple[str | None, Path]:
    if "=" in value:
        model, raw_path = value.split("=", 1)
        model = model.strip() or None
    else:
        model, raw_path = None, value
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    return model, path


def load_rows(specs: list[str], answer_field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_override, path in (parse_input_spec(spec) for spec in specs):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                instance_id = str(row.get("instance_id") or row.get("id") or "").strip()
                answer = row.get(answer_field)
                if not instance_id or not isinstance(answer, str):
                    raise ValueError(
                        f"{path}:{line_number} requires a non-empty id and "
                        f"string {answer_field!r}"
                    )
                question = str(row.get("question") or "")
                if not question or "gold_answer" not in row or "gold_explanation" not in row:
                    raise ValueError(
                        f"{path}:{line_number} requires question, gold_answer, "
                        "and gold_explanation"
                    )
                source_model = (
                    model_override or row.get("model") or row.get("source_model")
                    or row.get("target_model") or row.get("verifier_generator") or path.stem
                )
                rows.append({
                    "question_key": str(row.get("question_id") or question),
                    "question": question,
                    "gold_answer": row["gold_answer"],
                    "gold_explanation": row["gold_explanation"],
                    "answer": answer,
                    "model": str(source_model),
                    "instance_id": instance_id,
                })
    if not rows:
        raise ValueError("No input rows found")
    return rows


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        answer_text = row["answer"].strip()
        key_material = json.dumps(
            [row["question_key"], answer_text], ensure_ascii=False, separators=(",", ":")
        )
        key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
        if key not in groups:
            groups[key] = {
                "answer_id": f"ans_{len(order) + 1:06d}",
                "question_key": row["question_key"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "gold_explanation": row["gold_explanation"],
                "answer": answer_text,
                "original_mappings": [],
            }
            order.append(key)
        groups[key]["original_mappings"].append(
            {"model": row["model"], "instance_id": row["instance_id"]}
        )
    return [groups[key] for key in order]


def make_batches(answers: list[dict[str, Any]], seed: int, batch_size: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for answer in answers:
        grouped[answer["question_key"]].append(answer)
    batches: list[dict[str, Any]] = []
    for question_key, question_answers in grouped.items():
        for start in range(0, len(question_answers), batch_size):
            chunk = question_answers[start:start + batch_size]
            shuffled = list(chunk)
            random.Random(f"{seed}:{question_key}:{start}").shuffle(shuffled)
            entries = [
                {"label": f"A{index}", "answer_id": item["answer_id"],
                 "answer": item["answer"], "original_mappings": item["original_mappings"]}
                for index, item in enumerate(shuffled, 1)
            ]
            batches.append({
                "request_id": f"{question_key}:{start // batch_size + 1}",
                "question_key": question_key,
                "question": chunk[0]["question"],
                "gold_answer": chunk[0]["gold_answer"],
                "gold_explanation": chunk[0]["gold_explanation"],
                "answers": entries,
            })
    return batches


def user_prompt(batch: dict[str, Any]) -> str:
    answers = "\n\n".join(f'{entry["label"]}: {entry["answer"]}' for entry in batch["answers"])
    return (
        f"Question:\n{batch['question']}\n\n"
        f"Reference answer:\n{batch['gold_answer']}\n\n"
        f"Reference explanation:\n{batch['gold_explanation']}\n\n"
        f"Anonymous answers:\n{answers}\n\n"
        "Return one correct/incorrect judgment for every anonymous answer."
    )


def parse_judgments(raw: str, labels: set[str]) -> dict[str, str]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned invalid JSON: {exc}") from exc
    judgments = parsed.get("judgments") if isinstance(parsed, dict) else None
    if not isinstance(judgments, list):
        raise ValueError("Judge response must contain a judgments list")
    result: dict[str, str] = {}
    for item in judgments:
        if not isinstance(item, dict):
            raise ValueError("Every judgment must be an object")
        label = str(item.get("label") or "")
        judgment = str(item.get("judgment") or "").strip().lower()
        if label not in labels or judgment not in VALID_LABELS or label in result:
            raise ValueError(f"Invalid or duplicate judgment: {item!r}")
        result[label] = judgment
    if set(result) != labels:
        raise ValueError(f"Judge did not return exactly one result per answer; missing={sorted(labels - set(result))}")
    return result


def load_model_config(path: Path, judge: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    try:
        return config["models"][judge]
    except KeyError as exc:
        raise KeyError(f"Judge {judge!r} is missing from {path}") from exc


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write derived input manifests only; never use this for judge results."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Persist one attempt without replacing any existing result."""
    payload = json.dumps(record, ensure_ascii=False) + "\n"
    needs_newline = False
    if path.is_file() and path.stat().st_size:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            needs_newline = handle.read(1) != b"\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(("\n" if needs_newline else "") + payload)
        handle.flush()
        os.fsync(handle.fileno())


def record_answer_ids(record: dict[str, Any]) -> list[str]:
    entries = record.get("judgments") or record.get("missing_labels") or []
    return sorted(str(entry["answer_id"]) for entry in entries
                  if isinstance(entry, dict) and entry.get("answer_id"))


def batch_fingerprint(batch: dict[str, Any]) -> str:
    """Detect changed content, references, labels, or mappings in new records."""
    payload = json.dumps(
        {"system_prompt": SYSTEM_PROMPT, "batch": batch},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_history(path: Path) -> dict[str, list[dict[str, Any]]]:
    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.is_file():
        return history
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}. Existing output was NOT "
                    "overwritten. Back up the file and repair/remove only the damaged "
                    "line before restarting. --no-resume does not bypass this check."
                ) from exc
            if not isinstance(record, dict) or not record.get("request_id"):
                raise ValueError(f"{path}:{line_number}: expected an object with request_id")
            history[str(record["request_id"])].append(record)
    return history


def completed_matches(record: dict[str, Any], batch: dict[str, Any], judge: str) -> bool:
    # Legacy successful records have no status or fingerprint, as in the example.
    if record.get("status", "completed") != "completed":
        return False
    if record.get("judge", judge) != judge:
        return False
    if record.get("question_key", batch["question_key"]) != batch["question_key"]:
        return False
    entries = record.get("judgments")
    if not isinstance(entries, list) or len(entries) != len(batch["answers"]):
        return False
    if any(not isinstance(item, dict) or item.get("judgment") not in VALID_LABELS for item in entries):
        return False
    expected = {item["answer_id"]: item for item in batch["answers"]}
    if record_answer_ids(record) != sorted(expected):
        return False
    # Mappings also guard against sequential answer IDs being reassigned.
    for item in entries:
        if item.get("original_mappings") != expected[item["answer_id"]]["original_mappings"]:
            return False
    fingerprint = record.get("request_fingerprint")
    return fingerprint is None or fingerprint == batch_fingerprint(batch)


def run(args: argparse.Namespace) -> None:
    rows = load_rows(args.input, args.answer_field)
    answers = deduplicate(rows)
    batches = make_batches(answers, args.seed, args.batch_size)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    # These are reproducible manifests, not the append-only judgment history.
    write_jsonl(output_dir / "deduplicated_answers.jsonl", answers)
    write_jsonl(output_dir / "requests.jsonl", batches)
    if args.dry_run:
        print(f"[DRY RUN] answers={len(answers)} batches={len(batches)}")
        return

    for judge in args.judges:
        judge_path = output_dir / f"judgments_{judge}.jsonl"
        history = load_history(judge_path)
        pending = []
        skipped = 0
        for batch in batches:
            # Any matching success wins, even if a later attempt failed.
            if args.resume and any(
                completed_matches(record, batch, judge)
                for record in history.get(batch["request_id"], [])
            ):
                skipped += 1
            else:
                pending.append(batch)
        print(
            f"[RESUME] judge={judge} total_batches={len(batches)} "
            f"skipped_completed={skipped} pending={len(pending)}",
            flush=True,
        )
        if not args.resume:
            print("[NOTICE] --no-resume reruns all batches but APPENDS; existing results are preserved.", flush=True)
        if not pending:
            print(f"[DONE] judge={judge} all {len(batches)} batches already completed; no API calls.", flush=True)
            continue

        cfg = load_model_config(Path(args.models_config), judge)
        client = get_client(
            provider=cfg["provider"], model_name=cfg["model_name"],
            temperature=args.temperature, max_output_tokens=args.max_output_tokens,
            deepseek_thinking_enabled=True, deepseek_reasoning_effort="high",
        )
        succeeded = failed = 0
        for index, batch in enumerate(pending, 1):
            base = {
                "judge": judge, "request_id": batch["request_id"],
                "question_key": batch["question_key"],
                "answer_ids": sorted(entry["answer_id"] for entry in batch["answers"]),
                "request_fingerprint": batch_fingerprint(batch),
            }
            print(f"[GENERATE] judge={judge} pending={index}/{len(pending)} request={batch['request_id']}", flush=True)
            try:
                raw = client.generate(SYSTEM_PROMPT, user_prompt(batch))
                parsed = parse_judgments(raw, {entry["label"] for entry in batch["answers"]})
                result = {
                    **base, "status": "completed",
                    "judgments": [
                        {"label": entry["label"], "judgment": parsed[entry["label"]],
                         "answer_id": entry["answer_id"], "original_mappings": entry["original_mappings"]}
                        for entry in batch["answers"]
                    ],
                }
            except Exception as error:
                result = {
                    **base, "status": "failed", "error": repr(error),
                    "missing_labels": [
                        {"label": entry["label"], "answer_id": entry["answer_id"],
                         "original_mappings": entry["original_mappings"]}
                        for entry in batch["answers"]
                    ],
                }
                print(f"[FAILED] judge={judge} request={batch['request_id']}: {error!r}", flush=True)
            # Write failures are deliberately not caught: stop rather than continue
            # generating results that cannot be saved.
            append_jsonl(judge_path, result)
            if result["status"] == "completed":
                succeeded += 1
                print(f"[SAVED] judge={judge} request={batch['request_id']}", flush=True)
            else:
                failed += 1
        print(
            f"[DONE] judge={judge} answers={len(answers)} batches={len(batches)} "
            f"skipped={skipped} new_completed={succeeded} new_failed={failed}", flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="MODEL=JSONL")
    parser.add_argument("--output", default="llm_annotation/results")
    parser.add_argument("--models-config", default="config/models.yaml")
    parser.add_argument("--answer-field", default="verifier_response")
    parser.add_argument("--judge", dest="judges", action="append", default=None)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=4096,
                        help="Combined visible-output/reasoning budget; use 4096+ with reasoning enabled.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_false", dest="resume",
                        help="Rerun all batches and append attempts. Does NOT overwrite existing results.")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 12:
        parser.error("--batch-size must be between 1 and 12")
    if args.judges is None:
        args.judges = list(DEFAULT_JUDGES)
    run(args)


if __name__ == "__main__":
    main()
