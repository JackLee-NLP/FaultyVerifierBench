#!/usr/bin/env python3
"""Drop failed judgment lines that were later completed for the same request.

A failed line is removed only when a completed line has the same judge,
request_id, and answer_ids. Failures with no completed replacement are kept.
Completed lines and their order are unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def answer_ids(record: dict[str, Any]) -> tuple[str, ...]:
    raw = record.get("answer_ids")
    if not isinstance(raw, list):
        entries = record.get("judgments") or record.get("missing_labels") or []
        raw = [
            entry.get("answer_id")
            for entry in entries
            if isinstance(entry, dict) and entry.get("answer_id")
        ]
    return tuple(sorted(str(item) for item in raw))


def identity(record: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    return (
        str(record.get("judge") or ""),
        str(record.get("request_id") or ""),
        answer_ids(record),
    )


def load_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return handle.readlines()


def completed_identities(lines: list[str], path: Path) -> set[tuple[str, str, tuple[str, ...]]]:
    found: set[tuple[str, str, tuple[str, ...]]] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        if str(record.get("status") or "completed") == "completed":
            found.add(identity(record))
    return found


def prune(lines: list[str], path: Path) -> tuple[list[str], int, int]:
    completed = completed_identities(lines, path)
    kept: list[str] = []
    removed = kept_failed = 0
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            kept.append(line if line.endswith("\n") else line + "\n")
            continue
        record = json.loads(line)
        status = str(record.get("status") or "completed")
        if status != "completed" and identity(record) in completed:
            removed += 1
            print(
                f"[REMOVE] {path}:{line_number} request_id={record.get('request_id')} "
                f"status={status}",
                flush=True,
            )
            continue
        if status != "completed":
            kept_failed += 1
        kept.append(line if line.endswith("\n") else line + "\n")
    return kept, removed, kept_failed


def write_atomic(path: Path, lines: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(lines)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(paths: list[Path], dry_run: bool) -> None:
    if not paths:
        raise SystemExit("No judgment files given.")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Judgment file not found: {path}")
        lines = load_lines(path)
        kept, removed, kept_failed = prune(lines, path)
        print(
            f"[DONE] file={path} lines={len(lines)} removed_recovered_failures={removed} "
            f"kept_unrecovered_failures={kept_failed}",
            flush=True,
        )
        if dry_run:
            print("[DRY RUN] file was not modified.", flush=True)
            continue
        if removed:
            write_atomic(path, kept)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files", nargs="*", type=Path,
        help="Judgment JSONL files. Defaults to the Qwen results file.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = args.files or [here / "results_qwen" / "judgments_qwen3_8_flash.jsonl"]
    run(paths, args.dry_run)


if __name__ == "__main__":
    main()
