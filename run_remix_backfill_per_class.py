from __future__ import annotations

import argparse
import json
import os
from typing import Any

if __package__ in {None, ""}:
    import sys

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from validation_acceleration import parse_per_class_from_log
else:
    from .validation_acceleration import parse_per_class_from_log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill per-class proxy metrics from run log.txt into result.json files.")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--result-name", default="result.json")
    parser.add_argument("--log-name", default="log.txt")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _candidate_task_names(payload: dict[str, Any]) -> list[str]:
    return [
        key
        for key, value in payload.items()
        if isinstance(value, dict) and "summary" in value and "validation_mode" in value
    ]


def _backfill_run(result_path: str, log_path: str, *, dry_run: bool) -> bool:
    with open(result_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    with open(log_path, "r", encoding="utf-8") as f:
        per_class = parse_per_class_from_log(f.read())
    if not per_class:
        return False

    touched = False
    for key in _candidate_task_names(payload):
        task_payload = payload[key]
        if task_payload.get("per_class") == per_class:
            continue
        task_payload["per_class"] = per_class
        touched = True

    if touched and not dry_run:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return touched


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    runs_root = os.path.abspath(args.runs_root)
    patched = 0
    skipped = 0

    for entry in sorted(os.listdir(runs_root)):
        run_dir = os.path.join(runs_root, entry)
        if not os.path.isdir(run_dir):
            continue
        result_path = os.path.join(run_dir, args.result_name)
        log_path = os.path.join(run_dir, args.log_name)
        if not (os.path.exists(result_path) and os.path.exists(log_path)):
            skipped += 1
            continue
        if _backfill_run(result_path, log_path, dry_run=bool(args.dry_run)):
            patched += 1
            print(f"patched {result_path}")
        else:
            skipped += 1

    print(f"done patched={patched} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
