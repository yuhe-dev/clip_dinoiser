from __future__ import annotations

import argparse
import os
import sys

if __package__ in {None, ""}:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from slice_remix.dataset import read_jsonl, write_jsonl
    from slice_remix.results import build_result_manifest_rows, load_result_entries
else:
    from .slice_remix.dataset import read_jsonl, write_jsonl
    from .slice_remix.results import build_result_manifest_rows, load_result_entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect remix training result entries into a result manifest.")
    parser.add_argument("--rows-path", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-path", required=True)
    return parser


def run(args: argparse.Namespace) -> int:
    rows = read_jsonl(os.path.abspath(args.rows_path))
    result_entries = load_result_entries(os.path.abspath(args.results_dir))
    manifest_rows = build_result_manifest_rows(rows, result_entries)

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_jsonl(output_path, manifest_rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
