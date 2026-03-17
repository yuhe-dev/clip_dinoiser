from __future__ import annotations

import argparse
import os
import sys

if __package__ in {None, ""}:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from slice_remix.dataset import read_jsonl, write_jsonl
    from slice_remix.labels import attach_measured_gain
else:
    from .slice_remix.dataset import read_jsonl, write_jsonl
    from .slice_remix.labels import attach_measured_gain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attach measured evaluation gains to remix response rows.")
    parser.add_argument("--rows-path", required=True)
    parser.add_argument("--result-manifest", required=True)
    parser.add_argument("--metric-path", required=True)
    parser.add_argument("--output-path", required=True)
    return parser


def run(args: argparse.Namespace) -> int:
    rows = read_jsonl(os.path.abspath(args.rows_path))
    manifest_rows = read_jsonl(os.path.abspath(args.result_manifest))
    manifest_by_candidate = {
        row["candidate_id"]: row
        for row in manifest_rows
    }

    labeled_rows = []
    for row in rows:
        manifest_row = manifest_by_candidate.get(row["candidate_id"])
        if manifest_row is None:
            labeled_rows.append(row)
            continue
        labeled_rows.append(
            attach_measured_gain(
                row,
                baseline_result_path=manifest_row["baseline_result_path"],
                candidate_result_path=manifest_row["candidate_result_path"],
                metric_path=args.metric_path,
            )
        )

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_jsonl(output_path, labeled_rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
