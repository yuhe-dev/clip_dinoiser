from __future__ import annotations

import argparse
import json
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge surrogate experiment results back into random-subset dataset rows.")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-path", required=True)
    return parser


def read_jsonl(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(dict(json.loads(line)))
    return rows


def write_jsonl(path: str, rows: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_label_metrics(result: dict[str, object]) -> dict[str, object]:
    task_payload = dict((result.get("coco_stuff") or {}))
    proxy_summary = task_payload.get("proxy_summary")
    full_summary = task_payload.get("full_summary")
    summary = task_payload.get("summary")
    return {
        "validation_mode": result.get("validation_mode"),
        "proxy_summary": proxy_summary,
        "full_summary": full_summary,
        "summary": summary,
        "timing": result.get("timing"),
    }


def run(args: argparse.Namespace) -> int:
    rows = read_jsonl(os.path.abspath(args.dataset_jsonl))
    runs_root = os.path.abspath(args.runs_root)
    merged: list[dict[str, object]] = []
    for row in rows:
        experiment_id = str(row["experiment_id"])
        result_path = os.path.join(runs_root, experiment_id, "result.json")
        merged_row = dict(row)
        merged_row["result_path"] = result_path if os.path.exists(result_path) else None
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            merged_row["label_metrics"] = _extract_label_metrics(result)
            merged_row["label_ready"] = True
        else:
            merged_row["label_metrics"] = None
            merged_row["label_ready"] = False
        merged.append(merged_row)

    write_jsonl(os.path.abspath(args.output_path), merged)
    print(f"merged surrogate rows={len(merged)} output={os.path.abspath(args.output_path)}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
