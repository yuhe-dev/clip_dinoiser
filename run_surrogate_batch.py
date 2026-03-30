from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run surrogate subset experiments in parallel across multiple GPUs.")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--config", default="feature_experiment_fast_cached_slide")
    parser.add_argument("--python-bin", default=sys.executable or "python")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--master-port-base", type=int, default=29600)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    return parser


def _progress(message: str) -> None:
    print(f"[surrogate_batch] {message}", file=sys.stderr, flush=True)


def read_jsonl(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(dict(json.loads(line)))
    return rows


def filter_rows(rows: list[dict[str, object]], *, split: str, limit: int) -> list[dict[str, object]]:
    selected = rows
    if split != "all":
        selected = [row for row in selected if str(row.get("split")) == split]
    if limit > 0:
        selected = selected[: int(limit)]
    return selected


def run(args: argparse.Namespace, log_fn=_progress) -> int:
    rows = filter_rows(
        read_jsonl(os.path.abspath(args.dataset_jsonl)),
        split=str(args.split),
        limit=int(args.limit),
    )
    gpu_ids = [token.strip() for token in str(args.gpus).split(",") if token.strip()]
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    os.makedirs(args.runs_root, exist_ok=True)

    pending = [row for row in rows if row.get("manifest_path")]
    active: list[dict[str, object]] = []
    cursor = 0
    finished = 0
    while cursor < len(pending) or active:
        while cursor < len(pending) and len(active) < len(gpu_ids):
            row = pending[cursor]
            slot = len(active)
            gpu_id = gpu_ids[slot]
            port = int(args.master_port_base) + slot
            experiment_id = str(row["experiment_id"])
            out_dir = os.path.join(os.path.abspath(args.runs_root), experiment_id)
            os.makedirs(out_dir, exist_ok=True)
            result_path = os.path.join(out_dir, "result.json")
            log_path = os.path.join(out_dir, "stdout.log")
            if os.path.exists(result_path):
                log_fn(f"skip experiment_id={experiment_id} reason=result_exists")
                finished += 1
                cursor += 1
                continue

            command = [
                str(args.python_bin),
                "-m",
                "torch.distributed.run",
                "--nproc_per_node=1",
                "--master_port",
                str(port),
                "run_remix_training_experiment.py",
                "--config",
                str(args.config),
                "--subset-manifest",
                str(row["manifest_path"]),
                "--output-dir",
                out_dir,
                "--result-name",
                "result.json",
                "--seed",
                str(int(row.get("training_seed", 0))),
            ]
            log_fn(f"launch gpu={gpu_id} port={port} experiment_id={experiment_id}")
            with open(log_path, "w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=os.getcwd(),
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            active.append(
                {
                    "process": process,
                    "gpu_id": gpu_id,
                    "experiment_id": experiment_id,
                }
            )
            cursor += 1

        time.sleep(2.0)
        next_active: list[dict[str, object]] = []
        for item in active:
            process = item["process"]
            code = process.poll()
            if code is None:
                next_active.append(item)
                continue
            finished += 1
            experiment_id = str(item["experiment_id"])
            if int(code) == 0:
                log_fn(f"done experiment_id={experiment_id}")
            else:
                log_fn(f"failed experiment_id={experiment_id} exit_code={code}")
        active = next_active

    log_fn(f"all jobs finished count={finished}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
