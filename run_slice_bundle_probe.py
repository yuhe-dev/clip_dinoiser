from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

if __package__ in {None, ""}:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from slice_discovery.assembler import _load_json, _load_records
else:
    from .slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from .slice_discovery.assembler import _load_json, _load_records


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe slice bundle inputs stage-by-stage in isolated subprocesses.")
    parser.add_argument("--data-root", default="./data/data_feature")
    parser.add_argument("--schema-path", default="./docs/feature_schema/unified_processed_feature_schema.json")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--internal-stage", choices=["schema", "quality", "difficulty", "coverage"], default=None)
    return parser


def _stage_target(data_root: str, schema_path: str, stage: str) -> str:
    if stage == "schema":
        return os.path.abspath(schema_path)
    return os.path.join(os.path.abspath(data_root), stage, f"{stage}_processed_features.npy")


def _run_internal_stage(data_root: str, schema_path: str, stage: str) -> int:
    target = _stage_target(data_root, schema_path, stage)
    if stage == "schema":
        payload = _load_json(target)
        print(json.dumps({"stage": stage, "status": "ok", "schema_version": payload.get("schema_version", "")}))
        return 0

    records = _load_records(target)
    first_image = str(records[0]["image_rel"]) if records else ""
    print(json.dumps({"stage": stage, "status": "ok", "record_count": len(records), "first_image_rel": first_image}))
    return 0


def _probe_stage(script_path: str, data_root: str, schema_path: str, stage: str) -> dict[str, object]:
    command = [
        sys.executable,
        script_path,
        "--data-root",
        data_root,
        "--schema-path",
        schema_path,
        "--output-path",
        os.devnull,
        "--internal-stage",
        stage,
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    stage_result: dict[str, object] = {
        "returncode": int(result.returncode),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if result.returncode == 0:
        stage_result["status"] = "ok"
        if result.stdout.strip():
            try:
                stage_result["details"] = json.loads(result.stdout.strip().splitlines()[-1])
            except json.JSONDecodeError:
                pass
    else:
        stage_result["status"] = "crashed" if result.returncode < 0 else "failed"
        if result.returncode < 0:
            stage_result["signal"] = int(-result.returncode)
    return stage_result


def run(args: argparse.Namespace) -> int:
    data_root = os.path.abspath(args.data_root)
    schema_path = os.path.abspath(args.schema_path)
    output_path = os.path.abspath(args.output_path)

    if args.internal_stage is not None:
        return _run_internal_stage(data_root, schema_path, args.internal_stage)

    script_path = os.path.abspath(__file__)
    stages = {}
    for stage in ["schema", "quality", "difficulty", "coverage"]:
        stages[stage] = _probe_stage(script_path, data_root, schema_path, stage)
        print(f"{stage}: {stages[stage]['status']} (returncode={stages[stage]['returncode']})")
        sys.stdout.flush()

    payload = {
        "data_root": data_root,
        "schema_path": schema_path,
        "stages": stages,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
