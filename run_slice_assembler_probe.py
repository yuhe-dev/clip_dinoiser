from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

if __package__ in {None, ""}:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from slice_discovery.assembler import ProcessedFeatureAssembler, _load_json, _load_records
else:
    from .slice_discovery.runtime_compat import ensure_numpy_pickle_compat

    ensure_numpy_pickle_compat()
    from .slice_discovery.assembler import ProcessedFeatureAssembler, _load_json, _load_records


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe assembler substages in isolated subprocesses.")
    parser.add_argument("--data-root", default="./data/data_feature")
    parser.add_argument("--schema-path", default="./docs/feature_schema/unified_processed_feature_schema.json")
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--internal-stage",
        choices=[
            "load_quality",
            "load_difficulty",
            "load_coverage",
            "validate_alignment",
            "build_quality_blocks",
            "build_difficulty_blocks",
            "build_coverage_blocks",
            "flat_view",
            "save",
        ],
        default=None,
    )
    return parser


def _bundle_path(data_root: str, dimension: str) -> str:
    return os.path.join(os.path.abspath(data_root), dimension, f"{dimension}_processed_features.npy")


def _load_all(data_root: str, schema_path: str):
    schema = _load_json(os.path.abspath(schema_path))
    quality_records = _load_records(_bundle_path(data_root, "quality"))
    difficulty_records = _load_records(_bundle_path(data_root, "difficulty"))
    coverage_records = _load_records(_bundle_path(data_root, "coverage"))
    return schema, quality_records, difficulty_records, coverage_records


def _run_internal_stage(data_root: str, schema_path: str, stage: str) -> int:
    if stage == "load_quality":
        records = _load_records(_bundle_path(data_root, "quality"))
        print(json.dumps({"stage": stage, "status": "ok", "record_count": len(records)}))
        return 0
    if stage == "load_difficulty":
        records = _load_records(_bundle_path(data_root, "difficulty"))
        print(json.dumps({"stage": stage, "status": "ok", "record_count": len(records)}))
        return 0
    if stage == "load_coverage":
        records = _load_records(_bundle_path(data_root, "coverage"))
        print(json.dumps({"stage": stage, "status": "ok", "record_count": len(records)}))
        return 0

    schema, quality_records, difficulty_records, coverage_records = _load_all(data_root, schema_path)
    records_by_dimension = {
        "quality": quality_records,
        "difficulty": difficulty_records,
        "coverage": coverage_records,
    }

    if stage == "validate_alignment":
        ProcessedFeatureAssembler._validate_alignment(records_by_dimension)
        print(json.dumps({"stage": stage, "status": "ok"}))
        return 0

    if stage.startswith("build_") and stage.endswith("_blocks"):
        dimension = stage[len("build_") : -len("_blocks")]
        dimension_schema = dict(schema["dimensions"][dimension])
        rows = 0
        widths: dict[str, int] = {}
        for feature_name, feature_spec in dimension_schema["features"].items():
            field_names = list(feature_spec["model_input_fields"])
            feature_rows = [
                ProcessedFeatureAssembler._extract_feature_row(dict(record), str(feature_name), field_names)
                for record in records_by_dimension[dimension]
            ]
            rows = len(feature_rows)
            widths[str(feature_name)] = int(feature_rows[0].shape[0]) if feature_rows else 0
        print(json.dumps({"stage": stage, "status": "ok", "row_count": rows, "feature_widths": widths}))
        return 0

    assembler = ProcessedFeatureAssembler.from_processed_records(
        quality_records=quality_records,
        difficulty_records=difficulty_records,
        coverage_records=coverage_records,
        schema=schema,
    )

    if stage == "flat_view":
        flat = assembler.get_flat_view()
        print(json.dumps({"stage": stage, "status": "ok", "flat_shape": list(flat.shape)}))
        return 0

    if stage == "save":
        with tempfile.TemporaryDirectory() as tmpdir:
            assembler.save(tmpdir)
            print(json.dumps({"stage": stage, "status": "ok", "saved_files": sorted(os.listdir(tmpdir))}))
        return 0

    raise ValueError(f"Unsupported stage: {stage}")


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
    payload: dict[str, object] = {
        "returncode": int(result.returncode),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if result.returncode == 0:
        payload["status"] = "ok"
        if result.stdout.strip():
            try:
                payload["details"] = json.loads(result.stdout.strip().splitlines()[-1])
            except json.JSONDecodeError:
                pass
    else:
        payload["status"] = "crashed" if result.returncode < 0 else "failed"
        if result.returncode < 0:
            payload["signal"] = int(-result.returncode)
    return payload


def run(args: argparse.Namespace) -> int:
    data_root = os.path.abspath(args.data_root)
    schema_path = os.path.abspath(args.schema_path)

    if args.internal_stage is not None:
        return _run_internal_stage(data_root, schema_path, args.internal_stage)

    script_path = os.path.abspath(__file__)
    stages: dict[str, dict[str, object]] = {}
    stage_order = [
        "load_quality",
        "load_difficulty",
        "load_coverage",
        "validate_alignment",
        "build_quality_blocks",
        "build_difficulty_blocks",
        "build_coverage_blocks",
        "flat_view",
        "save",
    ]
    for stage in stage_order:
        stages[stage] = _probe_stage(script_path, data_root, schema_path, stage)
        print(f"{stage}: {stages[stage]['status']} (returncode={stages[stage]['returncode']})")
        sys.stdout.flush()

    with open(os.path.abspath(args.output_path), "w", encoding="utf-8") as f:
        json.dump({"data_root": data_root, "schema_path": schema_path, "stages": stages}, f, indent=2, ensure_ascii=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
