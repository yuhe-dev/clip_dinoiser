from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in {None, ""}:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from slice_discovery.assembler import ProcessedFeatureAssembler
else:
    from .slice_discovery.assembler import ProcessedFeatureAssembler


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble processed slice features and persist assembler-stage debug artifacts.")
    parser.add_argument("--data-root", default="./data/data_feature")
    parser.add_argument("--schema-path", default="./docs/feature_schema/unified_processed_feature_schema.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-samples", type=int, default=None)
    return parser


def run(args: argparse.Namespace) -> int:
    data_root = os.path.abspath(args.data_root)
    schema_path = os.path.abspath(args.schema_path)
    output_dir = os.path.abspath(args.output_dir)

    assembler = ProcessedFeatureAssembler.from_processed_paths(
        quality_path=os.path.join(data_root, "quality", "quality_processed_features.npy"),
        difficulty_path=os.path.join(data_root, "difficulty", "difficulty_processed_features.npy"),
        coverage_path=os.path.join(data_root, "coverage", "coverage_processed_features.npy"),
        schema_path=schema_path,
        limit_samples=args.limit_samples,
    )
    assembler.save(output_dir)

    debug_summary = assembler.get_debug_summary()
    with open(os.path.join(output_dir, "assembler_debug.json"), "w", encoding="utf-8") as f:
        json.dump(debug_summary, f, indent=2, ensure_ascii=False)

    print(f"sample_count={assembler.sample_count}")
    print(f"flat_shape={tuple(debug_summary['flat_shape'])}")
    print(f"block_order={','.join(assembler.list_blocks())}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
