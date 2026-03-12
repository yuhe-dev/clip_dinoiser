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
    from slice_discovery.projector import SliceFeatureProjector
else:
    from .slice_discovery.assembler import ProcessedFeatureAssembler
    from .slice_discovery.projector import SliceFeatureProjector


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project assembled slice features and persist projector-stage debug artifacts.")
    parser.add_argument("--assembled-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scalar-scaler", choices=["zscore", "none"], default="zscore")
    parser.add_argument("--block-weighting", choices=["equal_by_block", "none"], default="none")
    return parser


def run(args: argparse.Namespace) -> int:
    assembled_dir = os.path.abspath(args.assembled_dir)
    output_dir = os.path.abspath(args.output_dir)

    projector = SliceFeatureProjector(
        scalar_scaler=args.scalar_scaler,
        block_weighting=args.block_weighting,
    )
    assembled = ProcessedFeatureAssembler.load(assembled_dir)
    projected = projector.fit_transform(assembled)
    projector.save(projected, output_dir)

    debug_summary = projector.get_debug_summary(projected)
    with open(os.path.join(output_dir, "projector_debug.json"), "w", encoding="utf-8") as f:
        json.dump(debug_summary, f, indent=2, ensure_ascii=False)

    print(f"projected_shape={tuple(debug_summary['matrix_shape'])}")
    print(f"all_finite={debug_summary['all_finite']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
