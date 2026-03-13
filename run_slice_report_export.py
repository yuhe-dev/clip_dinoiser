from __future__ import annotations

import argparse
import os
import sys

if __package__ in {None, ""}:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from slice_discovery.report_exporter import SliceReportExporter
else:
    from .slice_discovery.report_exporter import SliceReportExporter


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a static slice report bundle for frontend consumption.")
    parser.add_argument("--projected-dir", required=True)
    parser.add_argument("--cluster-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-root")
    return parser


def run(args: argparse.Namespace) -> int:
    exporter = SliceReportExporter()
    exporter.export(
        projected_dir=os.path.abspath(args.projected_dir),
        cluster_dir=os.path.abspath(args.cluster_dir),
        output_dir=os.path.abspath(args.output_dir),
        image_root=os.path.abspath(args.image_root) if args.image_root else None,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
