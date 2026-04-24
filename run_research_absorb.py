"""Materialize the current runtime index from experiment cards and artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    ROOT = os.path.abspath(os.path.dirname(__file__))
    PARENT = os.path.dirname(ROOT)
    if PARENT not in sys.path:
        sys.path.insert(0, PARENT)

from clip_dinoiser.research_harness.absorber import build_runtime_index, write_runtime_index
from clip_dinoiser.research_harness.controller import discover_repo_root
from clip_dinoiser.research_harness.registry import resolve_repo_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate runtime index from experiment cards and artifacts.")
    parser.add_argument("--scan-dir", default=".slicetune/experiments")
    parser.add_argument("--output-path", default=".slicetune/state/runtime_index.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = discover_repo_root(Path(args.scan_dir))
    scan_dir = resolve_repo_path(repo_root, args.scan_dir)
    output_path = resolve_repo_path(repo_root, args.output_path)
    payload = build_runtime_index(repo_root, scan_dir)
    write_runtime_index(output_path, payload)
    print(f"runtime_index_written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
