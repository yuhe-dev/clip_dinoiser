"""Generate phase-locked experiment proposals from runtime index and policy."""

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

from clip_dinoiser.research_harness.controller import discover_repo_root
from clip_dinoiser.research_harness.proposer import (
    build_proposal_index,
    build_proposals,
    load_proposal_policy,
    materialize_proposals,
    write_proposal_index,
)
from clip_dinoiser.research_harness.registry import load_json, resolve_repo_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate phase-locked research proposals.")
    parser.add_argument("--scan-dir", default=".slicetune/experiments")
    parser.add_argument("--runtime-index-path", default=".slicetune/state/runtime_index.json")
    parser.add_argument("--task-board-path", default=".slicetune/state/task_board.json")
    parser.add_argument("--proposal-policy", default=".slicetune/runtime/proposal_policy.json")
    parser.add_argument("--proposal-index-path", default=".slicetune/state/proposal_index.json")
    parser.add_argument("--proposals-dir", default=".slicetune/proposals")
    parser.add_argument("--materialize", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = discover_repo_root(Path(args.scan_dir))
    scan_dir = resolve_repo_path(repo_root, args.scan_dir)
    runtime_index_path = resolve_repo_path(repo_root, args.runtime_index_path)
    task_board_path = resolve_repo_path(repo_root, args.task_board_path)
    proposal_policy_path = resolve_repo_path(repo_root, args.proposal_policy)
    proposal_index_path = resolve_repo_path(repo_root, args.proposal_index_path)
    proposals_dir = resolve_repo_path(repo_root, args.proposals_dir)

    runtime_index = load_json(runtime_index_path)
    task_board = load_json(task_board_path) if task_board_path.exists() else {"entries": []}
    proposal_policy = load_proposal_policy(proposal_policy_path)
    proposals = build_proposals(
        runtime_index=runtime_index,
        proposal_policy=proposal_policy,
        scan_dir=scan_dir,
        task_board=task_board,
    )
    if args.materialize:
        proposals = materialize_proposals(
            proposals,
            proposal_policy=proposal_policy,
            scan_dir=scan_dir,
            proposals_dir=proposals_dir,
        )
    payload = build_proposal_index(proposals)
    write_proposal_index(proposal_index_path, payload)
    print(f"proposal_index_written: {proposal_index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
