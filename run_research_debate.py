"""CLI helper to assemble or validate debate bundles."""

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

from clip_dinoiser.research_harness.debate import load_debate_bundle, validate_debate_bundle
from clip_dinoiser.research_harness.controller import load_controller_policy
from clip_dinoiser.research_harness.registry import resolve_repo_path, write_json
from clip_dinoiser.research_harness.runtime import utc_now_iso


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble or validate debate bundles.")
    parser.add_argument("--bundle-path", required=True, help="Path to debate bundle JSON.")
    parser.add_argument("--repo-root", default=".", help="Repo root for resolving artifact paths.")
    parser.add_argument("--controller-policy", default="", help="Optional controller policy to inherit min_debate_rounds from.")
    parser.add_argument("--min-rounds", type=int, default=0, help="Explicit minimum debate rounds.")
    parser.add_argument("--validate-only", action="store_true", help="Validate an existing bundle without writing.")
    parser.add_argument("--decision", default="", help="Debate decision (approve/reject/revise).")
    parser.add_argument("--round-count", type=int, default=0, help="Number of debate rounds.")
    parser.add_argument("--reviewer-count", type=int, default=0, help="Number of reviewers.")
    parser.add_argument("--design-card", default="", help="Design card path.")
    parser.add_argument("--arbiter-decision", default="", help="Arbiter decision path.")
    parser.add_argument(
        "--review-card",
        action="append",
        default=[],
        help="Review card path (can be provided multiple times).",
    )
    return parser


def _assemble_bundle(args: argparse.Namespace, repo_root: Path) -> dict:
    bundle = {
        "decision": args.decision or "pending",
        "round_count": int(args.round_count),
        "reviewer_count": int(args.reviewer_count),
        "artifact_paths": {
            "design_card": args.design_card or "",
            "review_cards": [item for item in args.review_card if item],
            "arbiter_decision": args.arbiter_decision or "",
        },
        "updated_at_utc": utc_now_iso(),
    }
    return bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    bundle_path = resolve_repo_path(repo_root, args.bundle_path)
    min_rounds = max(int(args.min_rounds), 1)
    if args.controller_policy:
        policy = load_controller_policy(resolve_repo_path(repo_root, args.controller_policy))
        min_rounds = max(min_rounds, int(policy.get("min_debate_rounds", 1)))

    if args.validate_only:
        bundle = load_debate_bundle(bundle_path)
    else:
        bundle = _assemble_bundle(args, repo_root)
        write_json(bundle_path, bundle)

    ok, reason = validate_debate_bundle(
        bundle,
        repo_root=repo_root,
        min_rounds=max(min_rounds, int(args.round_count), 1),
        require_artifacts=False,
    )
    if not ok:
        print(f"debate_invalid: {reason}")
        return 2
    print("debate_valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
