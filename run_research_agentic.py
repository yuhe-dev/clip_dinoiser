"""Generate agentic research artifacts for experiment cards."""

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

from clip_dinoiser.research_harness.agentic import ensure_agentic_artifacts
from clip_dinoiser.research_harness.controller import discover_repo_root
from clip_dinoiser.research_harness.registry import load_experiment_card, resolve_repo_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate agentic planning/analysis/literature artifacts.")
    parser.add_argument("--experiment-card", required=True)
    parser.add_argument("--execute-literature-search", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    card_path = Path(args.experiment_card)
    repo_root = discover_repo_root(card_path)
    resolved_card_path = resolve_repo_path(repo_root, str(card_path))
    card = load_experiment_card(resolved_card_path)
    paths = ensure_agentic_artifacts(
        repo_root=repo_root,
        card=card,
        card_path=resolved_card_path,
        execute_literature_search=bool(args.execute_literature_search),
    )
    print(f"agentic_artifacts_written: {len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
