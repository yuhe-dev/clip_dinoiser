from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in {None, ""}:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from slice_remix.analysis import build_analysis_report
    from slice_remix.dataset import read_jsonl
else:
    from .slice_remix.analysis import build_analysis_report
    from .slice_remix.dataset import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an analysis report for remix recommendations and surrogate quality.")
    parser.add_argument("--response-dataset", required=True)
    parser.add_argument("--recommendation-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--surrogate-model", choices=["linear", "quadratic"])
    parser.add_argument("--bootstrap-models", type=int)
    parser.add_argument("--kappa", type=float)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--metric-path")
    parser.add_argument("--baseline-result-path")
    parser.add_argument("--recommended-result-path")
    return parser


def _progress(message: str) -> None:
    print(f"[remix_analysis_report] {message}", file=sys.stderr, flush=True)


def run(args: argparse.Namespace, log_fn=_progress) -> int:
    response_dataset = os.path.abspath(args.response_dataset)
    recommendation_path = os.path.abspath(args.recommendation_path)
    output_path = os.path.abspath(args.output_path)

    log_fn("loading recommendation")
    with open(recommendation_path, "r", encoding="utf-8") as f:
        recommendation = json.load(f)

    log_fn("loading labeled response rows")
    response_rows = read_jsonl(response_dataset)
    context = recommendation.get("context", {})
    model_name = args.surrogate_model or context.get("surrogate_model", "linear")
    bootstrap_models = int(args.bootstrap_models) if args.bootstrap_models is not None else int(context.get("bootstrap_models", 1))
    kappa = float(args.kappa) if args.kappa is not None else float(context.get("kappa", 0.0))

    log_fn(
        f"building analysis report model={model_name} bootstrap_models={bootstrap_models} "
        f"top_k={int(args.top_k)} kappa={kappa:.3f}"
    )
    report = build_analysis_report(
        response_rows=response_rows,
        recommendation=recommendation,
        model_name=model_name,
        bootstrap_models=bootstrap_models,
        kappa=kappa,
        top_k=int(args.top_k),
        baseline_result_path=os.path.abspath(args.baseline_result_path) if args.baseline_result_path else None,
        recommended_result_path=os.path.abspath(args.recommended_result_path) if args.recommended_result_path else None,
        metric_path=args.metric_path,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log_fn(f"writing analysis report output={output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
