"""Debate bundle validation and auto-generation for research gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from .contracts import ExperimentCard
from .registry import load_json, resolve_repo_path, write_json
from .runtime import utc_now_iso


def load_debate_bundle(path: str | Path) -> Dict[str, Any]:
    return load_json(path)


def _artifact_paths(bundle: Dict[str, Any]) -> list[str]:
    artifacts = dict(bundle.get("artifact_paths", {}))
    if not artifacts:
        artifacts = {
            "design_card": bundle.get("design_card_path", ""),
            "review_cards": list(bundle.get("review_card_paths", [])),
            "arbiter_decision": bundle.get("decision_card_path", ""),
        }
    paths: list[str] = []
    for key in ("design_card", "arbiter_decision"):
        value = artifacts.get(key)
        if value:
            paths.append(str(value))
    review_cards = artifacts.get("review_cards", [])
    if isinstance(review_cards, list):
        paths.extend([str(item) for item in review_cards if item])
    return paths


def normalize_debate_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = dict(bundle.get("artifact_paths", {}))
    if not artifacts:
        artifacts = {
            "design_card": bundle.get("design_card_path", ""),
            "review_cards": list(bundle.get("review_card_paths", [])),
            "arbiter_decision": bundle.get("decision_card_path", ""),
        }
    normalized = {
        "debate_id": str(bundle.get("debate_id", bundle.get("experiment_id", ""))),
        "decision": str(bundle.get("decision", "pending")),
        "round_count": int(bundle.get("round_count", 0)),
        "reviewer_count": int(bundle.get("reviewer_count", 0)),
        "artifact_paths": {
            "design_card": str(artifacts.get("design_card", "")),
            "review_cards": [str(item) for item in artifacts.get("review_cards", []) if item],
            "arbiter_decision": str(artifacts.get("arbiter_decision", "")),
        },
        "updated_at_utc": str(bundle.get("updated_at_utc", utc_now_iso())),
    }
    if "created_by" in bundle:
        normalized["created_by"] = str(bundle.get("created_by", ""))
    if "created_at_utc" in bundle:
        normalized["created_at_utc"] = str(bundle.get("created_at_utc", ""))
    # Keep legacy aliases for compatibility with existing artifacts and docs.
    normalized["design_card_path"] = normalized["artifact_paths"]["design_card"]
    normalized["review_card_paths"] = list(normalized["artifact_paths"]["review_cards"])
    normalized["decision_card_path"] = normalized["artifact_paths"]["arbiter_decision"]
    return normalized


def _debate_base_path(repo_root: str | Path, card: ExperimentCard) -> Path:
    bundle_path = resolve_repo_path(repo_root, card.debate_bundle_path)
    stem = bundle_path.name
    if stem.endswith("_bundle.json"):
        stem = stem[: -len("_bundle.json")]
    else:
        stem = bundle_path.stem
    return bundle_path.parent / stem


def _output_dir(repo_root: str | Path, card: ExperimentCard) -> Path | None:
    if not card.output_dir:
        return None
    return resolve_repo_path(repo_root, card.output_dir)


def _load_optional_json(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def _review_findings(card: ExperimentCard, *, repo_root: str | Path) -> Tuple[List[str], List[str]]:
    blockers: List[str] = []
    warnings: List[str] = []
    output_dir = _output_dir(repo_root, card)
    design_pack = _load_optional_json(output_dir / "agentic" / "design_pack.json" if output_dir else None)
    rubric = _load_optional_json(output_dir / "agentic" / "evaluation_rubric.json" if output_dir else None)
    query_plan = _load_optional_json(output_dir / "agentic" / "literature_query_plan.json" if output_dir else None)
    search_report = _load_optional_json(output_dir / "agentic" / "literature_search_report.json" if output_dir else None)

    if not card.hypothesis:
        blockers.append("缺少 hypothesis，无法形成可审计研究问题")
    if not card.output_dir:
        blockers.append("缺少 output_dir，无法形成稳定 artifact 归档")
    if not card.phase:
        blockers.append("缺少 phase，无法通过阶段约束")
    if card.loop_kind in {"noise_floor", "same_subset_multi_seed"} and not card.judge_policy_path:
        blockers.append("缺少 judge_policy_path，无法形成独立 judge 规则")
    if card.loop_kind == "same_subset_multi_seed" and not card.input_path:
        blockers.append("缺少固定 subset manifest 输入")
    if design_pack:
        mutation_scope = dict(design_pack.get("mutation_scope", {}))
        if not mutation_scope.get("changed") or not mutation_scope.get("frozen"):
            blockers.append("design_pack 未明确 changed/frozen 变量，难以形成有效科研对照")
        if not design_pack.get("confound_guardrails"):
            warnings.append("design_pack 缺少 confound guardrails，建议补充混杂因素控制")
    else:
        warnings.append("缺少 design_pack，当前 debate 只能基于 card 做结构审计")
    if rubric:
        if not rubric.get("success_criteria"):
            warnings.append("evaluation_rubric 未写明 success criteria")
    elif card.loop_kind in {"noise_floor", "same_subset_multi_seed", "literature_radar"}:
        warnings.append("缺少 evaluation_rubric，judge 将缺少预注册解释上下文")
    if card.loop_kind == "literature_radar":
        if not query_plan:
            blockers.append("literature_radar 缺少 query_plan，无法审查检索空间是否合理")
        if search_report and int(search_report.get("summary", {}).get("ranked_result_count", 0)) <= 0:
            warnings.append("literature_radar 当前尚未找到可排序的方法，可能需要重构 query")
    if not card.requires_debate:
        warnings.append("当前卡片未强制 debate，但仍生成了 debate 记录")
    if bool(card.metadata.get("design_only", False)):
        warnings.append("该卡片为 design_only，debate 通过后也不会自动进入执行")
    return blockers, warnings


def _render_design_markdown(
    card: ExperimentCard,
    blockers: List[str],
    warnings: List[str],
    *,
    repo_root: str | Path,
) -> str:
    output_dir = _output_dir(repo_root, card)
    design_pack = _load_optional_json(output_dir / "agentic" / "design_pack.json" if output_dir else None)
    rubric = _load_optional_json(output_dir / "agentic" / "evaluation_rubric.json" if output_dir else None)
    lines = [
        "# Debate Design Card",
        "",
        f"- `experiment_id`: {card.experiment_id}",
        f"- `phase`: {card.phase}",
        f"- `loop_kind`: {card.loop_kind}",
        f"- `owner`: {card.owner}",
        f"- `budget_tier`: {card.budget_tier}",
        f"- `hypothesis`: {card.hypothesis or 'MISSING'}",
        f"- `input_path`: {card.input_path or 'N/A'}",
        f"- `output_dir`: {card.output_dir or 'MISSING'}",
        f"- `judge_policy_path`: {card.judge_policy_path or 'MISSING'}",
        "",
        "## Debate Focus",
        "",
        "- 该任务是否符合当前 phase。",
        "- 该任务是否有独立 judge、稳定 artifact、可审计输入。",
        "- 该任务是可执行任务还是 design_only 任务。",
    ]
    if design_pack:
        lines.extend(
            [
                "",
                "## Design Pack Snapshot",
                "",
                f"- `design_class`: {design_pack.get('design_class', '')}",
                f"- `objective`: {design_pack.get('objective', '')}",
            ]
        )
        for item in design_pack.get("execution_recipe", []):
            lines.append(f"- recipe: {item}")
    if rubric:
        lines.extend(["", "## Rubric Snapshot", ""])
        for item in rubric.get("success_criteria", [])[:4]:
            lines.append(f"- success: {item}")
    if blockers:
        lines.extend(["", "## Blocking Risks", ""])
        lines.extend([f"- {item}" for item in blockers])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {item}" for item in warnings])
    return "\n".join(lines).rstrip() + "\n"


def _render_review_markdown(
    *,
    reviewer_name: str,
    focus: str,
    blockers: List[str],
    warnings: List[str],
    focused_findings: List[str] | None = None,
) -> str:
    decision = "revise" if blockers else "approve_with_warnings" if warnings else "approve"
    lines = [
        f"# Review Card: {reviewer_name}",
        "",
        f"- `focus`: {focus}",
        f"- `review_decision`: {decision}",
        "",
        "## Findings",
        "",
    ]
    focused_findings = list(focused_findings or [])
    if not blockers and not warnings and not focused_findings:
        lines.append("- 未发现阻止执行的结构性问题。")
    for item in focused_findings:
        lines.append(f"- NOTE: {item}")
    for item in blockers:
        lines.append(f"- BLOCK: {item}")
    for item in warnings:
        lines.append(f"- WARN: {item}")
    return "\n".join(lines).rstrip() + "\n"


def _render_decision_markdown(blockers: List[str], warnings: List[str]) -> Tuple[str, str]:
    decision = "approve" if not blockers else "revise"
    rationale = "无阻断项，可进入下一 gate。" if not blockers else "存在阻断项，需要修订后重新 debate。"
    lines = [
        "# Arbiter Decision",
        "",
        f"- `decision`: {decision}",
        f"- `rationale`: {rationale}",
        f"- `blocking_count`: {len(blockers)}",
        f"- `warning_count`: {len(warnings)}",
        "",
        "## Summary",
        "",
    ]
    if blockers:
        lines.extend([f"- BLOCK: {item}" for item in blockers])
    if warnings:
        lines.extend([f"- WARN: {item}" for item in warnings])
    if not blockers and not warnings:
        lines.append("- 设计与执行契约完整。")
    return decision, "\n".join(lines).rstrip() + "\n"


def auto_generate_debate_bundle(
    *,
    card: ExperimentCard,
    repo_root: str | Path,
    force: bool = False,
) -> Path:
    bundle_path = resolve_repo_path(repo_root, card.debate_bundle_path)
    if bundle_path.exists() and not force:
        normalized = normalize_debate_bundle(load_debate_bundle(bundle_path))
        write_json(bundle_path, normalized)
        return bundle_path

    base = _debate_base_path(repo_root, card)
    blockers, warnings = _review_findings(card, repo_root=repo_root)
    output_dir = _output_dir(repo_root, card)
    design_pack = _load_optional_json(output_dir / "agentic" / "design_pack.json" if output_dir else None)
    rubric = _load_optional_json(output_dir / "agentic" / "evaluation_rubric.json" if output_dir else None)
    query_plan = _load_optional_json(output_dir / "agentic" / "literature_query_plan.json" if output_dir else None)
    method_cards_path = output_dir / "agentic" / "method_cards.jsonl" if output_dir else None
    design_path = base.parent / f"{base.name}_design.md"
    review_skeptic_path = base.parent / f"{base.name}_review_skeptic.md"
    review_benchmark_path = base.parent / f"{base.name}_review_benchmark.md"
    review_literature_path = base.parent / f"{base.name}_review_literature.md"
    review_harness_path = base.parent / f"{base.name}_review_harness.md"
    decision_path = base.parent / f"{base.name}_decision.md"

    design_path.parent.mkdir(parents=True, exist_ok=True)
    design_path.write_text(_render_design_markdown(card, blockers, warnings, repo_root=repo_root), encoding="utf-8")
    skeptic_findings: List[str] = []
    benchmark_findings: List[str] = []
    literature_findings: List[str] = []
    harness_findings: List[str] = []
    if design_pack:
        skeptic_findings.append(f"changed={design_pack.get('mutation_scope', {}).get('changed', [])}")
        skeptic_findings.append(f"frozen={design_pack.get('mutation_scope', {}).get('frozen', [])}")
    if rubric:
        benchmark_findings.extend([f"success={item}" for item in rubric.get("success_criteria", [])[:3]])
    if query_plan:
        literature_findings.append(f"query_count={len(query_plan.get('queries', []))}")
    if method_cards_path and method_cards_path.exists():
        literature_findings.append("已有 method_cards，可进入 read/reproduce 分流")
    else:
        literature_findings.append("暂无 method_cards，若这是 radar loop 则执行后需要补齐")
    harness_findings.append(f"output_dir={card.output_dir or 'MISSING'}")
    harness_findings.append(f"judge_policy_path={card.judge_policy_path or 'N/A'}")
    review_skeptic_path.write_text(
        _render_review_markdown(
            reviewer_name="Skeptic",
            focus="研究假设、混杂变量、替代解释",
            blockers=blockers,
            warnings=warnings,
            focused_findings=skeptic_findings,
        ),
        encoding="utf-8",
    )
    review_benchmark_path.write_text(
        _render_review_markdown(
            reviewer_name="Benchmark Steward",
            focus="benchmark 可比性、success criteria、phase 契约",
            blockers=blockers,
            warnings=warnings,
            focused_findings=benchmark_findings,
        ),
        encoding="utf-8",
    )
    review_literature_path.write_text(
        _render_review_markdown(
            reviewer_name="Literature Critic",
            focus="相关工作覆盖、方法空间、是否值得复现",
            blockers=[],
            warnings=warnings,
            focused_findings=literature_findings,
        ),
        encoding="utf-8",
    )
    review_harness_path.write_text(
        _render_review_markdown(
            reviewer_name="Harness Reviewer",
            focus="artifact 完整性、judge policy、runtime readiness",
            blockers=blockers,
            warnings=warnings,
            focused_findings=harness_findings,
        ),
        encoding="utf-8",
    )
    decision, decision_markdown = _render_decision_markdown(blockers, warnings)
    decision_path.write_text(decision_markdown, encoding="utf-8")

    bundle = normalize_debate_bundle(
        {
            "debate_id": card.experiment_id,
            "decision": decision,
            "round_count": 3,
            "reviewer_count": 4,
            "artifact_paths": {
                "design_card": str(design_path.relative_to(Path(repo_root).resolve())),
                "review_cards": [
                    str(review_skeptic_path.relative_to(Path(repo_root).resolve())),
                    str(review_benchmark_path.relative_to(Path(repo_root).resolve())),
                    str(review_literature_path.relative_to(Path(repo_root).resolve())),
                    str(review_harness_path.relative_to(Path(repo_root).resolve())),
                ],
                "arbiter_decision": str(decision_path.relative_to(Path(repo_root).resolve())),
            },
            "created_by": "auto_debate",
            "created_at_utc": utc_now_iso(),
            "updated_at_utc": utc_now_iso(),
        }
    )
    write_json(bundle_path, bundle)
    return bundle_path


def validate_debate_bundle(
    bundle: Dict[str, Any],
    *,
    repo_root: str | Path,
    min_rounds: int,
    require_artifacts: bool = False,
) -> Tuple[bool, str]:
    bundle = normalize_debate_bundle(bundle)
    decision = str(bundle.get("decision", "")).lower()
    round_count = int(bundle.get("round_count", 0))
    reviewer_count = int(bundle.get("reviewer_count", 0))
    if decision != "approve":
        return False, f"debate decision is '{decision}', not approve"
    if round_count < min_rounds:
        return False, f"debate rounds {round_count} are below minimum {min_rounds}"
    if reviewer_count < 1:
        return False, "debate bundle must record at least one reviewer"

    if require_artifacts:
        repo_root = Path(repo_root)
        for raw_path in _artifact_paths(bundle):
            artifact_path = (repo_root / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path)
            if not artifact_path.exists():
                return False, f"debate artifact missing: {artifact_path}"
    return True, ""
