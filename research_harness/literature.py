"""Literature radar query planning, retrieval, ranking, and summarization."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from .agentic_registry import write_agentic_artifact, write_agentic_jsonl, write_agentic_markdown
from .contracts import ExperimentCard
from .registry import resolve_repo_path
from .runtime import utc_now_iso


VENUE_PRIORITY = [
    "NeurIPS",
    "ICML",
    "ICLR",
    "CVPR",
    "ICCV",
    "ECCV",
    "TPAMI",
    "CHI",
    "UIST",
    "VIS",
    "ACL",
    "EMNLP",
    "NAACL",
]


def _tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", str(text).lower()) if len(token) >= 3]


def _keyword_families(card: ExperimentCard) -> Dict[str, List[str]]:
    stage_terms = {
        "noise_floor": ["training noise", "evaluation variance", "reproducibility", "segmentation"],
        "same_subset_multi_seed": ["training noise", "seed variance", "data composition", "semantic segmentation"],
        "learner_sensitivity_ladder": ["data selection", "learner sensitivity", "dataset pruning", "segmentation"],
        "feature_intervention_matrix": ["feature intervention", "data valuation", "example reweighting", "subpopulation shift"],
        "literature_radar": ["data selection", "slice discovery", "distribution optimization", "segmentation"],
    }
    return {
        "core": stage_terms.get(card.loop_kind, ["data selection", "segmentation", "training data"]),
        "modality": ["image segmentation", "semantic segmentation", "vision"],
        "phase": [card.phase.lower().replace(" ", "_"), "feature signal audit"],
    }


def build_literature_query_plan(
    card: ExperimentCard,
    *,
    context_packet: Dict[str, Any] | None = None,
    analysis_brief: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context_packet = context_packet or {}
    analysis_brief = analysis_brief or {}
    families = _keyword_families(card)
    trigger_reasons = list(card.metadata.get("trigger_reasons", []))
    if not trigger_reasons:
        trigger_reasons = list(analysis_brief.get("branch_recommendation", {}).get("recommended_actions", []))[:2]
    queries = [
        " ".join(families["core"][:2] + families["modality"][:1]),
        " ".join(families["core"][:3]),
        " ".join(["semantic segmentation", "data selection", "reweighting"]),
        " ".join(["subpopulation", "slice discovery", "vision model validation"]),
    ]
    # Deduplicate while preserving order.
    deduped_queries: List[str] = []
    seen = set()
    for query in queries:
        query = " ".join(query.split())
        if query and query not in seen:
            seen.add(query)
            deduped_queries.append(query)
    return {
        "generated_by": "literature_radar_v1",
        "generated_at_utc": utc_now_iso(),
        "experiment_id": card.experiment_id,
        "phase": card.phase,
        "loop_kind": card.loop_kind,
        "source_experiment_id": str(card.metadata.get("source_experiment_id", "")),
        "failure_mode": trigger_reasons,
        "queries": deduped_queries,
        "venue_priority": VENUE_PRIORITY,
        "ranking_axes": [
            "stage relevance",
            "modality match",
            "venue quality",
            "recency",
            "reproduction feasibility",
        ],
        "source_paths": {
            "context_packet": context_packet.get("source_path", ""),
        },
    }


def _http_json(url: str, *, user_agent: str = "SliceTuneResearchHarness/1.0", timeout: int = 20) -> Dict[str, Any]:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def search_openalex(query: str, *, per_page: int = 8, timeout: int = 20) -> List[Dict[str, Any]]:
    url = (
        "https://api.openalex.org/works?search="
        + quote_plus(query)
        + f"&per-page={int(per_page)}&sort=relevance_score:desc"
    )
    payload = _http_json(url, timeout=timeout)
    return list(payload.get("results", []))


def _abstract_text(payload: Dict[str, Any]) -> str:
    inverse = payload.get("abstract_inverted_index") or {}
    if not inverse:
        return ""
    pairs: List[tuple[int, str]] = []
    for word, positions in inverse.items():
        for pos in positions:
            pairs.append((int(pos), str(word)))
    return " ".join(word for _idx, word in sorted(pairs))


def _venue_score(venue_name: str) -> float:
    for index, venue in enumerate(VENUE_PRIORITY):
        if venue.lower() in venue_name.lower():
            return max(0.0, 1.0 - 0.05 * index)
    return 0.2 if venue_name else 0.0


def rank_literature_results(
    query_plan: Dict[str, Any],
    *,
    results_by_query: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    query_tokens = set()
    for query in query_plan.get("queries", []):
        query_tokens.update(_tokenize(query))

    merged: Dict[str, Dict[str, Any]] = {}
    for query, items in results_by_query.items():
        for item in items:
            work_id = str(item.get("id", item.get("doi", item.get("display_name", ""))))
            title = str(item.get("display_name", ""))
            abstract = _abstract_text(item)
            venue = str(((item.get("primary_location") or {}).get("source") or {}).get("display_name", ""))
            year = int(item.get("publication_year", 0) or 0)
            tokens = set(_tokenize(title + " " + abstract))
            overlap = len(query_tokens & tokens)
            relevance_score = float(item.get("relevance_score", 0.0) or 0.0)
            venue_score = _venue_score(venue)
            year_score = 0.0
            if year >= 2024:
                year_score = 0.3
            elif year >= 2021:
                year_score = 0.2
            elif year >= 2018:
                year_score = 0.1
            score = overlap + venue_score + year_score + relevance_score * 0.05
            existing = merged.get(work_id)
            if existing and existing["score"] >= score:
                existing["matched_queries"].append(query)
                continue
            merged[work_id] = {
                "work_id": work_id,
                "title": title,
                "abstract": abstract[:1200],
                "venue": venue,
                "year": year,
                "openalex_id": str(item.get("id", "")),
                "landing_page_url": str((item.get("primary_location") or {}).get("landing_page_url", "")),
                "pdf_url": str((item.get("primary_location") or {}).get("pdf_url", "")),
                "score": score,
                "matched_queries": [query],
                "query_overlap": overlap,
                "relevance_score": relevance_score,
                "stage_relevance": "high" if overlap >= 4 else "medium" if overlap >= 2 else "low",
                "reproduction_feasibility": (
                    "high"
                    if str((item.get("primary_location") or {}).get("pdf_url", "")) or str((item.get("primary_location") or {}).get("landing_page_url", ""))
                    else "medium"
                ),
                "recommendation": "reproduce" if score >= 4.0 else "read" if score >= 2.5 else "park",
            }
    return sorted(merged.values(), key=lambda item: (-float(item["score"]), -int(item["year"] or 0), item["title"]))


def summarize_literature_results(
    query_plan: Dict[str, Any],
    ranked_results: List[Dict[str, Any]],
    *,
    search_errors: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    search_errors = dict(search_errors or {})
    recommendation_counts = {"reproduce": 0, "read": 0, "park": 0}
    top_titles: List[str] = []
    unique_venues = set()
    for item in ranked_results:
        recommendation = str(item.get("recommendation", "park"))
        recommendation_counts[recommendation] = recommendation_counts.get(recommendation, 0) + 1
        if len(top_titles) < 5:
            top_titles.append(str(item.get("title", "")))
        venue = str(item.get("venue", ""))
        if venue:
            unique_venues.add(venue)
    query_count = len(query_plan.get("queries", []))
    return {
        "query_count": query_count,
        "ranked_result_count": len(ranked_results),
        "reproduce_count": recommendation_counts.get("reproduce", 0),
        "read_count": recommendation_counts.get("read", 0),
        "park_count": recommendation_counts.get("park", 0),
        "search_error_count": len(search_errors),
        "search_error_ratio": (len(search_errors) / query_count) if query_count else 0.0,
        "top_score": float(ranked_results[0]["score"]) if ranked_results else 0.0,
        "top_titles": top_titles,
        "unique_venue_count": len(unique_venues),
        "query_plan_queries": list(query_plan.get("queries", [])),
    }


def render_literature_markdown(query_plan: Dict[str, Any], ranked_results: List[Dict[str, Any]]) -> str:
    lines = [
        "# Literature Radar",
        "",
        f"- `experiment_id`: {query_plan.get('experiment_id', '')}",
        f"- `source_experiment_id`: {query_plan.get('source_experiment_id', '')}",
        "",
        "## Queries",
        "",
    ]
    for query in query_plan.get("queries", []):
        lines.append(f"- {query}")
    lines.extend(["", "## Top Methods", ""])
    for item in ranked_results[:10]:
        lines.append(
            f"- [{item.get('recommendation')}] {item.get('title')} ({item.get('year')}, {item.get('venue')}) score={item.get('score'):.2f}"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_literature_artifacts(
    card: ExperimentCard,
    *,
    repo_root: str | Path,
    context_packet: Dict[str, Any] | None = None,
    analysis_brief: Dict[str, Any] | None = None,
    execute_search: bool = False,
) -> Dict[str, str]:
    output_dir = resolve_repo_path(repo_root, card.output_dir)
    query_plan = build_literature_query_plan(card, context_packet=context_packet, analysis_brief=analysis_brief)
    query_plan_path = write_agentic_artifact(output_dir, "literature_query_plan", query_plan)
    result_paths = {"literature_query_plan_path": str(query_plan_path.resolve())}
    if not execute_search:
        return result_paths

    results_by_query: Dict[str, List[Dict[str, Any]]] = {}
    search_errors: Dict[str, str] = {}
    for query in query_plan.get("queries", []):
        try:
            results_by_query[query] = search_openalex(query)
        except Exception as exc:
            results_by_query[query] = []
            search_errors[query] = str(exc)
    ranked = rank_literature_results(query_plan, results_by_query=results_by_query)
    summary = summarize_literature_results(query_plan, ranked, search_errors=search_errors)
    search_report = {
        "generated_by": "literature_radar_v1",
        "generated_at_utc": utc_now_iso(),
        "experiment_id": card.experiment_id,
        "query_plan": query_plan,
        "summary": summary,
        "search_errors": search_errors,
        "result_count": len(ranked),
        "top_titles": [item.get("title", "") for item in ranked[:10]],
    }
    report_path = write_agentic_artifact(output_dir, "literature_search_report", search_report)
    methods_path = write_agentic_jsonl(output_dir, "method_cards", ranked)
    markdown_path = write_agentic_markdown(output_dir, "literature_radar", render_literature_markdown(query_plan, ranked))
    result_paths.update(
        {
            "literature_search_report_path": str(report_path.resolve()),
            "method_cards_path": str(methods_path.resolve()),
            "literature_markdown_path": str(markdown_path.resolve()),
        }
    )
    return result_paths
