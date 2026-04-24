"""Persistence helpers for agentic research artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from .registry import write_json


def agentic_dir(output_dir: str | Path) -> Path:
    target = Path(output_dir) / "agentic"
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_agentic_artifact(output_dir: str | Path, name: str, payload: Dict[str, Any]) -> Path:
    target = agentic_dir(output_dir) / f"{name}.json"
    write_json(target, payload)
    return target


def write_agentic_jsonl(output_dir: str | Path, name: str, rows: Iterable[Dict[str, Any]]) -> Path:
    target = agentic_dir(output_dir) / f"{name}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            import json

            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def write_agentic_markdown(output_dir: str | Path, name: str, text: str) -> Path:
    target = agentic_dir(output_dir) / f"{name}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def summarize_key_value_pairs(payload: Dict[str, Any], keys: List[str]) -> str:
    lines = []
    for key in keys:
        value = payload.get(key)
        if value not in (None, "", [], {}):
            lines.append(f"- `{key}`: {value}")
    return "\n".join(lines).rstrip() + ("\n" if lines else "")
