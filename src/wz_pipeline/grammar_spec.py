from __future__ import annotations

import re
from functools import lru_cache

from .grammar_config import load_grammar_config
from .paths import GRAMMAR_SPEC_PATH


HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$")


def _compact_lines(body: str, max_lines: int = 4) -> list[str]:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    return lines[:max_lines]


def _display_title(title: str) -> str:
    return title.lstrip("#").strip()


@lru_cache(maxsize=1)
def load_grammar_spec_sections() -> dict[str, str]:
    if not GRAMMAR_SPEC_PATH.exists():
        return {}
    text = GRAMMAR_SPEC_PATH.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    current_title = ""
    current_lines: list[str] = []
    for raw_line in text.splitlines():
        match = HEADING_RE.match(raw_line)
        if match:
            if current_title:
                sections[current_title] = "\n".join(current_lines).strip()
            current_title = raw_line.strip()
            current_lines = []
            continue
        if current_title:
            current_lines.append(raw_line)
    if current_title:
        sections[current_title] = "\n".join(current_lines).strip()
    return sections


def compact_spec_section(title: str, *, max_lines: int = 4) -> str:
    body = load_grammar_spec_sections().get(title, "").strip()
    if not body:
        return _display_title(title)
    compact = _compact_lines(body, max_lines=max_lines)
    return _display_title(title) + "\n" + "\n".join(compact)


def load_spec_section_full(title: str) -> str:
    body = load_grammar_spec_sections().get(title, "").strip()
    if not body:
        return _display_title(title)
    return _display_title(title) + "\n" + body


def generation_spec_excerpt(*, max_lines: int = 3) -> str:
    config = load_grammar_config()
    parts: list[str] = []
    for title in config.get("generation_full_section_titles") or []:
        parts.append(load_spec_section_full(title))
    for title in config.get("generation_section_titles") or []:
        parts.append(compact_spec_section(title, max_lines=max_lines))
    return "\n\n".join(parts)


def relevant_spec_titles(text: str, reasons: list[str] | None = None) -> list[str]:
    sentence = str(text or "")
    grammar_config = load_grammar_config()
    marker_to_sections = grammar_config.get("marker_to_sections") or {}
    reason_to_sections = grammar_config.get("reason_to_sections") or {}
    ordered: list[str] = []
    seen: set[str] = set()

    def add(title: str) -> None:
        if title not in seen:
            seen.add(title)
            ordered.append(title)

    add("## 2. 生成总原则")
    for marker, titles in marker_to_sections.items():
        if marker in sentence:
            for title in titles:
                add(title)
    for reason in reasons or []:
        reason_key = str(reason).split(":", 1)[0]
        for title in reason_to_sections.get(reason_key, []):
            add(title)
    return ordered


def relevant_spec_labels(text: str, reasons: list[str] | None = None) -> list[str]:
    return [_display_title(title) for title in relevant_spec_titles(text, reasons)]


def relevant_spec_excerpt(text: str, reasons: list[str] | None = None, *, max_lines: int = 4) -> str:
    titles = relevant_spec_titles(text, reasons)
    return "\n\n".join(compact_spec_section(title, max_lines=max_lines) for title in titles)
