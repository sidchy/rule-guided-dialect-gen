from __future__ import annotations

import re
from functools import lru_cache

from .paths import DOCS_DIR


GRAMMAR_SPEC_PATH = DOCS_DIR / "温州话生成语法规范_潘悟云原书67-83页.md"
GENERATION_SECTION_TITLES = [
    "## 2. 生成总原则",
    "### 3.1 完成体：`爻`",
    "### 3.2 进行体：`著埭 + V`",
    "### 3.3 持续体：`V + 著埭`",
    "### 3.4 已然体：重读 `罢` 与句末轻读 `罢`",
    "### 3.5 起始体：`起`",
    "### 3.6 继续体：`落去`",
    "### 9.1 `不`",
    "### 9.2 `未`",
    "### 9.3 `冇` 的否定用法",
]
MARKER_SECTION_TITLES = {
    "爻": ["### 3.1 完成体：`爻`"],
    "罢": ["### 3.4 已然体：重读 `罢` 与句末轻读 `罢`"],
    "著埭": ["### 3.2 进行体：`著埭 + V`", "### 3.3 持续体：`V + 著埭`"],
    "起": ["### 3.5 起始体：`起`"],
    "落去": ["### 3.6 继续体：`落去`"],
    "不": ["### 9.1 `不`"],
    "未": ["### 9.2 `未`"],
    "冇": ["### 9.3 `冇` 的否定用法"],
}
REASON_SECTION_TITLES = {
    "completion_object_order": ["### 3.1 完成体：`爻`"],
    "completion_after_modal": ["### 3.1 完成体：`爻`"],
    "completion_marker_overused": ["### 3.1 完成体：`爻`"],
    "completion_followed_by_clause": ["### 3.1 完成体：`爻`"],
    "sentence_final_ba_overused": ["### 3.4 已然体：重读 `罢` 与句末轻读 `罢`"],
    "stative_progressive": ["### 3.2 进行体：`著埭 + V`"],
    "dynamic_postposed_zhedai": ["### 3.3 持续体：`V + 著埭`"],
    "qishi_object_order": ["### 3.5 起始体：`起`"],
    "continuative_object_order": ["### 3.6 继续体：`落去`"],
    "mandarin_negation": ["### 9.1 `不`", "### 9.2 `未`", "### 9.3 `冇` 的否定用法"],
}
HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$")


def _compact_lines(body: str, max_lines: int = 4) -> list[str]:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    return lines[:max_lines]


def _display_title(title: str) -> str:
    return title.lstrip("#").strip()


@lru_cache(maxsize=1)
def load_grammar_spec_sections() -> dict[str, str]:
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


def generation_spec_excerpt(*, max_lines: int = 3) -> str:
    return "\n\n".join(compact_spec_section(title, max_lines=max_lines) for title in GENERATION_SECTION_TITLES)


def relevant_spec_titles(text: str, reasons: list[str] | None = None) -> list[str]:
    sentence = str(text or "")
    ordered: list[str] = []
    seen: set[str] = set()

    def add(title: str) -> None:
        if title not in seen:
            seen.add(title)
            ordered.append(title)

    add("## 2. 生成总原则")
    for marker, titles in MARKER_SECTION_TITLES.items():
        if marker in sentence:
            for title in titles:
                add(title)
    for reason in reasons or []:
        reason_key = str(reason).split(":", 1)[0]
        for title in REASON_SECTION_TITLES.get(reason_key, []):
            add(title)
    return ordered


def relevant_spec_labels(text: str, reasons: list[str] | None = None) -> list[str]:
    return [_display_title(title) for title in relevant_spec_titles(text, reasons)]


def relevant_spec_excerpt(text: str, reasons: list[str] | None = None, *, max_lines: int = 4) -> str:
    titles = relevant_spec_titles(text, reasons)
    return "\n\n".join(compact_spec_section(title, max_lines=max_lines) for title in titles)
