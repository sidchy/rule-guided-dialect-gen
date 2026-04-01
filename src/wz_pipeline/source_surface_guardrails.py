from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from .paths import ROOT


CLEANED_RECORDS = ROOT / "data" / "cleaned" / "cleaned_records_primary.jsonl"
EXTRACTED_SHORT = ROOT / "data" / "extracted_training_sentences" / "short_8_20.jsonl"
EXTRACTED_LONG = ROOT / "data" / "extracted_training_sentences" / "long_20_30.jsonl"
REPLACEABLE_LEXICON = ROOT / "data" / "controlled_generation" / "assets" / "replaceable_lexicon.jsonl"
DENSE_WHITELIST = ROOT / "data" / "controlled_generation" / "assets" / "dense_slot_lexicon_whitelist.jsonl"

SOURCE_PATHS = [
    CLEANED_RECORDS,
    EXTRACTED_SHORT,
    EXTRACTED_LONG,
    REPLACEABLE_LEXICON,
    DENSE_WHITELIST,
]

PAREN_RE = re.compile(r"[（(][^）)]{1,4}[）)]")
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
MIN_TRUSTED_NGRAM_COUNT = 2
BOUNDARY_SINGLE_CHARS = set("我你妳尔侬卬伊渠其阿个着了罢啊呢嘛就再还也都把将给在向从跟和同是有未冇不莫会能真显很较更来去起落")


def clean_surface(text: str) -> str:
    cleaned = PAREN_RE.sub("", str(text or ""))
    return re.sub(r"[\s\u3000]+", "", cleaned).strip()


def _iter_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _iter_text_runs(text: str) -> list[str]:
    cleaned = clean_surface(text)
    return [match.group(0) for match in CJK_RUN_RE.finditer(cleaned)]


def _iter_ngrams(text: str, *, min_len: int = 2, max_len: int = 4) -> list[str]:
    runs = _iter_text_runs(text)
    grams: list[str] = []
    for run in runs:
        for size in range(min_len, min(max_len, len(run)) + 1):
            grams.extend(run[idx : idx + size] for idx in range(0, len(run) - size + 1))
    return grams


def _extract_terms(row: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for key in ("wz_word", "wz_word_train", "core_word"):
        value = clean_surface(str(row.get(key) or ""))
        if value:
            terms.append(value)
    support_words = row.get("support_words")
    if isinstance(support_words, list):
        terms.extend(clean_surface(str(item)) for item in support_words if clean_surface(str(item)))
    elif isinstance(support_words, str):
        terms.extend(clean_surface(item) for item in support_words.split("|") if clean_surface(item))
    target_words = row.get("target_words")
    if isinstance(target_words, list):
        terms.extend(clean_surface(str(item)) for item in target_words if clean_surface(str(item)))
    return [term for term in terms if term]


def _extract_texts(row: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for key in ("wz_sentence", "example_wz_train", "example_wz_norm"):
        value = clean_surface(str(row.get(key) or ""))
        if value:
            texts.append(value)
    return texts


@lru_cache(maxsize=1)
def load_source_surface_stats() -> dict[str, Any]:
    term_counts: Counter[str] = Counter()
    ngram_counts: Counter[str] = Counter()
    max_term_len = 1

    for path in SOURCE_PATHS:
        for row in _iter_rows(path):
            for term in _extract_terms(row):
                term_counts[term] += 1
                max_term_len = max(max_term_len, len(term))
                for gram in _iter_ngrams(term):
                    ngram_counts[gram] += 1
            for text in _extract_texts(row):
                for gram in _iter_ngrams(text):
                    ngram_counts[gram] += 1

    trusted_terms = set(term_counts)
    trusted_terms.update(
        gram for gram, count in ngram_counts.items() if len(gram) >= 2 and count >= MIN_TRUSTED_NGRAM_COUNT
    )
    trusted_single_chars = set(BOUNDARY_SINGLE_CHARS)
    max_term_len = max([max_term_len, *(len(term) for term in trusted_terms)], default=1)
    return {
        "term_counts": dict(term_counts),
        "ngram_counts": dict(ngram_counts),
        "trusted_terms": frozenset(trusted_terms),
        "trusted_single_chars": frozenset(trusted_single_chars),
        "max_term_len": max_term_len,
    }


def source_surface_count(surface: str, *, stats: dict[str, Any] | None = None) -> int:
    normalized = clean_surface(surface)
    if not normalized:
        return 0
    payload = stats or load_source_surface_stats()
    return max(
        int(payload["term_counts"].get(normalized, 0)),
        int(payload["ngram_counts"].get(normalized, 0)),
    )


def _tokenize_run(run: str, trusted_terms: set[str], max_term_len: int) -> list[str]:
    tokens: list[str] = []
    idx = 0
    while idx < len(run):
        matched = ""
        upper = min(max_term_len, len(run) - idx)
        for size in range(upper, 1, -1):
            candidate = run[idx : idx + size]
            if candidate in trusted_terms:
                matched = candidate
                break
        if matched:
            tokens.append(matched)
            idx += len(matched)
        else:
            tokens.append(run[idx])
            idx += 1
    return tokens


def _trim_common_edges(text: str, trusted_single_chars: set[str]) -> str:
    trimmed = text
    while len(trimmed) >= 2 and trimmed[:1] in trusted_single_chars:
        trimmed = trimmed[1:]
    while len(trimmed) >= 2 and trimmed[-1:] in trusted_single_chars:
        trimmed = trimmed[:-1]
    return trimmed


def detect_unsupported_surface_terms(
    text: str,
    *,
    protected_terms: list[str] | None = None,
    stats: dict[str, Any] | None = None,
    min_source_count: int = 0,
) -> list[dict[str, Any]]:
    payload = stats or load_source_surface_stats()
    protected = {clean_surface(term) for term in (protected_terms or []) if clean_surface(term)}
    trusted_terms = set(payload["trusted_terms"]) | protected
    trusted_single_chars = set(payload.get("trusted_single_chars", ()))
    candidate_lengths = [int(payload["max_term_len"]), *(len(term) for term in protected)]
    max_term_len = max(candidate_lengths) if candidate_lengths else 1
    findings: list[dict[str, Any]] = []
    seen_surfaces: set[str] = set()

    for run in _iter_text_runs(text):
        tokens = _tokenize_run(run, trusted_terms, max_term_len)
        buffer = ""
        for token in tokens:
            if len(token) == 1 and token not in protected:
                buffer += token
                continue
            if len(buffer) >= 2:
                trimmed = _trim_common_edges(buffer, trusted_single_chars)
                if len(trimmed) >= 2:
                    count = source_surface_count(trimmed, stats=payload)
                    if count <= min_source_count and trimmed not in seen_surfaces:
                        findings.append({"surface": trimmed, "source_count": count})
                        seen_surfaces.add(trimmed)
            buffer = ""
        if len(buffer) >= 2:
            trimmed = _trim_common_edges(buffer, trusted_single_chars)
            if len(trimmed) >= 2:
                count = source_surface_count(trimmed, stats=payload)
                if count <= min_source_count and trimmed not in seen_surfaces:
                    findings.append({"surface": trimmed, "source_count": count})
                    seen_surfaces.add(trimmed)
    return findings
