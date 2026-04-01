#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "cleaned" / "cleaned_records_primary.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "llm_review" / "mandarin_headwords_heuristic.jsonl"

SPLIT_RE = re.compile(r"[.;。；|]")
LEADING_ENUM_RE = re.compile(r"^\d+[.、]\s*")
PARENS_RE = re.compile(r"\([^)]*\)")
INLINE_ENUM_RE = re.compile(r"(?:^|\s)\d+[.、]\s*")
BAD_ONLY_RE = re.compile(r"^[0-9一二三四五六七八九十]+$")

BAD_PREFIXES = [
    "一种",
    "指",
    "形容",
    "量词",
    "专指",
    "旧指",
    "旧时",
    "比喻",
    "泛指",
    "用来",
    "用于",
    "供",
]

BAD_CONTAINS = [
    "表示",
    "常见于",
    "可供",
    "入药",
    "植物",
    "灌木",
    "草本",
    "病",
    "病症",
    "传染病",
]

TAIL_PATTERNS = [
    r"的一个变种$",
    r"的一种$",
    r"的一类$",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_clause(text: str | None) -> str | None:
    if not text:
        return None
    text = INLINE_ENUM_RE.sub("", text.strip())
    text = LEADING_ENUM_RE.sub("", text.strip())
    text = PARENS_RE.sub("", text).strip(" ,，:：;；.")
    text = re.sub(r"\s+", "", text)
    return text or None


def refine_candidate(text: str | None) -> str | None:
    text = clean_clause(text)
    if not text:
        return None
    for pattern in TAIL_PATTERNS:
        text = re.sub(pattern, "", text)
    text = text.strip(" ,，:：;；.")
    if not text or BAD_ONLY_RE.fullmatch(text):
        return None
    return text


def choose_headword(definition: str, example_zh: str | None = None) -> tuple[str | None, float, str]:
    definition = INLINE_ENUM_RE.sub(" ", definition)
    clauses = [clean_clause(part) for part in SPLIT_RE.split(definition)]
    clauses = [c for c in clauses if c]
    if not clauses:
        return None, 0.0, "empty_definition"

    first = refine_candidate(clauses[0])
    if first and len(first) <= 8 and "的" not in first and all(prefix not in first for prefix in BAD_PREFIXES) and all(mark not in first for mark in BAD_CONTAINS):
        return first, 0.92, "short_direct_clause"

    # Strip obvious explanatory prefixes once.
    first_clause = clauses[0]
    for prefix in BAD_PREFIXES:
        if first_clause.startswith(prefix):
            candidate = refine_candidate(first_clause[len(prefix) :])
            if candidate and 1 <= len(candidate) <= 8 and "的" not in candidate and all(mark not in candidate for mark in BAD_CONTAINS):
                return candidate, 0.72, f"prefix_stripped:{prefix}"

    # If first clause contains a comma-separated synonym before explanation, prefer that.
    comma_head = re.split(r"[,，:：]", first_clause)[0]
    comma_head = refine_candidate(comma_head)
    if comma_head and 1 <= len(comma_head) <= 8 and "的" not in comma_head and all(mark not in comma_head for mark in BAD_CONTAINS):
        return comma_head, 0.68, "comma_head"

    # Use the Mandarin example as a weak hint only for very short lexical labels.
    if example_zh:
        example_hint = refine_candidate(re.split(r"[,.，。;；!?？！]", example_zh)[0])
        if example_hint and 1 <= len(example_hint) <= 8 and "的" not in example_hint:
            return example_hint, 0.45, "example_hint"

    return None, 0.0, "no_safe_headword"


def is_fixed_expression(record: dict[str, Any]) -> bool:
    if record["record_type"] == "fixed_expression":
        return True
    wz = record.get("wz_word_train") or ""
    return len(wz) >= 6


def iter_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for record in records:
        if not record.get("is_exact_primary"):
            continue
        if not record.get("v1_candidate"):
            continue
        if record.get("record_type") not in {"dictionary_entry", "lexicon_entry", "fixed_expression"}:
            continue
        if not record.get("wz_word_train") or not record.get("definition_norm"):
            continue
        out.append(record)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristically extract Mandarin headwords.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    candidates = iter_candidates(records)
    rows: list[dict[str, Any]] = []
    for record in candidates:
        headword, confidence, rule = choose_headword(
            record["definition_norm"],
            record.get("example_zh_train"),
        )
        fixed = is_fixed_expression(record)
        rows.append(
            {
                "dedup_exact_key": record["dedup_exact_key"],
                "source_file": record["source_file"],
                "source_row_id": record["source_row_id"],
                "record_type": record["record_type"],
                "wz_word_train": record.get("wz_word_train"),
                "definition_norm": record.get("definition_norm"),
                "example_wz_train": record.get("example_wz_train"),
                "example_zh_train": record.get("example_zh_train"),
                "extraction": {
                    "mandarin_headword": headword,
                    "definition_clean": record.get("definition_norm"),
                    "is_fixed_expression": fixed,
                    "is_good_word_pair": bool(headword) and not fixed and confidence >= 0.68,
                    "confidence": confidence,
                    "notes": rule,
                },
                "extractor": "heuristic_v1",
            }
        )
    write_jsonl(args.output, rows)
    print(f"Candidate pool: {len(candidates)}")
    print(f"Wrote heuristic extractions to {args.output}")


if __name__ == "__main__":
    main()
