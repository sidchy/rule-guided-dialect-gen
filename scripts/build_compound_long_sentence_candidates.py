#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "controlled_generation" / "skeletons" / "skeleton_templates.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "controlled_generation" / "long_sentence" / "compound_source_candidates.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "controlled_generation" / "reports" / "compound_long_sentence_summary.json"

TERMINAL_PUNCT_RE = re.compile(r"[。！？!?；;，,、]+$")
PRONOUN_PREFIXES = ("我", "你", "渠", "伊", "阿拉", "我拉", "你拉", "渠拉")
BAD_JOIN_ENDINGS = ("啊？", "吧。", "吗？", "啥？", "呢。")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def strip_terminal_punct(text: str) -> str:
    return TERMINAL_PUNCT_RE.sub("", text).strip()


def guess_subject_prefix(text: str) -> str:
    for prefix in PRONOUN_PREFIXES:
        if text.startswith(prefix):
            return prefix
    return text[:2]


def subject_compatible(a: str, b: str) -> bool:
    a_prefix = guess_subject_prefix(a)
    b_prefix = guess_subject_prefix(b)
    if a_prefix in PRONOUN_PREFIXES and b_prefix in PRONOUN_PREFIXES and a_prefix != b_prefix:
        return False
    return True


def terminal_join_safe(text: str) -> bool:
    return not any(text.endswith(ending) for ending in BAD_JOIN_ENDINGS)


def normalize_sentence_for_join(text: str) -> str:
    return strip_terminal_punct(text).replace(" ", "")


def should_pair(a: dict[str, Any], b: dict[str, Any]) -> tuple[bool, str]:
    if a["scene_id"] != b["scene_id"]:
        return False, "scene_mismatch"
    if a["primary_speech_function"] != b["primary_speech_function"]:
        return False, "speech_mismatch"
    if a["source_record_id"] == b["source_record_id"]:
        return False, "same_source"

    sent_a = a["source_wz_sentence"]
    sent_b = b["source_wz_sentence"]
    if not subject_compatible(sent_a, sent_b):
        return False, "subject_conflict"
    if not terminal_join_safe(sent_a) or not terminal_join_safe(sent_b):
        return False, "terminal_not_safe"

    joined = f"{normalize_sentence_for_join(sent_a)}，{normalize_sentence_for_join(sent_b)}。"
    joined_len = len(joined)
    if joined_len < 20 or joined_len > 30:
        return False, "length_out_of_range"
    return True, "ok"


def build_join_row(a: dict[str, Any], b: dict[str, Any], rank: int) -> dict[str, Any]:
    part_a = normalize_sentence_for_join(a["source_wz_sentence"])
    part_b = normalize_sentence_for_join(b["source_wz_sentence"])
    compound = f"{part_a}，{part_b}。"

    def pack_part(row: dict[str, Any]) -> dict[str, Any]:
        slots = row.get("slots") or []
        return {
            "source_record_id": row["source_record_id"],
            "source_file": row["source_file"],
            "skeleton_id": row["skeleton_id"],
            "wz_sentence": row["source_wz_sentence"],
            "zh_sentence": row["source_zh_sentence"],
            "slot_count": row["slot_count"],
            "slot_terms_wz": [slot.get("surface_wz") for slot in slots if slot.get("surface_wz")],
            "slot_terms_zh": [slot.get("surface_zh") for slot in slots if slot.get("surface_zh")],
            "slot_semantic_classes": [slot.get("semantic_class", "unknown") for slot in slots],
            "slot_kinds": [slot.get("slot_kind", "unknown") for slot in slots],
        }

    return {
        "compound_id": f"compound_{rank:06d}",
        "scene_id": a["scene_id"],
        "scene_label": a["scene_label"],
        "primary_speech_function": a["primary_speech_function"],
        "compound_wz_sentence": compound,
        "compound_length": len(compound),
        "join_strategy": "comma_compound",
        "parts": [
            pack_part(a),
            pack_part(b),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build controlled 20-30 char compound sentence candidates from shorter skeleton sources.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument("--target-min", type=int, default=20)
    parser.add_argument("--target-max", type=int, default=30)
    parser.add_argument("--max-pairs-per-group", type=int, default=300)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    short_rows = [
        row
        for row in rows
        if args.min_length <= len(row.get("source_wz_sentence") or "") <= args.max_length
    ]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in short_rows:
        grouped[(row["scene_id"], row["primary_speech_function"])].append(row)

    output_rows: list[dict[str, Any]] = []
    reject_counts = Counter()
    rank = 1
    for _, bucket in grouped.items():
        limit = min(len(bucket), args.max_pairs_per_group)
        scoped = bucket[:limit]
        for i, row_a in enumerate(scoped):
            for row_b in scoped[i + 1 :]:
                ok, reason = should_pair(row_a, row_b)
                if not ok:
                    reject_counts[reason] += 1
                    continue
                output_rows.append(build_join_row(row_a, row_b, rank=rank))
                rank += 1

    write_jsonl(args.output, output_rows)
    write_json(
        args.summary,
        {
            "input_short_rows": len(short_rows),
            "compound_candidate_rows": len(output_rows),
            "scene_counts": dict(Counter(row["scene_id"] for row in output_rows)),
            "speech_function_counts": dict(Counter(row["primary_speech_function"] for row in output_rows)),
            "length_counts": dict(Counter(row["compound_length"] for row in output_rows)),
            "source_file_counts": dict(
                Counter(part["source_file"] for row in output_rows for part in row["parts"])
            ),
            "reject_reason_counts": dict(reject_counts),
            "output_path": str(args.output),
        },
    )

    print(f"Input short rows: {len(short_rows)}")
    print(f"Compound candidate rows: {len(output_rows)}")
    print(f"Wrote output to {args.output}")
    print(f"Wrote summary to {args.summary}")


if __name__ == "__main__":
    main()
