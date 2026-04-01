#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "controlled_generation" / "long_sentence" / "compound_source_candidates.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "controlled_generation" / "long_sentence" / "self_checked"
DEFAULT_SUMMARY = ROOT / "data" / "controlled_generation" / "reports" / "compound_self_check_summary.json"
EXCLUDED_SEMANTIC_CLASSES = {"unknown", "action", "adjective", "time"}
WEAK_SHARED_TERMS = {
    "我",
    "你",
    "渠",
    "伊",
    "阿拉",
    "我拉",
    "你拉",
    "渠拉",
    "别人",
    "个人",
    "大家人",
}
WEAK_SHARED_ZH = {"单数第一人称", "单数第二人称", "单数第三人称", "他人", "人"}
PRONOUN_PREFIXES = ("我", "你", "渠", "伊", "阿拉", "我拉", "你拉", "渠拉")
PUNCT_RE = re.compile(r"[，,；;。！？!?]")


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


def subject_prefix(text: str) -> str:
    for prefix in PRONOUN_PREFIXES:
        if text.startswith(prefix):
            return prefix
    return text[:2]


def normalize_text(text: str) -> str:
    return PUNCT_RE.sub("", text).replace(" ", "")


def semantic_set(part: dict[str, Any]) -> set[str]:
    return {
        item
        for item in (part.get("slot_semantic_classes") or [])
        if item and item not in EXCLUDED_SEMANTIC_CLASSES
    }


def text_overlap(a: list[str], b: list[str]) -> set[str]:
    return {item for item in a if item and item in set(b)}


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    part_a, part_b = row["parts"]
    sentence_a = part_a["wz_sentence"]
    sentence_b = part_b["wz_sentence"]
    norm_a = normalize_text(sentence_a)
    norm_b = normalize_text(sentence_b)

    shared_wz = sorted(text_overlap(part_a.get("slot_terms_wz") or [], part_b.get("slot_terms_wz") or []))
    shared_zh = sorted(text_overlap(part_a.get("slot_terms_zh") or [], part_b.get("slot_terms_zh") or []))
    shared_semantics = sorted(semantic_set(part_a) & semantic_set(part_b))
    strong_shared_wz = [item for item in shared_wz if item not in WEAK_SHARED_TERMS and len(item) >= 2]
    strong_shared_zh = [item for item in shared_zh if item not in WEAK_SHARED_ZH and len(item) >= 2]
    strong_shared_semantics = [item for item in shared_semantics if item != "person"]
    explicit_subject_match = subject_prefix(sentence_a) == subject_prefix(sentence_b)
    punct_count = len(PUNCT_RE.findall(row["compound_wz_sentence"]))
    length = row["compound_length"]
    length_centered = 23 <= length <= 29
    balanced_parts = abs(len(sentence_a) - len(sentence_b)) <= 6
    slot_density = (part_a.get("slot_count") or 0) + (part_b.get("slot_count") or 0)
    similarity_ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()

    hard_reasons: list[str] = []
    soft_reasons: list[str] = []

    if row["primary_speech_function"] in {"statement", "evaluation"} and not (strong_shared_wz or strong_shared_zh or strong_shared_semantics):
        hard_reasons.append("topic_weak")
    if not (strong_shared_wz or strong_shared_zh or strong_shared_semantics) and row["primary_speech_function"] in {"reminder", "planning"}:
        hard_reasons.append("topic_bridge_weak")
    if norm_a == norm_b:
        hard_reasons.append("duplicate_clause")
    elif norm_a in norm_b or norm_b in norm_a:
        hard_reasons.append("near_duplicate_clause")
    elif similarity_ratio >= 0.72:
        hard_reasons.append("high_similarity_clause")
    if punct_count > 3:
        hard_reasons.append("punctuation_heavy")
    if slot_density < 3 and not (strong_shared_wz or strong_shared_zh or strong_shared_semantics):
        hard_reasons.append("low_slot_support")

    score = 0.0
    if length_centered:
        score += 0.2
    else:
        score += 0.1
    if balanced_parts:
        score += 0.12
    if explicit_subject_match:
        score += 0.14
    if strong_shared_wz:
        score += 0.22
    elif shared_wz:
        score += 0.08
    if strong_shared_zh:
        score += 0.18
    elif shared_zh:
        score += 0.06
    if strong_shared_semantics:
        score += 0.18
    elif shared_semantics:
        score += 0.05
    if slot_density >= 4:
        score += 0.08
    elif slot_density >= 3:
        score += 0.04

    if row["primary_speech_function"] == "question":
        soft_reasons.append("question_join_needs_review")
        score -= 0.06
    if row["primary_speech_function"] == "explanation":
        soft_reasons.append("explanation_join_needs_review")
        score -= 0.04
    if not explicit_subject_match:
        soft_reasons.append("subject_not_explicitly_matched")
        score -= 0.04
    if not strong_shared_wz and not strong_shared_zh and strong_shared_semantics:
        soft_reasons.append("semantic_only_match")
        score -= 0.03
    if similarity_ratio >= 0.6:
        soft_reasons.append("similarity_risk")
        score -= 0.04

    score = round(max(min(score, 0.99), 0.0), 4)
    accepted = not hard_reasons and score >= 0.5
    needs_review = not accepted and score >= 0.36
    return {
        "shared_slot_terms_wz": shared_wz,
        "shared_slot_terms_zh": shared_zh,
        "shared_semantic_classes": shared_semantics,
        "strong_shared_slot_terms_wz": strong_shared_wz,
        "strong_shared_slot_terms_zh": strong_shared_zh,
        "strong_shared_semantic_classes": strong_shared_semantics,
        "explicit_subject_match": explicit_subject_match,
        "punctuation_count": punct_count,
        "slot_density": slot_density,
        "similarity_ratio": round(similarity_ratio, 4),
        "self_check_score": score,
        "self_check_hard_reasons": hard_reasons,
        "self_check_soft_reasons": soft_reasons,
        "self_check_pass": accepted,
        "needs_manual_review": needs_review,
    }


def rank_key(row: dict[str, Any]) -> tuple[float, int, int, str]:
    return (
        row.get("self_check_score", 0.0),
        len(row.get("shared_slot_terms_wz") or []) + len(row.get("shared_slot_terms_zh") or []),
        len(row.get("shared_semantic_classes") or []),
        row["compound_id"],
    )


def choose_pilot_rows(rows: list[dict[str, Any]], pilot_size: int, scene_quota: int, max_per_scene: int) -> list[dict[str, Any]]:
    scene_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene_buckets[row["scene_id"]].append(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    scene_counts: Counter[str] = Counter()

    for scene_id in sorted(scene_buckets):
        for row in sorted(scene_buckets[scene_id], key=rank_key, reverse=True):
            if scene_counts[scene_id] >= scene_quota:
                break
            selected.append(row)
            selected_ids.add(row["compound_id"])
            scene_counts[scene_id] += 1

    for row in sorted(rows, key=rank_key, reverse=True):
        if len(selected) >= pilot_size:
            break
        if row["compound_id"] in selected_ids:
            continue
        if scene_counts[row["scene_id"]] >= max_per_scene:
            continue
        selected.append(row)
        selected_ids.add(row["compound_id"])
        scene_counts[row["scene_id"]] += 1

    selected.sort(key=lambda row: (row["scene_id"], -row["self_check_score"], row["compound_id"]))
    return selected[:pilot_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-check and pilot-sample compound 20-30 char sentence candidates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--pilot-size", type=int, default=24)
    parser.add_argument("--scene-quota", type=int, default=1)
    parser.add_argument("--max-per-scene", type=int, default=4)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    scored_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for row in rows:
        merged = dict(row)
        merged.update(score_row(row))
        scored_rows.append(merged)
        if merged["self_check_pass"]:
            accepted_rows.append(merged)
        elif merged["needs_manual_review"]:
            review_rows.append(merged)

    pilot_rows = choose_pilot_rows(accepted_rows, args.pilot_size, args.scene_quota, args.max_per_scene)

    output_dir = args.output_dir
    write_jsonl(output_dir / "compound_candidates_scored.jsonl", scored_rows)
    write_jsonl(output_dir / "compound_candidates_accepted.jsonl", accepted_rows)
    write_jsonl(output_dir / "compound_candidates_review.jsonl", review_rows)
    write_jsonl(output_dir / "compound_candidates_pilot.jsonl", pilot_rows)

    hard_reason_counts = Counter(reason for row in scored_rows for reason in row["self_check_hard_reasons"])
    soft_reason_counts = Counter(reason for row in scored_rows for reason in row["self_check_soft_reasons"])
    write_json(
        args.summary,
        {
            "input_rows": len(rows),
            "accepted_rows": len(accepted_rows),
            "review_rows": len(review_rows),
            "pilot_rows": len(pilot_rows),
            "scene_counts": dict(Counter(row["scene_id"] for row in accepted_rows)),
            "speech_function_counts": dict(Counter(row["primary_speech_function"] for row in accepted_rows)),
            "score_distribution": dict(Counter(round(row["self_check_score"], 1) for row in accepted_rows)),
            "hard_reason_counts": dict(hard_reason_counts),
            "soft_reason_counts": dict(soft_reason_counts),
            "auto_correction_signals": [
                "如果 topic_weak 过高，下一轮强制要求共享 slot term 或共享非 unknown semantic class。",
                "如果 punctuation_heavy 过高，下一轮只允许无内部逗号句参与拼接。",
                "如果 question_join_needs_review 过高，下一轮先停用 question/explanation 拼接。",
            ],
            "pilot_output_path": str(output_dir / "compound_candidates_pilot.jsonl"),
        },
    )

    print(f"Input rows: {len(rows)}")
    print(f"Accepted rows: {len(accepted_rows)}")
    print(f"Review rows: {len(review_rows)}")
    print(f"Pilot rows: {len(pilot_rows)}")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
