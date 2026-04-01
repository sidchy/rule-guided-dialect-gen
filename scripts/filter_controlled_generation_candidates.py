#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "controlled_generation" / "candidates" / "generation_candidates_raw.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "controlled_generation" / "filtered"
SLOT_RE = re.compile(r"(\[SLOT_\d+\])")


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


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", "", text)


def compile_template_regex(template: str) -> tuple[re.Pattern[str], list[str]]:
    parts = SLOT_RE.split(template)
    slot_ids = [part[1:-1] for part in parts if part.startswith("[SLOT_")]
    pattern = "^"
    for part in parts:
        if part.startswith("[SLOT_"):
            pattern += "(.+?)"
        else:
            pattern += re.escape(part)
    pattern += "$"
    return re.compile(pattern), slot_ids


def extract_slot_values(template: str, sentence: str) -> tuple[bool, dict[str, str]]:
    pattern, slot_ids = compile_template_regex(template)
    match = pattern.match(sentence)
    if not match:
        return False, {}
    groups = match.groups()
    slot_values = {slot_id: value for slot_id, value in zip(slot_ids, groups)}
    if any(not value.strip() for value in slot_values.values()):
        return False, slot_values
    return True, slot_values


def crowded_required_in_slots(required_words: list[str], extracted_slot_values: dict[str, str]) -> bool:
    if len(required_words) <= 1:
        return False
    for value in extracted_slot_values.values():
        hit_count = sum(1 for word in required_words if word and word in value)
        if hit_count >= 2:
            return True
    return False


def build_slot_pool_term_map(slot_candidate_pools: dict[str, Any]) -> dict[str, set[str]]:
    term_map: dict[str, set[str]] = {}
    for slot_id, pool in (slot_candidate_pools or {}).items():
        if isinstance(pool, list):
            term_map[slot_id] = {
                str(item.get("term")).strip()
                for item in pool
                if isinstance(item, dict) and str(item.get("term") or "").strip()
            }
        else:
            term_map[slot_id] = set()
    return term_map


def dense_slot_metrics(row: dict[str, Any], extracted_slot_values: dict[str, str]) -> dict[str, Any]:
    eligible_slot_ids = [str(item) for item in (row.get("eligible_slot_ids") or []) if str(item)]
    required_slot_replacements = [
        str(item) for item in (row.get("required_slot_replacements") or []) if str(item)
    ]
    slot_pool_map = build_slot_pool_term_map(row.get("slot_candidate_pools") or {})
    slot_values_model = row.get("slot_values_model") or {}

    if not eligible_slot_ids:
        return {
            "eligible_slot_replacement_ratio": None,
            "pool_compliance_ratio": None,
            "required_slot_replacements_hit": 0,
            "pool_cross_slot_violation": False,
            "dense_rewrite_pass": None,
        }

    replaced_hits = 0
    pool_hits = 0
    required_hits = 0
    cross_slot_violation = False
    for slot_id in eligible_slot_ids:
        source_value = (row.get("slot_source_values") or {}).get(slot_id)
        extracted_value = extracted_slot_values.get(slot_id)
        chosen_value = str(slot_values_model.get(slot_id) or extracted_value or "").strip()
        if extracted_value and source_value and extracted_value != source_value:
            replaced_hits += 1
        if slot_id in required_slot_replacements and extracted_value and source_value and extracted_value != source_value:
            required_hits += 1
        if chosen_value and chosen_value in slot_pool_map.get(slot_id, set()):
            pool_hits += 1
        elif chosen_value:
            other_slots = [other for other in eligible_slot_ids if other != slot_id]
            if any(chosen_value in slot_pool_map.get(other, set()) for other in other_slots):
                cross_slot_violation = True

    eligible_ratio = round(replaced_hits / max(len(eligible_slot_ids), 1), 4)
    pool_ratio = round(pool_hits / max(len(eligible_slot_ids), 1), 4)
    dense_rewrite_pass = (
        row.get("task_type") == "dense_slot_replacement"
        and eligible_ratio == 1.0
        and pool_ratio == 1.0
        and required_hits == len(required_slot_replacements)
        and not cross_slot_violation
    )
    return {
        "eligible_slot_replacement_ratio": eligible_ratio,
        "pool_compliance_ratio": pool_ratio,
        "required_slot_replacements_hit": required_hits,
        "pool_cross_slot_violation": cross_slot_violation,
        "dense_rewrite_pass": dense_rewrite_pass,
    }


def compute_quality(row: dict[str, Any], duplicate: bool) -> dict[str, Any]:
    sentence = row.get("sentence")
    if not isinstance(sentence, str) or not sentence.strip():
        return {
            "skeleton_preserved": False,
            "extracted_slot_values": {},
            "required_words_hit": 0,
            "preferred_words_hit": 0,
            "slot_values_changed": 0,
            "content_changed": False,
            "crowded_required_words": False,
            "duplicate": duplicate,
            "length_ratio": 0.0,
            "quality_score": 0.0,
            "accepted": False,
            "needs_manual_review": True,
        }
    source = row["source_wz_sentence"]
    preserved, extracted_slot_values = extract_slot_values(row["skeleton_template"], sentence)
    required_words = row.get("required_words") or []
    preferred_words = row.get("preferred_words") or []
    required_hits = sum(1 for word in required_words if word in sentence)
    preferred_hits = sum(1 for word in preferred_words if word in sentence)
    slot_source_values = row.get("slot_source_values") or {}
    min_slot_changes = int((row.get("quality_gate") or {}).get("min_slot_changes") or 1)
    slot_values_changed = sum(
        1
        for slot_id, source_value in slot_source_values.items()
        if extracted_slot_values.get(slot_id) and extracted_slot_values.get(slot_id) != source_value
    )
    content_changed = normalize_text(sentence) != normalize_text(source)
    length_ratio = round(len(sentence) / max(len(source), 1), 4)
    required_ok = required_hits == len(required_words)
    crowded_required = crowded_required_in_slots(required_words, extracted_slot_values)
    dense_metrics = dense_slot_metrics(row, extracted_slot_values)
    length_ok = 0.7 <= length_ratio <= 1.6
    quality = 0.0
    if preserved:
        quality += 0.45
    if required_ok:
        quality += 0.2
    if preferred_words:
        quality += 0.1 * (preferred_hits / len(preferred_words))
    else:
        quality += 0.1
    if slot_source_values:
        quality += 0.15 * (slot_values_changed / len(slot_source_values))
    if slot_values_changed >= min_slot_changes:
        quality += 0.05
    if content_changed:
        quality += 0.05
    if length_ok:
        quality += 0.05
    if crowded_required:
        quality -= 0.2
    if dense_metrics["eligible_slot_replacement_ratio"] is not None:
        quality += 0.05 * dense_metrics["eligible_slot_replacement_ratio"]
        quality += 0.05 * dense_metrics["pool_compliance_ratio"]
        if dense_metrics["pool_cross_slot_violation"]:
            quality -= 0.25
    quality = round(min(quality, 0.99), 4)
    base_accepted = (
        preserved
        and required_ok
        and content_changed
        and slot_values_changed >= min_slot_changes
        and not duplicate
        and length_ok
        and not crowded_required
    )
    accepted = base_accepted
    if row.get("task_type") == "dense_slot_replacement":
        accepted = (
            preserved
            and content_changed
            and not duplicate
            and length_ok
            and bool(dense_metrics["dense_rewrite_pass"])
        )
    needs_review = preserved and not accepted
    metrics = {
        "skeleton_preserved": preserved,
        "extracted_slot_values": extracted_slot_values,
        "required_words_hit": required_hits,
        "preferred_words_hit": preferred_hits,
        "slot_values_changed": slot_values_changed,
        "min_slot_changes_required": min_slot_changes,
        "content_changed": content_changed,
        "crowded_required_words": crowded_required,
        "duplicate": duplicate,
        "length_ratio": length_ratio,
        "quality_score": quality,
        "accepted": accepted,
        "needs_manual_review": needs_review,
    }
    metrics.update(dense_metrics)
    return metrics


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter and score generated controlled Wenzhou sentence candidates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    seen_sentences: set[str] = set()
    scored_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []

    for row in rows:
        normalized_sentence = normalize_text(row.get("sentence"))
        duplicate = normalized_sentence in seen_sentences
        if normalized_sentence:
            seen_sentences.add(normalized_sentence)
        quality = compute_quality(row, duplicate=duplicate)
        merged = dict(row)
        merged.update(quality)
        scored_rows.append(merged)
        if merged["accepted"]:
            accepted_rows.append(merged)
        elif merged["needs_manual_review"]:
            review_rows.append(merged)

    output_dir: Path = args.output_dir
    write_jsonl(output_dir / "generation_candidates_scored.jsonl", scored_rows)
    write_jsonl(output_dir / "generation_candidates_accepted.jsonl", accepted_rows)
    write_jsonl(output_dir / "generation_candidates_review.jsonl", review_rows)

    summary = {
        "input_rows": len(rows),
        "scored_rows": len(scored_rows),
        "accepted_rows": len(accepted_rows),
        "review_rows": len(review_rows),
        "crowded_required_rows": sum(1 for row in scored_rows if row["crowded_required_words"]),
        "accepted_multi_slot_changes": sum(1 for row in accepted_rows if row["slot_values_changed"] >= 2),
        "dense_task_rows": sum(1 for row in scored_rows if row.get("task_type") == "dense_slot_replacement"),
        "dense_rewrite_pass_rows": sum(1 for row in scored_rows if row.get("dense_rewrite_pass") is True),
        "dense_accepted_rows": sum(
            1 for row in accepted_rows if row.get("task_type") == "dense_slot_replacement"
        ),
        "dense_pool_cross_slot_violations": sum(
            1 for row in scored_rows if row.get("pool_cross_slot_violation")
        ),
        "accepted_scene_counts": dict(Counter(row["scene_id"] for row in accepted_rows)),
        "accepted_quality_distribution": dict(Counter(round(row["quality_score"], 1) for row in accepted_rows)),
    }
    write_summary(output_dir / "filter_summary.json", summary)

    print(f"Input candidate rows: {len(rows)}")
    print(f"Accepted rows: {len(accepted_rows)}")
    print(f"Review rows: {len(review_rows)}")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
