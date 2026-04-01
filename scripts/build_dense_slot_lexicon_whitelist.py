#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_controlled_generation_assets import ROOT


DEFAULT_INPUT = (
    ROOT / "data" / "controlled_generation" / "assets" / "dense_slot_lexicon_reservoir.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT / "data" / "controlled_generation" / "assets" / "dense_slot_lexicon_whitelist.jsonl"
)
DEFAULT_SUMMARY = (
    ROOT / "data" / "controlled_generation" / "reports" / "dense_slot_lexicon_whitelist_summary.json"
)
TARGET_SCENES = [
    "home_life",
    "food_dining",
    "transport_trip",
    "shopping_payment",
    "weather_safety",
]
WEAK_SCENES = {"home_life", "shopping_payment", "weather_safety"}
SEED_ALLOWED_SEMANTIC_CLASSES = {
    "device",
    "place",
    "institution",
    "action",
    "adjective",
    "person",
}
SEED_SCENE_CAPS = {
    "home_life": 24,
    "shopping_payment": 24,
    "weather_safety": 24,
}
SCENE_TARGET_CAPS = {
    "home_life": 80,
    "food_dining": 180,
    "transport_trip": 180,
    "shopping_payment": 80,
    "weather_safety": 40,
}
SCENE_SOURCE_CAPS = {
    ("transport_trip", "place_name"): 80,
}
ALLOWED_SEMANTIC_CLASSES = {
    "place",
    "institution",
    "food",
    "device",
    "transport",
    "person",
    "action",
    "adjective",
}
ALLOWED_SLOT_KINDS = {"noun", "place", "verb", "adjective"}
EXCLUDED_HEADWORD_HINTS = (
    "后置于",
    "前置于",
    "用于",
    "用在",
    "置于",
    "俗语",
    "助词",
    "量词",
    "副词",
    "介词",
    "连词",
    "语气词",
    "代词",
    "形容词",
    "动词",
    "名词",
    "补语",
    "状语",
)
EXCLUDED_DEFINITION_HINTS = (
    "俗语",
    "固定说法",
    "成语",
    "比喻",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_scene_tags(row: dict[str, Any]) -> list[str]:
    return [str(item) for item in (row.get("scene_tags") or []) if str(item)]


def choose_target_scene(row: dict[str, Any], target_scenes: list[str]) -> str | None:
    primary_scene = str(row.get("primary_scene_id") or "")
    if primary_scene in target_scenes:
        return primary_scene
    scene_tags = normalize_scene_tags(row)
    for scene_id in target_scenes:
        if scene_id in scene_tags:
            return scene_id
    return None


def semantic_priority(row: dict[str, Any]) -> int:
    semantic_class = str(row.get("semantic_class") or "")
    order = {
        "place": 5,
        "food": 5,
        "person": 4,
        "device": 4,
        "institution": 4,
        "transport": 4,
        "action": 3,
        "adjective": 2,
    }
    return order.get(semantic_class, 0)


def source_priority(row: dict[str, Any]) -> int:
    source = str(row.get("dense_reservoir_source") or "")
    order = {
        "replaceable_lexicon": 3,
        "example_mined": 2,
        "new_noun": 1,
        "place_name": 0,
    }
    return order.get(source, 0)


def reject_reason(row: dict[str, Any], target_scenes: set[str], min_confidence: float) -> str | None:
    if not row.get("usable_as_slot"):
        return "not_usable_as_slot"
    term = str(row.get("wz_word") or "")
    if not 2 <= len(term) <= 4:
        return "term_length_out_of_range"
    semantic_class = row.get("semantic_class", "unknown")
    if semantic_class not in ALLOWED_SEMANTIC_CLASSES:
        return f"semantic_class_{semantic_class}"
    slot_kind = row.get("slot_kind", "unknown")
    if slot_kind not in ALLOWED_SLOT_KINDS:
        return f"slot_kind_{slot_kind}"
    confidence = float(row.get("confidence", 0.0))
    if confidence < min_confidence:
        return "low_confidence"
    scene_tags = set(normalize_scene_tags(row))
    primary_scene = str(row.get("primary_scene_id") or "")
    if not (scene_tags & target_scenes or primary_scene in target_scenes):
        return "scene_out_of_scope"
    headword = str(row.get("mandarin_headword") or "").strip()
    if any(hint in headword for hint in EXCLUDED_HEADWORD_HINTS):
        return "meta_headword"
    definition = str(row.get("definition") or "").strip()
    if any(hint in definition for hint in EXCLUDED_DEFINITION_HINTS):
        return "definition_fixed_expression"
    if row.get("record_type") == "fixed_expression":
        return "fixed_expression_record"
    source = str(row.get("dense_reservoir_source") or "")
    if (
        not row.get("has_example")
        and source not in {"new_noun", "place_name", "scene_seed"}
    ):
        return "no_example"
    if source == "scene_seed":
        primary_scene = str(row.get("primary_scene_id") or "")
        if primary_scene not in WEAK_SCENES:
            return "scene_seed_out_of_scope"
        if semantic_class not in SEED_ALLOWED_SEMANTIC_CLASSES:
            return "scene_seed_semantic_not_allowed"
    return None


def build_whitelist(
    rows: list[dict[str, Any]],
    target_scenes: list[str],
    min_confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_scene_set = set(target_scenes)
    filtered_rows: list[dict[str, Any]] = []
    rejected_counts = Counter()

    for row in rows:
        reason = reject_reason(row, target_scene_set, min_confidence=min_confidence)
        merged = dict(row)
        if reason is None:
            assigned_scene = choose_target_scene(merged, target_scenes)
            if not assigned_scene:
                rejected_counts["scene_assignment_failed"] += 1
                continue
            merged["dense_whitelist_scene_id"] = assigned_scene
            merged["dense_whitelist_status"] = "keep"
            merged["dense_whitelist_reason"] = "high_precision_dense_keep"
            filtered_rows.append(merged)
        else:
            rejected_counts[reason] += 1

    filtered_rows.sort(
        key=lambda row: (
            -float(row.get("confidence", 0.0)),
            -int(bool(row.get("has_example"))),
            source_priority(row),
            semantic_priority(row),
            -len(str(row.get("wz_word") or "")),
            str(row.get("wz_word") or ""),
        ),
        reverse=True,
    )
    whitelist_rows: list[dict[str, Any]] = []
    scene_counts = Counter()
    scene_source_counts = Counter()
    for row in filtered_rows:
        scene_id = str(row.get("dense_whitelist_scene_id") or "")
        source = str(row.get("dense_reservoir_source") or "")
        if scene_counts[scene_id] >= SCENE_TARGET_CAPS.get(scene_id, 999999):
            continue
        if scene_source_counts[(scene_id, source)] >= SCENE_SOURCE_CAPS.get((scene_id, source), 999999):
            continue
        if source == "scene_seed" and scene_source_counts[(scene_id, source)] >= SEED_SCENE_CAPS.get(scene_id, 0):
            continue
        whitelist_rows.append(row)
        scene_counts[scene_id] += 1
        scene_source_counts[(scene_id, source)] += 1

    summary = {
        "input_rows": len(rows),
        "target_scenes": target_scenes,
        "kept_rows": len(whitelist_rows),
        "kept_scene_counts": dict(Counter(str(row.get("dense_whitelist_scene_id") or "") for row in whitelist_rows)),
        "kept_semantic_class_counts": dict(
            Counter(str(row.get("semantic_class") or "") for row in whitelist_rows)
        ),
        "kept_slot_kind_counts": dict(Counter(str(row.get("slot_kind") or "") for row in whitelist_rows)),
        "kept_source_counts": dict(Counter(str(row.get("dense_reservoir_source") or "") for row in whitelist_rows)),
        "scene_target_caps": SCENE_TARGET_CAPS,
        "scene_source_caps": {f"{scene}:{source}": cap for (scene, source), cap in SCENE_SOURCE_CAPS.items()},
        "seed_scene_caps": SEED_SCENE_CAPS,
        "rejected_reason_counts": dict(rejected_counts),
    }
    return whitelist_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a high-precision whitelist for dense slot replacement."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--min-confidence", type=float, default=0.82)
    parser.add_argument("--scenes", nargs="+", default=TARGET_SCENES)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    whitelist_rows, summary = build_whitelist(
        rows,
        target_scenes=args.scenes,
        min_confidence=args.min_confidence,
    )
    write_jsonl(args.output, whitelist_rows)
    write_json(args.summary, summary)

    print(f"Input rows: {len(rows)}")
    print(f"Whitelist rows: {len(whitelist_rows)}")
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
