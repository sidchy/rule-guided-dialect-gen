#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_controlled_generation_assets import ROOT, guess_semantic_class


DEFAULT_LEXICON = (
    ROOT / "data" / "controlled_generation" / "assets" / "replaceable_lexicon.jsonl"
)
DEFAULT_EXTERNAL = (
    ROOT / "data" / "controlled_generation" / "assets" / "external_term_catalog.json"
)
DEFAULT_OUTPUT = (
    ROOT / "data" / "controlled_generation" / "assets" / "dense_slot_lexicon_reservoir.jsonl"
)
DEFAULT_SUMMARY = (
    ROOT / "data" / "controlled_generation" / "reports" / "dense_slot_lexicon_reservoir_summary.json"
)
DEFAULT_SEED_TERMS = (
    ROOT / "data" / "controlled_generation" / "assets" / "dense_scene_seed_terms.json"
)
TARGET_SCENES = [
    "home_life",
    "food_dining",
    "transport_trip",
    "shopping_payment",
    "weather_safety",
]
ALLOWED_SEMANTIC_CLASSES = {
    "place",
    "institution",
    "food",
    "device",
    "transport",
    "person",
    "action",
    "adjective",
    "service",
    "document",
}
ALLOWED_SLOT_KINDS = {"noun", "place", "verb", "adjective", "unknown"}
WEAK_SCENES = {"home_life", "shopping_payment", "weather_safety"}
WEAK_SCENE_BACKFILL_CAPS = {
    "home_life": 120,
    "shopping_payment": 120,
    "weather_safety": 80,
}
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
    "补语",
    "状语",
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


def keep_base_lexicon_row(row: dict[str, Any], target_scenes: set[str], min_confidence: float) -> bool:
    if not row.get("usable_as_slot"):
        return False
    term = str(row.get("wz_word") or "")
    if not 2 <= len(term) <= 4:
        return False
    semantic_class = str(row.get("semantic_class") or "unknown")
    if semantic_class not in ALLOWED_SEMANTIC_CLASSES:
        return False
    slot_kind = str(row.get("slot_kind") or "unknown")
    if slot_kind not in ALLOWED_SLOT_KINDS:
        return False
    confidence = float(row.get("confidence", 0.0))
    if confidence < min_confidence:
        return False
    scene_tags = set(normalize_scene_tags(row))
    primary_scene = str(row.get("primary_scene_id") or "")
    if not (scene_tags & target_scenes or primary_scene in target_scenes):
        return False
    headword = str(row.get("mandarin_headword") or "").strip()
    if any(hint in headword for hint in EXCLUDED_HEADWORD_HINTS):
        return False
    return True


def maybe_backfill_unknown_row(row: dict[str, Any], target_scenes: set[str]) -> dict[str, Any] | None:
    if not row.get("usable_as_slot"):
        return None
    if not row.get("has_example"):
        return None
    term = str(row.get("wz_word") or "")
    if not 2 <= len(term) <= 4:
        return None
    scene_tags = set(normalize_scene_tags(row))
    primary_scene = str(row.get("primary_scene_id") or "")
    target_scene = primary_scene if primary_scene in WEAK_SCENES else ""
    if not target_scene:
        for scene_id in WEAK_SCENES:
            if scene_id in scene_tags:
                target_scene = scene_id
                break
    if target_scene not in target_scenes:
        return None
    if str(row.get("semantic_class") or "unknown") != "unknown":
        return None
    slot_kind = str(row.get("slot_kind") or "unknown")
    if slot_kind not in {"noun", "verb", "adjective", "place"}:
        return None
    confidence = float(row.get("confidence", 0.0))
    if confidence < 0.82:
        return None
    headword = str(row.get("mandarin_headword") or "").strip()
    if any(hint in headword for hint in EXCLUDED_HEADWORD_HINTS):
        return None
    inferred = guess_semantic_class(
        headword,
        str(row.get("definition") or ""),
        term,
        slot_kind,
        primary_scene_id=target_scene,
    )
    if inferred not in ALLOWED_SEMANTIC_CLASSES:
        return None
    merged = dict(row)
    merged["semantic_class"] = inferred
    merged["primary_scene_id"] = target_scene
    merged["dense_reservoir_source"] = "weak_scene_backfill"
    merged["dense_backfill_reason"] = "unknown_semantic_recovered"
    return merged


def make_external_row(term: str, scene_id: str, semantic_class: str, source_type: str) -> dict[str, Any]:
    return {
        "lexicon_id": f"dense_external::{source_type}::{term}",
        "wz_word": term,
        "mandarin_headword": term,
        "definition": f"dense reservoir external {source_type}",
        "slot_kind": "noun" if semantic_class not in {"action", "adjective", "place"} else (
            "verb" if semantic_class == "action" else ("adjective" if semantic_class == "adjective" else "place")
        ),
        "semantic_class": semantic_class,
        "confidence": 0.9 if source_type != "example_mined" else 0.86,
        "source_file": "external_term_catalog",
        "source_row_id": None,
        "record_type": source_type,
        "usable_as_slot": True,
        "has_example": source_type == "example_mined",
        "scene_tags": [scene_id],
        "scene_scores": {scene_id: 1},
        "primary_scene_id": scene_id,
        "primary_scene_confidence": 1.0,
        "topic_scene_tags": [scene_id],
        "topic_scene_scores": {scene_id: 1},
        "primary_topic_scene": scene_id,
        "topic_scene_confidence": 1.0,
        "dense_reservoir_source": source_type,
    }


def build_reservoir(
    lexicon_rows: list[dict[str, Any]],
    external_catalog: dict[str, Any],
    seed_terms: dict[str, list[dict[str, Any]]],
    target_scenes: list[str],
    min_confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_scene_set = set(target_scenes)
    reservoir_rows: list[dict[str, Any]] = []
    seen_terms: set[tuple[str, str]] = set()
    weak_scene_backfill_counts = Counter()

    for row in lexicon_rows:
        if not keep_base_lexicon_row(row, target_scene_set, min_confidence):
            continue
        merged = dict(row)
        merged["dense_reservoir_source"] = "replaceable_lexicon"
        key = (str(merged.get("wz_word") or ""), str(merged.get("primary_scene_id") or ""))
        if key in seen_terms:
            continue
        seen_terms.add(key)
        reservoir_rows.append(merged)

    for row in lexicon_rows:
        merged = maybe_backfill_unknown_row(row, target_scene_set)
        if not merged:
            continue
        scene_id = str(merged.get("primary_scene_id") or "")
        if weak_scene_backfill_counts[scene_id] >= WEAK_SCENE_BACKFILL_CAPS.get(scene_id, 0):
            continue
        key = (str(merged.get("wz_word") or ""), scene_id)
        if key in seen_terms:
            continue
        seen_terms.add(key)
        weak_scene_backfill_counts[scene_id] += 1
        reservoir_rows.append(merged)

    for item in external_catalog.get("new_nouns") or []:
        term = str(item.get("term") or "")
        scene_id = str(item.get("primary_scene_id") or "")
        semantic_class = str(item.get("semantic_class") or "unknown")
        if not term or not 2 <= len(term) <= 4 or scene_id not in target_scene_set:
            continue
        if semantic_class not in ALLOWED_SEMANTIC_CLASSES:
            semantic_class = guess_semantic_class(term, term, term, "noun", primary_scene_id=scene_id)
        if semantic_class not in ALLOWED_SEMANTIC_CLASSES:
            continue
        key = (term, scene_id)
        if key in seen_terms:
            continue
        seen_terms.add(key)
        reservoir_rows.append(make_external_row(term, scene_id, semantic_class, "new_noun"))

    for item in external_catalog.get("place_names") or []:
        term = str(item.get("term") or "")
        scene_id = "transport_trip"
        if not term or not 2 <= len(term) <= 4:
            continue
        key = (term, scene_id)
        if key in seen_terms:
            continue
        seen_terms.add(key)
        reservoir_rows.append(make_external_row(term, scene_id, "place", "place_name"))

    for item in external_catalog.get("example_mined_terms") or []:
        term = str(item.get("term") or "")
        scene_id = str(item.get("primary_topic_scene") or "")
        semantic_class = str(item.get("semantic_class") or "unknown")
        if not term or not 2 <= len(term) <= 4 or scene_id not in target_scene_set:
            continue
        if semantic_class not in ALLOWED_SEMANTIC_CLASSES:
            continue
        key = (term, scene_id)
        if key in seen_terms:
            continue
        seen_terms.add(key)
        reservoir_rows.append(make_external_row(term, scene_id, semantic_class, "example_mined"))

    for scene_id, items in seed_terms.items():
        if scene_id not in target_scene_set:
            continue
        for item in items or []:
            term = str(item.get("term") or "")
            semantic_class = str(item.get("semantic_class") or "unknown")
            if not term or not 2 <= len(term) <= 4:
                continue
            if semantic_class not in ALLOWED_SEMANTIC_CLASSES:
                continue
            key = (term, scene_id)
            if key in seen_terms:
                continue
            seen_terms.add(key)
            reservoir_rows.append(make_external_row(term, scene_id, semantic_class, "scene_seed"))

    reservoir_rows.sort(
        key=lambda row: (
            str(row.get("primary_scene_id") or ""),
            str(row.get("semantic_class") or ""),
            -float(row.get("confidence", 0.0)),
            str(row.get("wz_word") or ""),
        )
    )
    summary = {
        "input_lexicon_rows": len(lexicon_rows),
        "target_scenes": target_scenes,
        "reservoir_rows": len(reservoir_rows),
        "scene_counts": dict(Counter(str(row.get("primary_scene_id") or "") for row in reservoir_rows)),
        "semantic_class_counts": dict(Counter(str(row.get("semantic_class") or "") for row in reservoir_rows)),
        "source_counts": dict(Counter(str(row.get("dense_reservoir_source") or "") for row in reservoir_rows)),
        "weak_scene_backfill_counts": dict(weak_scene_backfill_counts),
    }
    return reservoir_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a broad dense lexicon reservoir.")
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed-terms", type=Path, default=DEFAULT_SEED_TERMS)
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--scenes", nargs="+", default=TARGET_SCENES)
    args = parser.parse_args()

    lexicon_rows = load_jsonl(args.lexicon)
    external_catalog = json.loads(args.external.read_text(encoding="utf-8")) if args.external.exists() else {}
    seed_terms = json.loads(args.seed_terms.read_text(encoding="utf-8")) if args.seed_terms.exists() else {}
    reservoir_rows, summary = build_reservoir(
        lexicon_rows,
        external_catalog,
        seed_terms,
        target_scenes=args.scenes,
        min_confidence=args.min_confidence,
    )
    write_jsonl(args.output, reservoir_rows)
    write_json(args.summary, summary)

    print(f"Input lexicon rows: {len(lexicon_rows)}")
    print(f"Reservoir rows: {len(reservoir_rows)}")
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
