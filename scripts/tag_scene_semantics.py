#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_controlled_generation_assets import (
    ROOT,
    build_scene_payload,
    build_speech_function_payload,
)


DEFAULT_LEXICON = ROOT / "data" / "controlled_generation" / "assets" / "replaceable_lexicon.jsonl"
DEFAULT_SKELETONS = ROOT / "data" / "controlled_generation" / "skeletons" / "skeleton_templates.jsonl"
DEFAULT_EXTERNAL = ROOT / "data" / "controlled_generation" / "assets" / "external_term_catalog.json"
DEFAULT_EXAMPLE_MINED = ROOT / "data" / "controlled_generation" / "assets" / "example_mined_lexicon.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "controlled_generation" / "tags"


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


def enrich_lexicon(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        scene = build_scene_payload(
            row.get("mandarin_headword") or "",
            row.get("definition") or "",
            row.get("wz_word") or "",
        )
        speech = build_speech_function_payload(
            row.get("definition") or "",
            row.get("wz_word") or "",
        )
        enriched.append(
            {
                "lexicon_id": row["lexicon_id"],
                "wz_word": row.get("wz_word"),
                "mandarin_headword": row.get("mandarin_headword"),
                "usable_as_slot": row.get("usable_as_slot", False),
                "primary_topic_scene": scene["primary_scene_id"],
                "topic_scene_tags": scene["scene_tags"],
                "topic_scene_scores": scene["scene_scores"],
                "topic_scene_confidence": scene["primary_scene_confidence"],
                "primary_speech_function": speech["primary_speech_function"],
                "speech_function_tags": speech["speech_function_tags"],
                "speech_function_scores": speech["speech_function_scores"],
                "speech_function_confidence": speech["speech_function_confidence"],
            }
        )
    return enriched


def enrich_skeletons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        scene = build_scene_payload(
            row.get("source_zh_sentence") or "",
            row.get("source_wz_sentence") or "",
            row.get("skeleton_template") or "",
        )
        speech = build_speech_function_payload(
            row.get("source_zh_sentence") or "",
            row.get("source_wz_sentence") or "",
            row.get("skeleton_template") or "",
        )
        enriched.append(
            {
                "skeleton_id": row["skeleton_id"],
                "source_record_id": row.get("source_record_id"),
                "source_wz_sentence": row.get("source_wz_sentence"),
                "skeleton_template": row.get("skeleton_template"),
                "slot_count": row.get("slot_count"),
                "primary_topic_scene": scene["primary_scene_id"],
                "topic_scene_tags": scene["scene_tags"],
                "topic_scene_scores": scene["scene_scores"],
                "topic_scene_confidence": scene["primary_scene_confidence"],
                "primary_speech_function": speech["primary_speech_function"],
                "speech_function_tags": speech["speech_function_tags"],
                "speech_function_scores": speech["speech_function_scores"],
                "speech_function_confidence": speech["speech_function_confidence"],
            }
        )
    return enriched


def enrich_external(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for key in ["new_nouns", "place_names", "example_mined_terms"]:
        rows = payload.get(key) or []
        enriched: list[dict[str, Any]] = []
        for row in rows:
            term = row.get("term") or ""
            scene = build_scene_payload(term)
            speech = build_speech_function_payload(term)
            enriched.append(
                {
                    "term": term,
                    "term_type": row.get("term_type"),
                    "primary_topic_scene": scene["primary_scene_id"],
                    "topic_scene_tags": scene["scene_tags"],
                    "topic_scene_scores": scene["scene_scores"],
                    "topic_scene_confidence": scene["primary_scene_confidence"],
                    "primary_speech_function": speech["primary_speech_function"],
                    "speech_function_tags": speech["speech_function_tags"],
                    "speech_function_scores": speech["speech_function_scores"],
                    "speech_function_confidence": speech["speech_function_confidence"],
                }
            )
        out[key] = enriched
    return out


def enrich_example_mined(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        term = row.get("term") or row.get("wz_word") or ""
        examples = row.get("evidence_examples") or []
        if row.get("llm_scene_review"):
            scene = {
                "primary_scene_id": row.get("primary_topic_scene") or row.get("primary_scene_id") or "daily_chat",
                "scene_tags": row.get("topic_scene_tags") or row.get("scene_tags") or ["daily_chat"],
                "scene_scores": row.get("topic_scene_scores") or row.get("scene_scores") or {},
                "primary_scene_confidence": row.get("topic_scene_confidence", 0.0),
            }
        else:
            scene = build_scene_payload(
                term,
                row.get("mandarin_headword") or "",
                row.get("wz_word") or "",
                *examples,
            )
        speech = build_speech_function_payload(
            term,
            row.get("wz_word") or "",
            *examples,
        )
        enriched.append(
            {
                "term": term,
                "wz_word": row.get("wz_word"),
                "mandarin_headword": row.get("mandarin_headword"),
                "paired_hits": row.get("paired_hits", 0),
                "wz_hits": row.get("wz_hits", 0),
                "zh_hits": row.get("zh_hits", 0),
                "usable_as_slot": row.get("usable_as_slot", False),
                "source_type": row.get("source_type", "example_mined"),
                "primary_topic_scene": scene["primary_scene_id"],
                "topic_scene_tags": scene["scene_tags"],
                "topic_scene_scores": scene["scene_scores"],
                "topic_scene_confidence": scene["primary_scene_confidence"],
                "primary_speech_function": speech["primary_speech_function"],
                "speech_function_tags": speech["speech_function_tags"],
                "speech_function_scores": speech["speech_function_scores"],
                "speech_function_confidence": speech["speech_function_confidence"],
                "evidence_examples": examples[:3],
                "llm_scene_review": row.get("llm_scene_review"),
                "safe_for_generation": row.get("safe_for_generation"),
            }
        )
    return enriched


def summarize(
    lexicon_rows: list[dict[str, Any]],
    skeleton_rows: list[dict[str, Any]],
    external_payload: dict[str, list[dict[str, Any]]],
    example_mined_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "lexicon_rows": len(lexicon_rows),
        "lexicon_scene_counts": dict(Counter(row["primary_topic_scene"] for row in lexicon_rows)),
        "lexicon_speech_counts": dict(Counter(row["primary_speech_function"] for row in lexicon_rows)),
        "skeleton_rows": len(skeleton_rows),
        "skeleton_scene_counts": dict(Counter(row["primary_topic_scene"] for row in skeleton_rows)),
        "skeleton_speech_counts": dict(Counter(row["primary_speech_function"] for row in skeleton_rows)),
        "new_nouns": len(external_payload.get("new_nouns") or []),
        "new_noun_scene_counts": dict(Counter(row["primary_topic_scene"] for row in external_payload.get("new_nouns") or [])),
        "place_names": len(external_payload.get("place_names") or []),
        "place_name_scene_counts": dict(Counter(row["primary_topic_scene"] for row in external_payload.get("place_names") or [])),
        "example_mined_terms": len(example_mined_rows),
        "example_mined_scene_counts": dict(Counter(row["primary_topic_scene"] for row in example_mined_rows)),
        "example_mined_speech_counts": dict(Counter(row["primary_speech_function"] for row in example_mined_rows)),
        "low_confidence_example_mined": sum(1 for row in example_mined_rows if row["topic_scene_confidence"] < 0.6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build explicit topic_scene and speech_function tag files for controlled-generation assets.")
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    parser.add_argument("--skeletons", type=Path, default=DEFAULT_SKELETONS)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--example-mined", type=Path, default=DEFAULT_EXAMPLE_MINED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    lexicon_rows = enrich_lexicon(load_jsonl(args.lexicon))
    skeleton_rows = enrich_skeletons(load_jsonl(args.skeletons))
    external_payload = enrich_external(json.loads(args.external.read_text(encoding="utf-8")))
    example_mined_rows = enrich_example_mined(load_jsonl(args.example_mined))

    output_dir: Path = args.output_dir
    write_jsonl(output_dir / "lexicon_scene_tags.jsonl", lexicon_rows)
    write_jsonl(output_dir / "skeleton_scene_tags.jsonl", skeleton_rows)
    write_json(output_dir / "external_term_scene_tags.json", external_payload)
    write_jsonl(output_dir / "example_mined_scene_tags.jsonl", example_mined_rows)
    write_json(
        output_dir / "tagging_summary.json",
        summarize(lexicon_rows, skeleton_rows, external_payload, example_mined_rows),
    )

    print(f"Lexicon tags: {len(lexicon_rows)}")
    print(f"Skeleton tags: {len(skeleton_rows)}")
    print(f"Example-mined tags: {len(example_mined_rows)}")
    print(f"New noun tags: {len(external_payload.get('new_nouns') or [])}")
    print(f"Place name tags: {len(external_payload.get('place_names') or [])}")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
