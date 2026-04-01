#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "controlled_generation" / "long_sentence" / "self_checked" / "compound_candidates_accepted.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "controlled_generation" / "long_sentence" / "api_reranked" / "compound_candidates_relation_batch_diverse.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "controlled_generation" / "reports" / "compound_relation_batch_diverse_summary.json"

PRIORITY_SCENES = [
    "home_life",
    "transport_trip",
    "food_dining",
    "shopping_payment",
    "weather_safety",
    "health_medical",
    "work_study",
    "digital_chat",
]


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


def rank_key(row: dict[str, Any]) -> tuple[float, int, int, int, str]:
    return (
        row.get("self_check_score", 0.0),
        len(row.get("strong_shared_slot_terms_wz") or []),
        len(row.get("strong_shared_slot_terms_zh") or []),
        len(row.get("strong_shared_semantic_classes") or []),
        row["compound_id"],
    )


def is_scene_worthy(row: dict[str, Any]) -> bool:
    if row.get("self_check_score", 0.0) < 0.75:
        return False
    if row["scene_id"] == "daily_chat":
        return False
    return bool(
        row.get("strong_shared_slot_terms_wz")
        or row.get("strong_shared_slot_terms_zh")
        or row.get("strong_shared_semantic_classes")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a scene-stratified relation-rerank batch for compound long sentences.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--per-scene", type=int, default=12)
    parser.add_argument("--backfill", type=int, default=24)
    args = parser.parse_args()

    rows = [row for row in load_jsonl(args.input) if is_scene_worthy(row)]
    scene_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene_buckets[row["scene_id"]].append(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    scene_counts: Counter[str] = Counter()

    for scene_id in PRIORITY_SCENES:
        bucket = sorted(scene_buckets.get(scene_id, []), key=rank_key, reverse=True)
        for row in bucket:
            if scene_counts[scene_id] >= args.per_scene:
                break
            selected.append(row)
            selected_ids.add(row["compound_id"])
            scene_counts[scene_id] += 1

    leftovers = sorted(rows, key=rank_key, reverse=True)
    for row in leftovers:
        if len(selected) >= sum(scene_counts.values()) + args.backfill:
            break
        if row["compound_id"] in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(row["compound_id"])
        scene_counts[row["scene_id"]] += 1

    selected.sort(key=lambda row: (row["scene_id"], -row.get("self_check_score", 0.0), row["compound_id"]))
    write_jsonl(args.output, selected)
    write_json(
        args.summary,
        {
            "input_rows": len(rows),
            "selected_rows": len(selected),
            "scene_counts": dict(Counter(row["scene_id"] for row in selected)),
            "speech_function_counts": dict(Counter(row["primary_speech_function"] for row in selected)),
            "output_path": str(args.output),
        },
    )

    print(f"Input worthy rows: {len(rows)}")
    print(f"Selected rows: {len(selected)}")
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
