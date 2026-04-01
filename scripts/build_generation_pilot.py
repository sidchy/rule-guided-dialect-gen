#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "controlled_generation" / "tasks" / "generation_tasks_full.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "controlled_generation" / "tasks" / "generation_tasks_pilot_v2.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "controlled_generation" / "tasks" / "generation_tasks_pilot_v2_summary.json"


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


def external_types(row: dict[str, Any]) -> set[str]:
    return {item["type"] for item in row.get("external_constraints") or []}


def rank_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    ext = external_types(row)
    return (
        len(ext),
        row.get("slot_count", len(row.get("slots") or [])),
        len(row.get("required_words") or []),
        row["task_id"],
    )


def choose_rows(
    rows: list[dict[str, Any]],
    selected_ids: set[str],
    count: int,
    scene_counter: Counter[str],
    max_per_scene: int,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for row in sorted(rows, key=rank_key, reverse=True):
        if row["task_id"] in selected_ids:
            continue
        if scene_counter[row["scene_id"]] >= max_per_scene:
            continue
        chosen.append(row)
        selected_ids.add(row["task_id"])
        scene_counter[row["scene_id"]] += 1
        if len(chosen) >= count:
            break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small diverse pilot task set for controlled generation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--scene-quota", type=int, default=1)
    parser.add_argument("--extra-count", type=int, default=9)
    parser.add_argument("--max-per-scene", type=int, default=3)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    scene_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    type_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    plain_rows: list[dict[str, Any]] = []

    for row in rows:
        scene_buckets[row["scene_id"]].append(row)
        ext = external_types(row)
        if not ext:
            plain_rows.append(row)
        for label in ext:
            type_buckets[label].append(row)

    selected_ids: set[str] = set()
    scene_counter: Counter[str] = Counter()
    pilot_rows: list[dict[str, Any]] = []

    for scene_id in sorted(scene_buckets):
        pilot_rows.extend(
            choose_rows(
                scene_buckets[scene_id],
                selected_ids,
                args.scene_quota,
                scene_counter,
                args.max_per_scene,
            )
        )

    for label in ["example_mined", "new_noun", "place_name"]:
        pilot_rows.extend(
            choose_rows(
                type_buckets.get(label, []),
                selected_ids,
                2,
                scene_counter,
                args.max_per_scene,
            )
        )

    pilot_rows.extend(
        choose_rows(
            plain_rows,
            selected_ids,
            max(args.extra_count, 0),
            scene_counter,
            args.max_per_scene,
        )
    )
    pilot_rows.sort(key=lambda row: (row["scene_id"], row["task_id"]))

    write_jsonl(args.output, pilot_rows)
    write_json(
        args.summary,
        {
            "input_rows": len(rows),
            "pilot_rows": len(pilot_rows),
            "scene_counts": dict(Counter(row["scene_id"] for row in pilot_rows)),
            "external_type_counts": dict(
                Counter(label for row in pilot_rows for label in external_types(row))
            ),
            "output_path": str(args.output),
        },
    )

    print(f"Input rows: {len(rows)}")
    print(f"Pilot rows: {len(pilot_rows)}")
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
