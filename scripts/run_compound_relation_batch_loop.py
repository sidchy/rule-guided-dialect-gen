#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_controlled_generation_assets import ROOT
from generate_controlled_sentences import build_client
from relation_rerank_compound_candidates import build_prompt, normalize_result


DEFAULT_INPUT = ROOT / "data" / "controlled_generation" / "long_sentence" / "api_reranked" / "compound_candidates_relation_batch_diverse.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "controlled_generation" / "long_sentence" / "api_reranked" / "compound_candidates_relation_batch_loop_results.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "controlled_generation" / "reports" / "compound_relation_batch_loop_summary.json"

CORE_SCENES = ["home_life", "food_dining", "transport_trip", "shopping_payment", "weather_safety", "health_medical"]


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


def group_round_robin(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["scene_id"]].append(row)
    ordered: list[dict[str, Any]] = []
    while True:
        added = False
        for scene_id in CORE_SCENES + sorted(scene for scene in buckets if scene not in CORE_SCENES):
            bucket = buckets.get(scene_id) or []
            if not bucket:
                continue
            ordered.append(bucket.pop(0))
            added = True
        if not added:
            break
    return ordered


def chunked(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def request_review(client: Any, model: str, row: dict[str, Any]) -> dict[str, Any]:
    system, user = build_prompt(row)
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def threshold_met(pass_rows: list[dict[str, Any]], min_passes: int, min_scenes: int, min_core_scene_hits: int) -> bool:
    if len(pass_rows) < min_passes:
        return False
    scene_counts = Counter(row["scene_id"] for row in pass_rows)
    if len(scene_counts) < min_scenes:
        return False
    core_hits = sum(1 for scene in CORE_SCENES if scene_counts.get(scene, 0) > 0)
    return core_hits >= min_core_scene_hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated batched relation rerank over diverse compound candidates until threshold is met or input is exhausted.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--provider", choices=["primary", "secondary", "qwen", "deepseek"], default="deepseek")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--min-passes", type=int, default=12)
    parser.add_argument("--min-scenes", type=int, default=4)
    parser.add_argument("--min-core-scene-hits", type=int, default=3)
    args = parser.parse_args()

    rows = group_round_robin(load_jsonl(args.input))
    batches = chunked(rows, args.batch_size)[: args.max_batches]
    client, model, provider = build_client(args.provider)

    reviewed_rows: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []
    batch_summaries: list[dict[str, Any]] = []

    for batch_index, batch in enumerate(batches, start=1):
        batch_results: list[dict[str, Any]] = []
        for row in batch:
            result = request_review(client, model, row)
            merged = dict(row)
            merged.update(normalize_result(result))
            merged["batch_index"] = batch_index
            batch_results.append(merged)
            reviewed_rows.append(merged)
            if merged["api_relation_pass"]:
                pass_rows.append(merged)

        batch_summary = {
            "batch_index": batch_index,
            "input_rows": len(batch_results),
            "pass_rows": sum(1 for row in batch_results if row["api_relation_pass"]),
            "scene_counts": dict(Counter(row["scene_id"] for row in batch_results if row["api_relation_pass"])),
            "relation_type_counts": dict(Counter(row["api_relation_type"] for row in batch_results if row["api_relation_pass"])),
        }
        batch_summaries.append(batch_summary)
        if threshold_met(pass_rows, args.min_passes, args.min_scenes, args.min_core_scene_hits):
            break

    write_jsonl(args.output, reviewed_rows)
    write_json(
        args.summary,
        {
            "reviewed_rows": len(reviewed_rows),
            "pass_rows": len(pass_rows),
            "fail_rows": sum(1 for row in reviewed_rows if not row["api_relation_pass"]),
            "scene_counts": dict(Counter(row["scene_id"] for row in pass_rows)),
            "relation_type_counts": dict(Counter(row["api_relation_type"] for row in pass_rows)),
            "issue_type_counts": dict(Counter(issue for row in reviewed_rows for issue in row["api_relation_issue_types"])),
            "threshold_met": threshold_met(pass_rows, args.min_passes, args.min_scenes, args.min_core_scene_hits),
            "batch_summaries": batch_summaries,
            "provider": provider,
            "model": model,
            "output_path": str(args.output),
        },
    )

    print(f"Reviewed rows: {len(reviewed_rows)}")
    print(f"Pass rows: {len(pass_rows)}")
    print(f"Threshold met: {threshold_met(pass_rows, args.min_passes, args.min_scenes, args.min_core_scene_hits)}")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
