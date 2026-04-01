#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_controlled_generation_assets import ROOT
from generate_controlled_sentences import build_client
from relation_rerank_compound_candidates import build_prompt, normalize_result


DEFAULT_INPUT = ROOT / "data" / "controlled_generation" / "long_sentence" / "api_reranked" / "compound_candidates_relation_batch_diverse.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "controlled_generation" / "overnight_runs" / "compound_relation"

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


def unique_pass_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        sent = row["compound_wz_sentence"]
        if sent in seen:
            continue
        seen.add(sent)
        unique.append(row)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a long batched relation-rerank loop overnight with checkpointing.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--provider", choices=["primary", "secondary", "qwen", "deepseek"], default="deepseek")
    parser.add_argument("--hours", type=float, default=7.5)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = group_round_robin(load_jsonl(args.input))
    client, model, provider = build_client(args.provider)
    reviewed_rows: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []
    fail_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    batch_summaries: list[dict[str, Any]] = []

    deadline = time.time() + args.hours * 3600
    index = 0
    batch_index = 0
    total_rows = len(rows)

    while time.time() < deadline and index < total_rows:
        batch = rows[index : index + args.batch_size]
        if not batch:
            break
        batch_index += 1
        batch_passes = 0
        batch_errors = 0

        for row in batch:
            last_error = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    result = request_review(client, model, row)
                    merged = dict(row)
                    merged.update(normalize_result(result))
                    merged["batch_index"] = batch_index
                    merged["attempt_count"] = attempt
                    reviewed_rows.append(merged)
                    if merged["api_relation_pass"]:
                        pass_rows.append(merged)
                        batch_passes += 1
                    else:
                        fail_rows.append(merged)
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    time.sleep(min(2 * attempt, 8))
            if last_error is not None:
                error_rows.append(
                    {
                        "compound_id": row["compound_id"],
                        "scene_id": row["scene_id"],
                        "compound_wz_sentence": row["compound_wz_sentence"],
                        "error": last_error,
                        "batch_index": batch_index,
                    }
                )
                batch_errors += 1

        index += len(batch)
        unique_pass = unique_pass_rows(pass_rows)
        batch_summary = {
            "batch_index": batch_index,
            "reviewed_rows": len(batch),
            "batch_pass_rows": batch_passes,
            "batch_error_rows": batch_errors,
            "cumulative_unique_pass_rows": len(unique_pass),
            "scene_counts": dict(Counter(row["scene_id"] for row in unique_pass)),
            "relation_type_counts": dict(Counter(row["api_relation_type"] for row in unique_pass)),
        }
        batch_summaries.append(batch_summary)

        write_jsonl(run_dir / "reviewed_rows.jsonl", reviewed_rows)
        write_jsonl(run_dir / "pass_rows_unique.jsonl", unique_pass)
        write_jsonl(run_dir / "error_rows.jsonl", error_rows)
        write_json(
            run_dir / "progress_summary.json",
            {
                "run_id": run_id,
                "provider": provider,
                "model": model,
                "input_rows": total_rows,
                "reviewed_rows": len(reviewed_rows),
                "pass_rows_total": len(pass_rows),
                "pass_rows_unique": len(unique_pass),
                "fail_rows": len(fail_rows),
                "error_rows": len(error_rows),
                "scene_counts": dict(Counter(row["scene_id"] for row in unique_pass)),
                "relation_type_counts": dict(Counter(row["api_relation_type"] for row in unique_pass)),
                "batch_summaries": batch_summaries,
                "completed_all_input": index >= total_rows,
                "deadline_reached": time.time() >= deadline,
            },
        )
        time.sleep(args.sleep_seconds)

    unique_pass = unique_pass_rows(pass_rows)
    write_json(
        run_dir / "final_summary.json",
        {
            "run_id": run_id,
            "provider": provider,
            "model": model,
            "input_rows": total_rows,
            "reviewed_rows": len(reviewed_rows),
            "pass_rows_total": len(pass_rows),
            "pass_rows_unique": len(unique_pass),
            "fail_rows": len(fail_rows),
            "error_rows": len(error_rows),
            "scene_counts": dict(Counter(row["scene_id"] for row in unique_pass)),
            "relation_type_counts": dict(Counter(row["api_relation_type"] for row in unique_pass)),
            "batch_summaries": batch_summaries,
            "completed_all_input": index >= total_rows,
            "deadline_reached": time.time() >= deadline,
            "run_dir": str(run_dir),
        },
    )

    print(f"Run dir: {run_dir}")
    print(f"Reviewed rows: {len(reviewed_rows)}")
    print(f"Unique pass rows: {len(unique_pass)}")
    print(f"Error rows: {len(error_rows)}")
    print(f"Completed all input: {index >= total_rows}")


if __name__ == "__main__":
    main()
