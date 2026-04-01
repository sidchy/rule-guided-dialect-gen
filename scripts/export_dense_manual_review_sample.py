#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from build_controlled_generation_assets import ROOT


DEFAULT_INPUT = (
    ROOT
    / "data"
    / "controlled_generation"
    / "filtered"
    / "generation_candidates_scored.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "controlled_generation"
    / "reports"
    / "dense_manual_review_sample.csv"
)
DEFAULT_CRITIC_INPUT = (
    ROOT
    / "data"
    / "controlled_generation"
    / "filtered"
    / "dense_pilot_v1"
    / "generation_candidates_critic.jsonl"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_pick(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows = [row for row in rows if row.get("task_type") == "dense_slot_replacement"]
    rows.sort(
        key=lambda row: (
            str(row.get("scene_id") or ""),
            int(not bool(row.get("accepted"))),
            -float(row.get("quality_score", 0.0)),
            str(row.get("task_id") or ""),
            int(row.get("candidate_rank") or 0),
        )
    )
    return rows[:limit]


def stable_pick_with_critic(
    scored_rows: list[dict[str, Any]],
    critic_rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if not critic_rows:
        return stable_pick(scored_rows, limit)
    critic_dense_rows = [row for row in critic_rows if row.get("task_type") == "dense_slot_replacement"]
    critic_dense_rows.sort(
        key=lambda row: (
            int(bool(row.get("critic_pass"))),
            str(row.get("scene_id") or ""),
            -float(row.get("critic_score", 0.0)),
            str(row.get("task_id") or ""),
            int(row.get("candidate_rank") or 0),
        ),
        reverse=True,
    )
    passes = [row for row in critic_dense_rows if row.get("critic_pass")]
    fails = [row for row in critic_dense_rows if not row.get("critic_pass")]
    if not passes:
        return fails[:limit]
    fail_quota = limit // 2
    pass_quota = limit - fail_quota
    return fails[:fail_quota] + passes[:pass_quota]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a fixed dense manual review sample CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--critic-input", type=Path, default=DEFAULT_CRITIC_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    rows = stable_pick_with_critic(
        load_jsonl(args.input),
        load_jsonl(args.critic_input),
        args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene_id",
        "task_id",
        "candidate_rank",
        "accepted",
        "quality_score",
        "critic_pass",
        "critic_score",
        "critic_issue_types",
        "critic_note",
        "dense_rewrite_pass",
        "eligible_slot_replacement_ratio",
        "pool_compliance_ratio",
        "source_wz_sentence",
        "sentence",
        "source_zh_sentence",
        "naturalness_note",
        "manual_keep",
        "manual_note",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    print(f"Input dense review rows: {len(rows)}")
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
