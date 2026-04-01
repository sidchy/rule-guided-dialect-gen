from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import apply_contract
from .jsonl import append_jsonl, read_jsonl, write_jsonl


def build_promotion_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("rule_gate_status") == "pass" and row.get("human_review_status") == "pass"
    ]


def write_promotion_candidates(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def promote_generated_rows(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    policy_version: str,
) -> list[dict[str, Any]]:
    existing_ids = {
        str(existing.get("sample_id", "")).strip()
        for existing in read_jsonl(output_path)
        if str(existing.get("sample_id", "")).strip()
    }
    promoted: list[dict[str, Any]] = []
    for row in rows:
        promoted_row = apply_contract(
            dict(row),
            source_type="reviewed_curated",
            origin_run_id=str(row.get("origin_run_id", "")),
            pipeline_name=str(row.get("pipeline_name", "")),
            policy_version=policy_version,
            scene_id=str(row.get("scene_id", "")),
            lane=str(row.get("lane", "")),
            core_word=str(row.get("core_word", "")),
            support_words=row.get("support_words", []),
            sample_prefix="curated",
        )
        sample_id = str(promoted_row.get("sample_id", "")).strip()
        if sample_id and sample_id in existing_ids:
            continue
        if sample_id:
            existing_ids.add(sample_id)
        promoted.append(promoted_row)
    append_jsonl(output_path, promoted)
    return promoted
