#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "data"
    / "controlled_generation"
    / "overnight_runs"
    / "compound_relation"
    / "20260331_002931"
    / "pass_rows_unique.jsonl"
)
DEFAULT_OUTPUT_CSV = (
    ROOT
    / "data"
    / "controlled_generation"
    / "reports"
    / "compound_relation_pass_table_20260331.csv"
)
DEFAULT_OUTPUT_MD = (
    ROOT
    / "data"
    / "controlled_generation"
    / "reports"
    / "compound_relation_pass_table_20260331_preview.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export passed compound sentence rows into a flat table."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=20,
        help="How many rows to include in the markdown preview.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def join_list(values: list[Any]) -> str:
    return " | ".join(str(item) for item in values if str(item).strip())


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    parts = row.get("parts", [])
    part1 = parts[0] if len(parts) > 0 else {}
    part2 = parts[1] if len(parts) > 1 else {}

    return {
        "compound_id": row.get("compound_id", ""),
        "scene_id": row.get("scene_id", ""),
        "scene_label": row.get("scene_label", ""),
        "speech_function": row.get("primary_speech_function", ""),
        "relation_type": row.get("api_relation_type", ""),
        "relation_score": row.get("api_relation_score", ""),
        "self_check_score": row.get("self_check_score", ""),
        "compound_length": row.get("compound_length", ""),
        "compound_wz_sentence": row.get("compound_wz_sentence", ""),
        "relation_note": row.get("api_relation_note", ""),
        "shared_slot_terms_wz": join_list(row.get("shared_slot_terms_wz", [])),
        "shared_slot_terms_zh": join_list(row.get("shared_slot_terms_zh", [])),
        "strong_shared_slot_terms_wz": join_list(
            row.get("strong_shared_slot_terms_wz", [])
        ),
        "strong_shared_slot_terms_zh": join_list(
            row.get("strong_shared_slot_terms_zh", [])
        ),
        "part_1_skeleton_id": part1.get("skeleton_id", ""),
        "part_1_source_file": part1.get("source_file", ""),
        "part_1_wz_sentence": part1.get("wz_sentence", ""),
        "part_1_zh_sentence": part1.get("zh_sentence", ""),
        "part_1_slot_terms_wz": join_list(part1.get("slot_terms_wz", [])),
        "part_1_slot_terms_zh": join_list(part1.get("slot_terms_zh", [])),
        "part_2_skeleton_id": part2.get("skeleton_id", ""),
        "part_2_source_file": part2.get("source_file", ""),
        "part_2_wz_sentence": part2.get("wz_sentence", ""),
        "part_2_zh_sentence": part2.get("zh_sentence", ""),
        "part_2_slot_terms_wz": join_list(part2.get("slot_terms_wz", [])),
        "part_2_slot_terms_zh": join_list(part2.get("slot_terms_zh", [])),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_preview(rows: list[dict[str, Any]], path: Path, preview_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        ("scene", "scene_label"),
        ("relation", "relation_type"),
        ("compound_wz_sentence", "compound_wz_sentence"),
        ("part_1_wz_sentence", "part_1_wz_sentence"),
        ("part_1_zh_sentence", "part_1_zh_sentence"),
        ("part_2_wz_sentence", "part_2_wz_sentence"),
        ("part_2_zh_sentence", "part_2_zh_sentence"),
    ]
    lines = ["# Compound Sentence Pass Table Preview", ""]
    lines.append("| " + " | ".join(label for label, _ in columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows[:preview_rows]:
        values = [
            str(row.get(field_name, "")).replace("\n", " ")
            for _, field_name in columns
        ]
        safe_values = [value.replace("|", "\\|") for value in values]
        lines.append("| " + " | ".join(safe_values) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_rows = load_jsonl(args.input)
    flat_rows = [flatten_row(row) for row in source_rows]
    write_csv(flat_rows, args.output_csv)
    write_preview(flat_rows, args.output_md, args.preview_rows)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output_csv": str(args.output_csv),
                "output_md": str(args.output_md),
                "rows": len(flat_rows),
                "preview_rows": min(args.preview_rows, len(flat_rows)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
