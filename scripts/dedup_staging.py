#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "staging" / "staging_records.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "cleaned"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_cluster_key(record: dict[str, Any]) -> str:
    record_type = record["record_type"]
    if record_type == "sentence_pair":
        return f"sentence::{record.get('wz_text_train') or ''}"
    return f"{record_type}::{record.get('wz_word_train') or ''}"


def make_exact_key(record: dict[str, Any]) -> str:
    record_type = record["record_type"]
    if record_type == "sentence_pair":
        wz = record.get("wz_text_train") or ""
        zh = record.get("zh_text_train") or ""
        return f"sentence::{wz}||{zh}"
    wz_word = record.get("wz_word_train") or ""
    definition = record.get("definition_norm") or ""
    ex_wz = record.get("example_wz_train") or ""
    ex_zh = record.get("example_zh_train") or ""
    return f"{record_type}::{wz_word}||{definition}||{ex_wz}||{ex_zh}"


def make_conflict_signature(record: dict[str, Any]) -> str:
    if record["record_type"] == "sentence_pair":
        return record.get("zh_text_train") or ""
    parts = [
        record.get("definition_norm") or "",
        record.get("example_wz_train") or "",
        record.get("example_zh_train") or "",
    ]
    return "||".join(parts)


def enrich_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cluster_counts = Counter()
    exact_counts = Counter()
    cluster_signatures: dict[str, set[str]] = {}

    for record in records:
        cluster_key = make_cluster_key(record)
        exact_key = make_exact_key(record)
        cluster_counts[cluster_key] += 1
        exact_counts[exact_key] += 1
        cluster_signatures.setdefault(cluster_key, set()).add(make_conflict_signature(record))

    seen_exact: set[str] = set()
    enriched: list[dict[str, Any]] = []
    for record in records:
        cluster_key = make_cluster_key(record)
        exact_key = make_exact_key(record)
        is_primary = exact_key not in seen_exact
        if is_primary:
            seen_exact.add(exact_key)
        conflict_flag = len(cluster_signatures[cluster_key]) > 1
        new_record = dict(record)
        new_record["dedup_cluster_key"] = cluster_key
        new_record["dedup_exact_key"] = exact_key
        new_record["cluster_size"] = cluster_counts[cluster_key]
        new_record["exact_duplicate_count"] = exact_counts[exact_key]
        new_record["is_exact_primary"] = is_primary
        new_record["is_exact_duplicate"] = not is_primary
        new_record["conflict_flag"] = conflict_flag
        new_record["v1_candidate"] = is_primary and not new_record["exclude_from_v1"]
        enriched.append(new_record)
    return enriched


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "total_records": len(records),
        "exact_primary_records": sum(1 for r in records if r["is_exact_primary"]),
        "exact_duplicates": sum(1 for r in records if r["is_exact_duplicate"]),
        "conflict_records": sum(1 for r in records if r["conflict_flag"]),
        "v1_candidates": sum(1 for r in records if r["v1_candidate"]),
        "record_type_counts": dict(Counter(r["record_type"] for r in records)),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate staging records and mark conflicts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to staging JSONL.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write cleaned outputs.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.input)
    enriched = enrich_records(records)

    primary_records = [r for r in enriched if r["is_exact_primary"]]
    review_records = [r for r in enriched if r["conflict_flag"] or r["exclude_from_v1"]]

    write_jsonl(output_dir / "cleaned_records_all.jsonl", enriched)
    write_jsonl(output_dir / "cleaned_records_primary.jsonl", primary_records)
    write_jsonl(output_dir / "review_candidates.jsonl", review_records)
    pd.DataFrame(primary_records).to_csv(
        output_dir / "cleaned_records_primary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "cleaned_summary.json").write_text(
        json.dumps(build_summary(enriched), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Loaded {len(records)} staging records")
    print(f"Primary records: {len(primary_records)}")
    print(f"Review candidates: {len(review_records)}")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
