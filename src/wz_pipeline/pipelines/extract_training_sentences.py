#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from wz_pipeline.contracts import apply_contract
from wz_pipeline.jsonl import write_jsonl
from wz_pipeline.registry import register_run
from wz_pipeline.review import export_review_tsv
from wz_pipeline.runs import build_summary, make_run_id, prepare_run_layout, write_json
from wz_pipeline.paths import ROOT


INPUT = ROOT / "data" / "cleaned" / "cleaned_records_primary.jsonl"
MIRROR_DIR = ROOT / "data" / "extracted_training_sentences"
POLICY_VERSION = "source_seed_v1_2026-04-01"
PIPELINE_NAME = "extract_training_sentences"

PAREN_ANNOTATION_RE = re.compile(r"[（(][^）)]{1,4}[）)]")
WHITESPACE_RE = re.compile(r"[\s\u3000]+")


def clean_wz(text: str) -> str:
    text = PAREN_ANNOTATION_RE.sub("", text)
    text = WHITESPACE_RE.sub("", text)
    text = text.lstrip("。，、；：\"'！？…-")
    return text.strip()


def clean_zh(text: str) -> str:
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text.rstrip("。").strip()


def sentence_id(wz: str) -> str:
    digest = hashlib.md5(wz.encode("utf-8")).hexdigest()[:12]
    return f"dict_ex_{digest}"


def write_summary(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract dictionary example sentences into source seed pool.")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--mirror-dir", type=Path, default=MIRROR_DIR)
    parser.add_argument("--run-id", type=str, default="")
    args = parser.parse_args()

    run_id = args.run_id or make_run_id(
        pipeline_name=PIPELINE_NAME,
        provider="source",
        model="dictionary",
        policy_version=POLICY_VERSION,
    )
    layout = prepare_run_layout(pipeline_name=PIPELINE_NAME, run_id=run_id)
    config = {
        "input": str(args.input),
        "mirror_dir": str(args.mirror_dir),
        "policy_version": POLICY_VERSION,
    }
    write_json(layout.config_path, config)

    rows_short: list[dict[str, object]] = []
    rows_long: list[dict[str, object]] = []
    rows_all: list[dict[str, object]] = []
    seen: set[str] = set()
    stats: Counter[str] = Counter()

    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            stats["total_records"] += 1

            ewt = str(row.get("example_wz_train") or "").strip()
            if not ewt:
                stats["no_example"] += 1
                continue

            wz = clean_wz(ewt)
            zh = clean_zh(str(row.get("example_zh_train") or ""))
            if not wz:
                stats["empty_after_clean"] += 1
                continue
            if wz in seen:
                stats["duplicate"] += 1
                continue
            seen.add(wz)

            char_len = len(wz)
            if char_len < 8:
                stats["too_short"] += 1
                continue
            if char_len > 30:
                stats["too_long"] += 1
                continue

            base_row = {
                "sentence_id": sentence_id(wz),
                "wz_sentence": wz,
                "zh_sentence": zh,
                "wz_word": row.get("wz_word_train", ""),
                "definition": str(row.get("definition_norm", "") or "")[:120],
                "char_len": char_len,
                "length_bucket": "short" if char_len <= 20 else "long",
                "record_type": row.get("record_type", ""),
                "source_file": row.get("source_file", ""),
                "source_row_id": row.get("source_row_id"),
            }
            contracted = apply_contract(
                base_row,
                source_type="extracted_seed",
                origin_run_id=run_id,
                pipeline_name=PIPELINE_NAME,
                policy_version=POLICY_VERSION,
                lane="mainline",
                core_word=str(base_row.get("wz_word", "")),
                support_words=[],
                sample_prefix="seed",
            )
            rows_all.append(contracted)
            if char_len <= 20:
                rows_short.append(contracted)
                stats["short_8_20"] += 1
            else:
                rows_long.append(contracted)
                stats["long_20_30"] += 1

    args.mirror_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.mirror_dir / "short_8_20.jsonl", rows_short)
    write_jsonl(args.mirror_dir / "long_20_30.jsonl", rows_long)
    write_jsonl(args.mirror_dir / "all_in_range.jsonl", rows_all)

    extraction_summary = {
        "input_file": str(args.input),
        "run_id": run_id,
        "policy_version": POLICY_VERSION,
        "total_records": stats["total_records"],
        "no_example": stats["no_example"],
        "empty_after_clean": stats["empty_after_clean"],
        "duplicate": stats["duplicate"],
        "too_short": stats["too_short"],
        "too_long": stats["too_long"],
        "short_8_20": stats["short_8_20"],
        "long_20_30": stats["long_20_30"],
        "total_in_range": stats["short_8_20"] + stats["long_20_30"],
        "has_zh_translation": sum(1 for row in rows_all if row.get("zh_sentence")),
    }
    write_summary(args.mirror_dir / "extraction_summary.json", extraction_summary)

    write_jsonl(layout.results_path, rows_all)
    write_jsonl(layout.rule_gate_path, rows_all)
    export_review_tsv(layout.review_tsv_path, [])
    write_jsonl(layout.promotion_candidates_path, [])

    summary = build_summary(
        pipeline_name=PIPELINE_NAME,
        run_id=run_id,
        config=config,
        raw_count=len(rows_all),
        rule_pass_count=0,
        extra={"mirror_outputs": str(args.mirror_dir), "rule_gate_applicable": False},
    )
    write_json(layout.summary_path, summary)
    register_run(
        {
            "pipeline_name": PIPELINE_NAME,
            "run_id": run_id,
            "config_path": str(layout.config_path),
            "results_path": str(layout.results_path),
            "rule_gate_path": str(layout.rule_gate_path),
            "review_tsv_path": str(layout.review_tsv_path),
            "summary_path": str(layout.summary_path),
            "policy_version": POLICY_VERSION,
        }
    )
    print(json.dumps(extraction_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

