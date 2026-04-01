#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "cleaned" / "cleaned_records_primary.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "llm_review" / "mandarin_headwords.jsonl"
DEFAULT_HEURISTIC = ROOT / "data" / "llm_review" / "mandarin_headwords_heuristic.jsonl"
SYSTEM_PROMPT = """你是温州话词典清洗助手。你的任务是根据温州话词条、普通话释义和例句，抽取一个适合训练“普通话词 -> 温州话词”任务的普通话词头。

严格要求：
1. 只输出 JSON，不要输出解释。
2. 如果释义里没有稳定、自然、简短的普通话词头，就返回 null。
3. 不要硬造词头，不要扩写，不要发挥。
4. `mandarin_headword` 应尽量是 2 到 8 个汉字的词或短语，不要是一整句解释。
5. 如果条目明显是固定表达、俗语、上下句、整句而不是单词短语，`is_good_word_pair` 应设为 false。
6. `confidence` 取 0 到 1 之间的小数。

返回 JSON schema：
{
  "mandarin_headword": string | null,
  "definition_clean": string | null,
  "is_fixed_expression": boolean,
  "is_good_word_pair": boolean,
  "confidence": number,
  "notes": string
}
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_user_prompt(record: dict[str, Any]) -> str:
    payload = {
        "record_type": record.get("record_type"),
        "source_file": record.get("source_file"),
        "wz_word": record.get("wz_word_train"),
        "definition": record.get("definition_norm"),
        "example_wz": record.get("example_wz_train"),
        "example_zh": record.get("example_zh_train"),
        "notes": record.get("notes_raw"),
    }
    return "请抽取普通话词头并返回 JSON：\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_client() -> tuple[OpenAI, str]:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY_2")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL")
    if not api_key:
        raise RuntimeError("No API key found in .env")
    if not model:
        raise RuntimeError("OPENAI_MODEL is required in .env")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


def request_extraction(client: OpenAI, model: str, record: dict[str, Any]) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(record)},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def iter_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in records:
        if not record.get("is_exact_primary"):
            continue
        if not record.get("v1_candidate"):
            continue
        if record.get("record_type") not in {"dictionary_entry", "lexicon_entry", "fixed_expression"}:
            continue
        if not record.get("wz_word_train"):
            continue
        if not record.get("definition_norm"):
            continue
        candidates.append(record)
    return candidates


def load_heuristic_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    rows = load_jsonl(path)
    return {row["dedup_exact_key"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Mandarin headwords from cleaned records with an API model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Primary cleaned JSONL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of candidates to process.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file.")
    parser.add_argument(
        "--heuristic",
        type=Path,
        default=DEFAULT_HEURISTIC,
        help="Optional heuristic extraction JSONL used to prioritize low-confidence candidates.",
    )
    parser.add_argument(
        "--max-heuristic-confidence",
        type=float,
        default=0.67,
        help="Only call API when heuristic confidence is at or below this value, or when no headword exists.",
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    candidates = iter_candidates(records)
    heuristic_map = load_heuristic_map(args.heuristic)
    filtered_candidates: list[dict[str, Any]] = []
    for record in candidates:
        heuristic = heuristic_map.get(record["dedup_exact_key"])
        if not heuristic:
            filtered_candidates.append(record)
            continue
        extraction = heuristic.get("extraction", {})
        headword = extraction.get("mandarin_headword")
        confidence = extraction.get("confidence", 0.0) or 0.0
        if (not headword) or confidence <= args.max_heuristic_confidence:
            filtered_candidates.append(record)

    existing_keys: set[str] = set()
    output_rows: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        output_rows = load_jsonl(args.output)
        existing_keys = {row["dedup_exact_key"] for row in output_rows}

    client, model = build_client()

    processed = 0
    for record in filtered_candidates:
        if processed >= args.limit:
            break
        dedup_exact_key = record["dedup_exact_key"]
        if dedup_exact_key in existing_keys:
            continue
        result = request_extraction(client, model, record)
        row = {
            "dedup_exact_key": dedup_exact_key,
            "source_file": record["source_file"],
            "source_row_id": record["source_row_id"],
            "record_type": record["record_type"],
            "wz_word_train": record.get("wz_word_train"),
            "definition_norm": record.get("definition_norm"),
            "example_wz_train": record.get("example_wz_train"),
            "example_zh_train": record.get("example_zh_train"),
            "model": model,
            "extraction": result,
        }
        output_rows.append(row)
        existing_keys.add(dedup_exact_key)
        processed += 1
        if args.sleep > 0:
            time.sleep(args.sleep)

    write_jsonl(args.output, output_rows)
    print(f"Candidate pool: {len(candidates)}")
    print(f"Filtered API pool: {len(filtered_candidates)}")
    print(f"Processed this run: {processed}")
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
