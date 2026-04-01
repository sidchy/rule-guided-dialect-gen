#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from generate_controlled_sentences import build_client
from openai import OpenAI

from build_controlled_generation_assets import ROOT


DEFAULT_INPUT = ROOT / "data" / "controlled_generation" / "long_sentence" / "self_checked" / "compound_candidates_accepted.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "controlled_generation" / "long_sentence" / "api_reranked" / "compound_candidates_relation_reranked.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "controlled_generation" / "reports" / "compound_relation_rerank_summary.json"


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


def build_prompt(row: dict[str, Any]) -> tuple[str, str]:
    system = (
        "你是温州话长句关系判定员。"
        "你不改写，只判断两个分句拼成一个 20-30 字长句时，关系是否明确、自然。"
        "只有在两个分句之间存在清楚可解释的关系时才能判 pass。"
        "允许的关系类型只有：parallel_support, reminder_reason, cause_effect, plan_obstacle, evaluation_example, sequence, contrast。"
        "如果只是同场景、同主语、同词汇，但没有自然承接关系，必须判 fail。"
        "只输出 JSON object。"
    )
    payload = {
        "scene_id": row.get("scene_id"),
        "speech_function": row.get("primary_speech_function"),
        "compound_wz_sentence": row.get("compound_wz_sentence"),
        "self_check_score": row.get("self_check_score"),
        "strong_shared_slot_terms_wz": row.get("strong_shared_slot_terms_wz") or [],
        "strong_shared_slot_terms_zh": row.get("strong_shared_slot_terms_zh") or [],
        "strong_shared_semantic_classes": row.get("strong_shared_semantic_classes") or [],
        "parts": [
            {
                "wz_sentence": part.get("wz_sentence"),
                "zh_sentence": part.get("zh_sentence"),
                "slot_terms_wz": part.get("slot_terms_wz") or [],
                "slot_terms_zh": part.get("slot_terms_zh") or [],
                "slot_semantic_classes": part.get("slot_semantic_classes") or [],
            }
            for part in row.get("parts") or []
        ],
        "output_schema": {
            "pass": "boolean",
            "score": "0.0-1.0 float",
            "relation_type": "parallel_support | reminder_reason | cause_effect | plan_obstacle | evaluation_example | sequence | contrast | invalid",
            "issue_types": ["topic_jump | clause_relation_weak | unnatural_wz | duplicate_content | weak_bridge"],
            "note": "short Chinese note under 40 chars",
        },
    }
    user = "根据下面 JSON 做关系判定，只返回 JSON：\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    return system, user


def request_review(client: OpenAI, model: str, row: dict[str, Any]) -> dict[str, Any]:
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


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    raw_score = result.get("score", 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    score = min(max(score, 0.0), 1.0)
    issue_types = result.get("issue_types") or []
    if not isinstance(issue_types, list):
        issue_types = []
    issue_types = [str(item) for item in issue_types][:5]
    relation_type = str(result.get("relation_type") or "invalid").strip() or "invalid"
    return {
        "api_relation_pass": bool(result.get("pass", False)),
        "api_relation_score": round(score, 4),
        "api_relation_type": relation_type,
        "api_relation_issue_types": issue_types,
        "api_relation_note": str(result.get("note") or "").strip()[:40],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Use a low-cost API model to relation-rerank compound long sentence candidates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--provider", choices=["primary", "secondary", "qwen", "deepseek"], default="deepseek")
    parser.add_argument("--limit", type=int, default=48)
    args = parser.parse_args()

    rows = sorted(load_jsonl(args.input), key=rank_key, reverse=True)
    if args.limit > 0:
        rows = rows[: args.limit]

    client, model, provider = build_client(args.provider)
    reviewed_rows: list[dict[str, Any]] = []
    for row in rows:
        result = request_review(client, model, row)
        merged = dict(row)
        merged.update(normalize_result(result))
        reviewed_rows.append(merged)

    write_jsonl(args.output, reviewed_rows)
    write_json(
        args.summary,
        {
            "input_rows": len(rows),
            "api_relation_pass_rows": sum(1 for row in reviewed_rows if row["api_relation_pass"]),
            "api_relation_fail_rows": sum(1 for row in reviewed_rows if not row["api_relation_pass"]),
            "relation_type_counts": dict(Counter(row["api_relation_type"] for row in reviewed_rows)),
            "issue_type_counts": dict(Counter(issue for row in reviewed_rows for issue in row["api_relation_issue_types"])),
            "provider": provider,
            "model": model,
            "output_path": str(args.output),
        },
    )

    print(f"Input rows: {len(rows)}")
    print(f"API relation pass rows: {sum(1 for row in reviewed_rows if row['api_relation_pass'])}")
    print(f"API relation fail rows: {sum(1 for row in reviewed_rows if not row['api_relation_pass'])}")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
