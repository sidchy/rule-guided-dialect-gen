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


DEFAULT_INPUT = ROOT / "data" / "controlled_generation" / "filtered" / "pilot_v3_qwen" / "generation_candidates_accepted.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "controlled_generation" / "filtered" / "pilot_v3_qwen" / "generation_candidates_critic.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "controlled_generation" / "filtered" / "pilot_v3_qwen" / "critic_summary.json"


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


def build_prompt(row: dict[str, Any]) -> tuple[str, str]:
    system = (
        "你是温州话受控生成质检员。"
        "你只做质量判定，不改写。"
        "重点判断："
        "1. 句子像不像自然温州话口语；"
        "2. required_words 是否被硬塞；"
        "3. 是否和 scene_id 大体匹配；"
        "4. 是否适合后续语音训练。"
        "如果语义明显别扭，即使格式过关，也必须判 fail。"
        "只输出 JSON object。"
    )
    payload = {
        "task_type": row.get("task_type"),
        "scene_id": row.get("scene_id"),
        "source_wz_sentence": row.get("source_wz_sentence"),
        "skeleton_template": row.get("skeleton_template"),
        "required_words": row.get("required_words"),
        "preferred_words": row.get("preferred_words"),
        "candidate_sentence": row.get("sentence"),
        "eligible_slot_ids": row.get("eligible_slot_ids") or [],
        "required_slot_replacements": row.get("required_slot_replacements") or [],
        "slot_values_model": row.get("slot_values_model") or {},
        "eligible_slot_replacement_ratio": row.get("eligible_slot_replacement_ratio"),
        "pool_compliance_ratio": row.get("pool_compliance_ratio"),
        "slots": [
            {
                "slot_id": slot.get("slot_id"),
                "surface_wz": slot.get("surface_wz"),
                "surface_zh": slot.get("surface_zh"),
                "slot_kind": slot.get("slot_kind"),
                "semantic_class": slot.get("semantic_class"),
            }
            for slot in row.get("slots") or []
        ],
        "output_schema": {
            "pass": "boolean",
            "score": "0.0-1.0 float",
            "issue_types": ["semantic_mismatch | forced_insertion | scene_mismatch | unnatural_wz | speech_training_risk"],
            "note": "short Chinese note under 30 chars",
        },
    }
    user = "根据下面 JSON 做质量判定，只返回 JSON：\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    return system, user


def request_review(client: OpenAI, model: str, row: dict[str, Any], timeout: float | None) -> dict[str, Any]:
    system, user = build_prompt(row)
    request_client = client.with_options(timeout=timeout) if timeout and timeout > 0 else client
    response = request_client.chat.completions.create(
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
    return {
        "critic_pass": bool(result.get("pass", False)),
        "critic_score": round(score, 4),
        "critic_issue_types": issue_types,
        "critic_note": str(result.get("note") or "").strip()[:30],
    }


def build_error_result(error_message: str) -> dict[str, Any]:
    return {
        "critic_pass": False,
        "critic_score": 0.0,
        "critic_issue_types": ["critic_request_failed"],
        "critic_note": error_message[:30],
        "critic_request_failed": True,
        "critic_request_error": error_message,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Use an LLM critic to review accepted controlled-generation candidates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--provider", choices=["primary", "secondary", "qwen", "deepseek"], default="qwen")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    client, model, provider = build_client(args.provider)

    reviewed_rows: list[dict[str, Any]] = []
    failed = 0
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] reviewing {row.get('task_id')}#{row.get('candidate_rank')}", flush=True)
        try:
            result = request_review(client, model, row, timeout=args.timeout)
            normalized = normalize_result(result)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            error_message = f"{type(exc).__name__}: {exc}"
            print(f"[{index}/{len(rows)}] failed {row.get('task_id')}#{row.get('candidate_rank')}: {error_message}", flush=True)
            normalized = build_error_result(error_message)
            failed += 1
        merged = dict(row)
        merged.update(normalized)
        reviewed_rows.append(merged)
        write_jsonl(args.output, reviewed_rows)
        write_json(
            args.summary,
            {
                "input_rows": len(rows),
                "reviewed_rows": len(reviewed_rows),
                "critic_pass_rows": sum(1 for item in reviewed_rows if item["critic_pass"]),
                "critic_fail_rows": sum(1 for item in reviewed_rows if not item["critic_pass"]),
                "failed_request_rows": failed,
                "issue_type_counts": dict(Counter(issue for item in reviewed_rows for issue in item["critic_issue_types"])),
                "provider": provider,
                "model": model,
                "output_path": str(args.output),
            },
        )

    print(f"Input rows: {len(rows)}")
    print(f"Critic pass rows: {sum(1 for row in reviewed_rows if row['critic_pass'])}")
    print(f"Critic fail rows: {sum(1 for row in reviewed_rows if not row['critic_pass'])}")
    print(f"Failed request rows: {failed}")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
