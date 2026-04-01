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


DEFAULT_INPUT = (
    ROOT
    / "data"
    / "controlled_generation"
    / "skeletons"
    / "dense_reusable_rule_candidates.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "controlled_generation"
    / "skeletons"
    / "dense_reusable_llm_reviewed.jsonl"
)
DEFAULT_SUMMARY = (
    ROOT
    / "data"
    / "controlled_generation"
    / "reports"
    / "dense_reusable_llm_review_summary.json"
)

VALID_TIERS = {"dense_reusable", "medium_reusable", "single_use"}
VALID_ISSUES = {
    "event_bound",
    "idiom_like",
    "slot_not_content",
    "semantic_frame_fragile",
    "daily_naturalness_risk",
    "too_few_rewriteable_slots",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def rank_key(row: dict[str, Any]) -> tuple[float, float, int, str]:
    return (
        float(row.get("dense_reuse_rule_score", 0.0)),
        float(row.get("skeleton_quality_score", 0.0)),
        int(row.get("dense_combination_upper_bound", 0)),
        row.get("skeleton_id", ""),
    )


def build_prompt(row: dict[str, Any]) -> tuple[str, str]:
    system = (
        "你是温州话受控生成骨架评审员。"
        "你不改写句子，只判断这条骨架是否适合做高密度实词替换。"
        "所谓高密度实词替换，是指保留固定块、虚词、语气和句法框架，只替换可替换内容槽位。"
        "你必须重点判断："
        "1. 这条句型是否足够抽象、可复用；"
        "2. 多个内容槽位同时替换后，语义框架会不会塌；"
        "3. 哪些槽位是真正的内容槽位，哪些虽然抽出来了但不该换。"
        "只有在至少两个槽位都是通用内容槽位，而且同时替换后句型仍像日常自然句时，才能判 dense_reusable。"
        "如果句子强绑定某个具体事件、俗语、婚宴、固定说法、亲属称呼格式，或者槽位像代词/时间/元说明词，必须从严降级。"
        "如果不确定，优先判 medium_reusable 或 single_use，不要轻易判 dense_reusable。"
        "不要发明新槽位，不要改写骨架。"
        "eligible_slot_ids 和 locked_slot_ids 只能从输入 slot_id 里选。"
        "只输出 JSON object。"
    )
    payload = {
        "scene_id": row.get("scene_id"),
        "scene_label": row.get("scene_label"),
        "speech_function": row.get("primary_speech_function"),
        "source_wz_sentence": row.get("source_wz_sentence"),
        "source_zh_sentence": row.get("source_zh_sentence"),
        "skeleton_template": row.get("skeleton_template"),
        "fixed_chunks": row.get("fixed_chunks") or [],
        "dense_reuse_rule_score": row.get("dense_reuse_rule_score"),
        "dense_combination_upper_bound": row.get("dense_combination_upper_bound"),
        "rule_dense_eligible_slot_ids": row.get("dense_eligible_slot_ids") or [],
        "slots": [
            {
                "slot_id": slot.get("slot_id"),
                "surface_wz": slot.get("surface_wz"),
                "surface_zh": slot.get("surface_zh"),
                "slot_kind": slot.get("slot_kind"),
                "semantic_class": slot.get("semantic_class"),
                "dense_replaceable_rule": slot.get("dense_replaceable_rule"),
                "dense_priority": slot.get("dense_priority"),
                "dense_lock_reasons": slot.get("dense_lock_reasons") or [],
                "dense_candidate_pool_total": slot.get("dense_candidate_pool_total", 0),
                "dense_candidate_pool_preview": slot.get("dense_candidate_pool_preview") or [],
            }
            for slot in row.get("slots") or []
        ],
        "output_schema": {
            "pass": "boolean",
            "dense_tier": "dense_reusable | medium_reusable | single_use",
            "eligible_slot_ids": ["subset of input slot_id"],
            "locked_slot_ids": ["subset of input slot_id"],
            "issue_types": [
                "event_bound | idiom_like | slot_not_content | semantic_frame_fragile | daily_naturalness_risk | too_few_rewriteable_slots"
            ],
            "note": "short Chinese note under 40 chars",
        },
    }
    user = "根据下面 JSON 做骨架复用性判定，只返回 JSON：\n" + json.dumps(
        payload, ensure_ascii=False, indent=2
    )
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


def normalize_slot_ids(values: Any, valid_slot_ids: set[str]) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for item in values:
        slot_id = str(item or "").strip()
        if slot_id in valid_slot_ids and slot_id not in seen:
            normalized.append(slot_id)
            seen.add(slot_id)
    return normalized


def normalize_result(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    valid_slot_ids = {slot.get("slot_id") for slot in row.get("slots") or [] if slot.get("slot_id")}
    slot_map = {slot.get("slot_id"): slot for slot in row.get("slots") or [] if slot.get("slot_id")}
    dense_tier = str(result.get("dense_tier") or "single_use").strip() or "single_use"
    if dense_tier not in VALID_TIERS:
        dense_tier = "single_use"
    eligible_slot_ids = normalize_slot_ids(result.get("eligible_slot_ids"), valid_slot_ids)
    locked_slot_ids = normalize_slot_ids(result.get("locked_slot_ids"), valid_slot_ids)
    if not locked_slot_ids:
        locked_slot_ids = sorted(valid_slot_ids - set(eligible_slot_ids))
    issue_types = result.get("issue_types") or []
    if not isinstance(issue_types, list):
        issue_types = []
    issue_types = [str(item) for item in issue_types if str(item) in VALID_ISSUES][:5]
    llm_pass = bool(result.get("pass", False))
    if dense_tier == "single_use":
        llm_pass = False
    if dense_tier == "dense_reusable" and len(eligible_slot_ids) < 2:
        dense_tier = "medium_reusable"
        llm_pass = False
        issue_types = list(dict.fromkeys(issue_types + ["too_few_rewriteable_slots"]))[:5]
    hard_locked_selected = any(
        any(
            reason in {"pronoun_like", "meta_headword_like", "slot_kind_function", "slot_kind_time", "semantic_class_time"}
            for reason in (slot_map.get(slot_id, {}).get("dense_lock_reasons") or [])
        )
        for slot_id in eligible_slot_ids
    )
    if hard_locked_selected:
        dense_tier = "single_use"
        llm_pass = False
        issue_types = list(dict.fromkeys(issue_types + ["slot_not_content"]))[:5]
    if dense_tier == "dense_reusable" and int(row.get("dense_combination_upper_bound", 0)) < 30:
        dense_tier = "medium_reusable"
        llm_pass = False
    return {
        "llm_dense_pass": llm_pass,
        "llm_dense_tier": dense_tier,
        "llm_dense_eligible_slot_ids": eligible_slot_ids,
        "llm_dense_locked_slot_ids": locked_slot_ids,
        "llm_dense_issue_types": issue_types,
        "llm_dense_note": str(result.get("note") or "").strip()[:40],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use an LLM to review whether skeletons are suitable for dense slot replacement."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--provider", choices=["primary", "secondary", "qwen", "deepseek"], default="deepseek")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--min-slot-count", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = sorted(load_jsonl(args.input), key=rank_key, reverse=True)
    if args.min_slot_count > 0:
        rows = [row for row in rows if int(row.get("slot_count", 0)) >= args.min_slot_count]
    if args.limit > 0:
        rows = rows[: args.limit]

    reviewed_rows: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if args.resume and args.output.exists():
        reviewed_rows = load_jsonl(args.output)
        completed_ids = {row.get("skeleton_id") for row in reviewed_rows}

    client, model, provider = build_client(args.provider)
    for row in rows:
        if row.get("skeleton_id") in completed_ids:
            continue
        result = request_review(client, model, row)
        merged = dict(row)
        merged.update(normalize_result(row, result))
        reviewed_rows.append(merged)
        completed_ids.add(row.get("skeleton_id"))

    reviewed_rows.sort(key=rank_key, reverse=True)
    write_jsonl(args.output, reviewed_rows)
    write_json(
        args.summary,
        {
            "input_rows": len(rows),
            "reviewed_rows": len(reviewed_rows),
            "llm_dense_pass_rows": sum(1 for row in reviewed_rows if row.get("llm_dense_pass")),
            "llm_dense_tier_counts": dict(Counter(row.get("llm_dense_tier") for row in reviewed_rows)),
            "issue_type_counts": dict(
                Counter(issue for row in reviewed_rows for issue in row.get("llm_dense_issue_types") or [])
            ),
            "provider": provider,
            "model": model,
            "output_path": str(args.output),
        },
    )

    print(f"Input rows: {len(rows)}")
    print(f"Reviewed rows: {len(reviewed_rows)}")
    print(f"LLM dense pass rows: {sum(1 for row in reviewed_rows if row.get('llm_dense_pass'))}")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
