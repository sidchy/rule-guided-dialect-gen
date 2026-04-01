#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generate_controlled_sentences import build_client
from openai import OpenAI

from build_controlled_generation_assets import ROOT, SCENE_CATALOG


DEFAULT_INPUT = ROOT / "data" / "controlled_generation" / "assets" / "example_mined_lexicon.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "controlled_generation" / "assets" / "example_mined_lexicon_reviewed.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "controlled_generation" / "tags" / "example_mined_scene_review_summary.json"


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


def build_scene_guide() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for scene in SCENE_CATALOG:
        rows.append(
            {
                "scene_id": scene["scene_id"],
                "label": scene["label"],
                "intents": " / ".join(scene["intents"]),
            }
        )
    return rows


def build_prompt(row: dict[str, Any]) -> tuple[str, str]:
    system = (
        "你是温州话语料场景标注员。"
        "任务是给一个从温州话例句中挖出的高频生活词重新判断 topic_scene。"
        "只能从给定 scene_id 列表里选。"
        "如果证据不足或跨场景太强，就降低 confidence，safe_for_generation 设为 false。"
        "输出必须是 JSON object。"
    )
    payload = {
        "allowed_scenes": build_scene_guide(),
        "term": row.get("term"),
        "current_primary_topic_scene": row.get("primary_topic_scene"),
        "current_topic_scene_tags": row.get("topic_scene_tags"),
        "current_topic_scene_confidence": row.get("topic_scene_confidence"),
        "evidence_examples": row.get("evidence_examples") or [],
        "paired_hits": row.get("paired_hits", 0),
        "wz_hits": row.get("wz_hits", 0),
        "zh_hits": row.get("zh_hits", 0),
        "output_schema": {
            "primary_topic_scene": "allowed scene_id",
            "topic_scene_tags": ["up to 3 allowed scene_id"],
            "confidence": "0.0-1.0 float",
            "safe_for_generation": "boolean",
            "usage_note": "short Chinese note under 20 chars",
        },
    }
    user = "根据下面 JSON 做重标注，只返回 JSON：\n" + json.dumps(payload, ensure_ascii=False, indent=2)
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


def normalize_review(result: dict[str, Any], fallback_scene: str) -> dict[str, Any]:
    allowed = {scene["scene_id"] for scene in SCENE_CATALOG}
    primary = result.get("primary_topic_scene")
    if primary not in allowed:
        primary = fallback_scene
    tags = [tag for tag in (result.get("topic_scene_tags") or []) if tag in allowed]
    if primary not in tags:
        tags = [primary] + tags
    tags = tags[:3]
    confidence_raw = result.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)
    return {
        "primary_topic_scene": primary,
        "topic_scene_tags": tags,
        "topic_scene_confidence": round(confidence, 4),
        "safe_for_generation": bool(result.get("safe_for_generation", False)),
        "usage_note": str(result.get("usage_note") or "").strip()[:20],
    }


def build_scene_scores(tags: list[str]) -> dict[str, int]:
    return {tag: max(3 - idx, 1) for idx, tag in enumerate(tags[:3])}


def merge_review(row: dict[str, Any], review: dict[str, Any], provider: str, model: str) -> dict[str, Any]:
    merged = dict(row)
    scene_scores = build_scene_scores(review["topic_scene_tags"])
    merged.update(
        {
            "scene_tags": review["topic_scene_tags"],
            "scene_scores": scene_scores,
            "primary_scene_id": review["primary_topic_scene"],
            "primary_scene_confidence": review["topic_scene_confidence"],
            "topic_scene_tags": review["topic_scene_tags"],
            "topic_scene_scores": scene_scores,
            "primary_topic_scene": review["primary_topic_scene"],
            "topic_scene_confidence": review["topic_scene_confidence"],
            "safe_for_generation": review["safe_for_generation"],
            "llm_scene_review": {
                "provider": provider,
                "model": model,
                "usage_note": review["usage_note"],
            },
        }
    )
    if not review["safe_for_generation"]:
        merged["usable_as_slot"] = False
    return merged


def load_existing_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    return {row["term"]: row for row in rows if row.get("term")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Use an LLM to relabel low-confidence example-mined scene tags.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--provider", choices=["primary", "secondary", "qwen", "deepseek"], default="primary")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    existing_map = load_existing_map(args.output) if args.resume else {}
    client, model, provider = build_client(args.provider)

    reviewed = 0
    reused = 0
    output_rows: list[dict[str, Any]] = []
    low_confidence_rows = [row for row in rows if row.get("topic_scene_confidence", 0.0) < args.threshold]
    low_confidence_terms = {row.get("term") for row in low_confidence_rows[: args.limit]}

    for row in rows:
        term = row.get("term")
        if term in low_confidence_terms:
            if term in existing_map:
                output_rows.append(existing_map[term])
                reused += 1
                continue
            result = request_review(client, model, row)
            normalized = normalize_review(result, fallback_scene=row.get("primary_topic_scene", "daily_chat"))
            output_rows.append(merge_review(row, normalized, provider, model))
            reviewed += 1
        elif term in existing_map:
            output_rows.append(existing_map[term])
        else:
            output_rows.append(row)

    write_jsonl(args.output, output_rows)
    write_json(
        args.summary,
        {
            "provider": provider,
            "model": model,
            "threshold": args.threshold,
            "input_rows": len(rows),
            "low_confidence_rows": len(low_confidence_rows),
            "reviewed_rows_this_run": reviewed,
            "reused_rows": reused,
            "output_rows": len(output_rows),
            "output_path": str(args.output),
        },
    )

    print(f"Input rows: {len(rows)}")
    print(f"Low-confidence rows: {len(low_confidence_rows)}")
    print(f"Reviewed this run: {reviewed}")
    print(f"Reused existing rows: {reused}")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
