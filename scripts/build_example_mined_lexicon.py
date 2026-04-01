#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from build_controlled_generation_assets import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RECORDS,
    build_scene_payload,
    build_speech_function_payload,
    write_summary,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "assets" / "example_mined_lexicon.jsonl"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "assets" / "example_mined_lexicon_summary.json"

SEED_TERMS = [
    "苹果",
    "香蕉",
    "橘子",
    "葡萄",
    "西瓜",
    "牛奶",
    "酸奶",
    "鸡蛋",
    "面包",
    "米饭",
    "奶茶",
    "咖啡",
    "冰箱",
    "空调",
    "风扇",
    "电脑",
    "笔记本",
    "平板",
    "手机",
    "耳机",
    "充电器",
    "充电宝",
    "电视",
    "遥控器",
    "洗衣机",
    "热水器",
    "电动车",
    "自行车",
    "公交",
    "地铁",
    "高铁",
    "导航",
    "快递",
    "包裹",
    "订单",
    "退款",
    "外卖",
    "药店",
    "医院",
    "挂号",
    "药片",
    "电梯",
    "门锁",
    "钥匙",
    "口罩",
    "作业",
    "老师",
    "会议",
    "文档",
    "照片",
    "视频",
    "消息",
    "微信",
]


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


def compact(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", "", text)


def gather_hits(records: list[dict[str, Any]], terms: list[str]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    ordered_terms = sorted(set(terms), key=lambda item: (-len(item), item))
    for term in ordered_terms:
        buckets[term] = {
            "term": term,
            "paired_hits": 0,
            "wz_hits": 0,
            "zh_hits": 0,
            "evidence_examples": [],
            "source_row_ids": set(),
            "source_files": set(),
        }

    for record in records:
        if not record.get("v1_candidate"):
            continue
        zh_text = compact(record.get("zh_text_train") or record.get("example_zh_train"))
        wz_text = compact(record.get("wz_text_train") or record.get("example_wz_train"))
        if not zh_text and not wz_text:
            continue
        for term in ordered_terms:
            zh_has = term in zh_text
            wz_has = term in wz_text
            if not zh_has and not wz_has:
                continue
            bucket = buckets[term]
            bucket["zh_hits"] += int(zh_has)
            bucket["wz_hits"] += int(wz_has)
            if zh_has and wz_has:
                bucket["paired_hits"] += 1
            bucket["source_files"].add(record.get("source_file"))
            bucket["source_row_ids"].add(str(record.get("source_row_id")))
            sentence = wz_text or zh_text
            if sentence and len(bucket["evidence_examples"]) < 5 and sentence not in bucket["evidence_examples"]:
                bucket["evidence_examples"].append(sentence)
    return list(buckets.values())


def build_row(hit: dict[str, Any]) -> dict[str, Any] | None:
    paired_hits = hit["paired_hits"]
    wz_hits = hit["wz_hits"]
    zh_hits = hit["zh_hits"]
    if paired_hits == 0 and wz_hits == 0:
        return None

    if paired_hits >= 2:
        confidence = 0.88
        usable = True
    elif paired_hits == 1:
        confidence = 0.8
        usable = True
    elif wz_hits >= 2:
        confidence = 0.7
        usable = True
    else:
        confidence = 0.55
        usable = False

    scene_payload = build_scene_payload(hit["term"], *hit["evidence_examples"])
    speech_payload = build_speech_function_payload(hit["term"], *hit["evidence_examples"])
    return {
        "term": hit["term"],
        "wz_word": hit["term"],
        "mandarin_headword": hit["term"],
        "source_type": "example_mined",
        "slot_kind": "noun",
        "paired_hits": paired_hits,
        "wz_hits": wz_hits,
        "zh_hits": zh_hits,
        "evidence_count": len(hit["evidence_examples"]),
        "evidence_examples": hit["evidence_examples"],
        "source_file_count": len(hit["source_files"]),
        "source_row_count": len(hit["source_row_ids"]),
        "confidence": confidence,
        "usable_as_slot": usable,
        "scene_tags": scene_payload["scene_tags"],
        "scene_scores": scene_payload["scene_scores"],
        "primary_scene_id": scene_payload["primary_scene_id"],
        "primary_scene_confidence": scene_payload["primary_scene_confidence"],
        "topic_scene_tags": scene_payload["scene_tags"],
        "topic_scene_scores": scene_payload["scene_scores"],
        "primary_topic_scene": scene_payload["primary_scene_id"],
        "topic_scene_confidence": scene_payload["primary_scene_confidence"],
        "speech_function_tags": speech_payload["speech_function_tags"],
        "speech_function_scores": speech_payload["speech_function_scores"],
        "primary_speech_function": speech_payload["primary_speech_function"],
        "speech_function_confidence": speech_payload["speech_function_confidence"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine high-frequency contemporary daily terms from examples.")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--min-confidence", type=float, default=0.7)
    args = parser.parse_args()

    records = load_jsonl(args.records)
    hits = gather_hits(records, SEED_TERMS)
    rows = [row for row in (build_row(hit) for hit in hits) if row is not None]
    rows.sort(key=lambda row: (-int(row["usable_as_slot"]), -row["paired_hits"], -row["wz_hits"], row["term"]))
    write_jsonl(args.output, rows)

    usable_rows = [row for row in rows if row["confidence"] >= args.min_confidence and row["usable_as_slot"]]
    write_summary(
        args.summary,
        {
            "seed_term_count": len(SEED_TERMS),
            "example_mined_count": len(rows),
            "usable_slot_count": len(usable_rows),
            "scene_counts": dict(Counter(row["primary_topic_scene"] for row in rows)),
        },
    )

    print(f"Loaded cleaned records: {len(records)}")
    print(f"Example-mined terms: {len(rows)}")
    print(f"Usable slot terms: {len(usable_rows)}")
    print(f"Wrote outputs to {args.output}")


if __name__ == "__main__":
    main()
