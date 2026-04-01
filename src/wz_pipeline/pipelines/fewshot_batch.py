#!/usr/bin/env python3
"""
Few-shot dictionary-grounded long sentence batch generation.

Improvements over pilot:
  - Word-combo dedup: no two tasks share the same 3-word set
  - Scene-balanced task allocation
  - Checkpoint/resume: writes results incrementally, skips completed task_ids on restart
  - Cross-sentence dedup pool (global)
  - Human-review-friendly output: one TSV file for easy expert checking

Usage:
  python scripts/generate_long_sentences_fewshot_batch.py --tasks 200 --provider deepseek
  python scripts/generate_long_sentences_fewshot_batch.py --tasks 200 --provider deepseek --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(LEGACY_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_SCRIPTS_DIR))

from build_controlled_generation_assets import ROOT, SCENE_CATALOG, SEMANTIC_CLASS_HINTS, build_scene_payload
from generate_controlled_sentences import build_client
from wz_pipeline.contracts import apply_contract
from wz_pipeline.grammar_guardrails import (
    GRAMMAR_PROMPT_RULES,
    GRAMMAR_REPAIR_SYSTEM_PROMPT,
    GRAMMAR_USER_RULES,
    build_grammar_repair_user_prompt,
    grammar_validation_reasons,
    should_attempt_grammar_repair,
)
from wz_pipeline.grammar_spec import GRAMMAR_SPEC_PATH, relevant_spec_labels
from wz_pipeline.jsonl import read_jsonl, write_jsonl
from wz_pipeline.registry import register_run
from wz_pipeline.review import export_review_tsv
from wz_pipeline.runs import build_summary, make_run_id, prepare_run_layout, write_json
from wz_pipeline.scene_policy import DEFAULT_SCENES, FOCUS_SCENES, MODERN_SIDECAR_SCENES, PRIORITY_SCENES
from wz_pipeline.source_surface_guardrails import detect_unsupported_surface_terms

# ---- paths ----
CLEANED_RECORDS = ROOT / "data" / "cleaned" / "cleaned_records_primary.jsonl"
EXTRACTED_SHORT = ROOT / "data" / "extracted_training_sentences" / "short_8_20.jsonl"
EXTRACTED_LONG = ROOT / "data" / "extracted_training_sentences" / "long_20_30.jsonl"
OUT_DIR = ROOT / "data" / "generated_long_sentences" / "fewshot_batch"
REPLACEABLE_LEXICON = ROOT / "data" / "controlled_generation" / "assets" / "replaceable_lexicon.jsonl"
DENSE_WHITELIST = ROOT / "data" / "controlled_generation" / "assets" / "dense_slot_lexicon_whitelist.jsonl"
EXTERNAL_TERM_CATALOG = ROOT / "data" / "controlled_generation" / "assets" / "external_term_catalog.json"
DENSE_SCENE_SEEDS = ROOT / "data" / "controlled_generation" / "assets" / "dense_scene_seed_terms.json"
FEWSHOT_CORE_POLICY = ROOT / "data" / "controlled_generation" / "assets" / "fewshot_core_policy.json"
MODERN_ANCHOR_CANDIDATES = ROOT / "data" / "generated_long_sentences" / "reports" / "modern_anchor_candidates.json"
MODERN_ANCHOR_EXAMPLES = ROOT / "data" / "generated_long_sentences" / "reports" / "modern_anchor_examples.jsonl"
PIPELINE_NAME = "fewshot_batch"

PAREN_RE = re.compile(r"[（(][^）)]{1,4}[）)]")
TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")

BAD_DEFINITION_HINTS = (
    "后置于",
    "前置于",
    "用于",
    "用在",
    "俗语",
    "助词",
    "量词",
    "副词",
    "介词",
    "连词",
    "语气词",
    "代词",
    "作副词",
    "在动词后",
)
NON_DAILY_DEFINITION_HINTS = (
    "比喻",
    "借指",
    "引申",
    "戏称",
    "绰号",
    "一种舞",
    "一类人",
)
ARCHAIC_DEFINITION_HINTS = (
    "旧俗",
    "旧时",
    "旧称",
    "古代",
    "古时",
    "旧式",
    "老式",
    "银圆",
    "酒筵",
    "传统木结构",
    "旧社会",
)
ARCHAIC_ALLOWED_HINTS = (
    "今指",
    "现在也指",
    "现指",
)
ABSTRACT_TIME_HINTS = (
    "明后天",
    "明天",
    "后天",
    "昨天",
    "今天",
    "以前",
    "以后",
    "现在",
    "刚才",
    "过去",
    "过些日子",
)
STOPWORDS = {
    "一个",
    "一种",
    "一件",
    "这个",
    "那个",
    "这里",
    "那里",
    "什么",
    "事情",
    "东西",
    "个",
    "了",
    "的",
    "着",
    "过",
    "在",
    "是",
    "有",
    "很",
    "真",
}
POLICY_VERSION = "track_b_mainline_v3_2026-04-01"
SHOPPING_OLD_MONEY_TERMS = {"银圆"}
SHOPPING_PROMPT_BLOCKED_TERMS = {"银圆", "屋宕", "银行", "鸡卵"}
SHOPPING_AMOUNT_TERMS = {
    "番钿", "十番钿", "五十番钿", "一百番钿", "两百番钿", "几十番钿", "百来番钿",
}
FOOD_SUPPORT_PRIORITY_TERMS = {
    "米饭",
    "鸡蛋",
    "苹果",
    "葡萄",
    "西瓜",
    "杨梅",
    "牛奶",
    "面包",
    "奶茶",
    "外卖",
    "鱼丸",
    "汤圆",
}
FOOD_BEVERAGE_TERMS = {"奶茶", "牛奶", "咖啡", "可乐"}
FOOD_STAPLE_TERMS = {"米饭", "面包", "外卖", "鱼丸", "汤圆", "鸡蛋"}
FOOD_ACTION_TERMS = {"吃饭", "烧菜", "煮饭", "点单", "打包"}
HOME_SUPPORT_PRIORITY_TERMS = {
    "洗衣机", "冰箱", "空调", "热水器", "电饭煲", "微波炉",
    "路由器", "阳台", "厨房", "书房", "收拾", "整理", "晾起",
}
SHOPPING_SUPPORT_PRIORITY_TERMS = {
    "付款码", "收款码", "支付宝", "二维码", "订单", "快递",
    "快递柜", "优惠券", "会员码", "超市", "网店", "直播间",
    "收银台", "客服", "售后", "电商",
    "番钿", "十番钿", "五十番钿", "一百番钿", "两百番钿", "几十番钿", "百来番钿",
}
TRANSPORT_SUPPORT_PRIORITY_TERMS = {
    "地铁", "高铁", "动车", "轻轨", "网约车", "共享单车",
    "导航", "车站", "地铁站", "高铁站", "机场", "候车室",
    "安检口", "行李箱", "车站大道", "五马街", "南塘街",
}
WEATHER_SUPPORT_PRIORITY_TERMS = {
    "雨伞", "雨衣", "口罩", "充电宝", "保温杯", "头盔",
    "应急包", "应急灯", "地库", "路口", "预警", "预报",
    "避雨", "保暖", "防滑",
}
SCENE_SUPPORT_POOL_SIZES = {
    "food_dining": 6,
    "home_life": 5,
    "shopping_payment": 5,
    "transport_trip": 5,
    "weather_safety": 5,
    "digital_chat": 4,
    "work_study": 4,
}


def clean_wz(text: str) -> str:
    text = PAREN_RE.sub("", text)
    return re.sub(r"[\s\u3000]+", "", text).strip()


def clean_zh(text: str) -> str:
    return re.sub(r"[\s\u3000]+", " ", str(text or "")).strip()


def normalize_scene_list(scene_arg: str | None) -> list[str]:
    if not scene_arg:
        return []
    return [scene.strip() for scene in scene_arg.split(",") if scene.strip()]


def slot_kind_from_semantic(semantic_class: str) -> str:
    if semantic_class in {"action"}:
        return "verb"
    if semantic_class in {"adjective"}:
        return "adjective"
    if semantic_class in {"place"}:
        return "place"
    return "noun"


def definition_looks_archaic(definition: str) -> bool:
    if not definition:
        return False
    if any(hint in definition for hint in ARCHAIC_ALLOWED_HINTS):
        return False
    return any(hint in definition for hint in ARCHAIC_DEFINITION_HINTS)


def has_sense_marker(text: str) -> bool:
    return any(marker in text for marker in ("(", ")", "（", "）"))


def load_core_policy() -> dict[str, Any]:
    if not FEWSHOT_CORE_POLICY.exists():
        return {"global": {}, "scene_policies": {}}
    return json.loads(FEWSHOT_CORE_POLICY.read_text(encoding="utf-8"))


def load_modern_anchor_gate() -> dict[str, dict[str, Any]]:
    if not MODERN_ANCHOR_CANDIDATES.exists():
        return {}
    try:
        data = json.loads(MODERN_ANCHOR_CANDIDATES.read_text(encoding="utf-8"))
    except Exception:
        return {}
    gates: dict[str, dict[str, Any]] = {}
    for scene_id, scene_data in (data.get("scenes") or {}).items():
        candidates = scene_data.get("candidates") or []
        gates[scene_id] = {
            "threshold_met": bool(scene_data.get("threshold_met")),
            "allowed_words": {str(row.get("wz_word") or "").strip() for row in candidates if str(row.get("wz_word") or "").strip()},
            "allowed_sentences": {clean_wz(str(row.get("wz_sentence") or "")) for row in candidates if str(row.get("wz_sentence") or "").strip()},
        }
    return gates


def scene_policy(scene_id: str) -> dict[str, Any]:
    return CORE_POLICY.get("scene_policies", {}).get(scene_id, {})


def global_deny_terms() -> set[str]:
    return set(CORE_POLICY.get("global", {}).get("deny_terms", []))


def stable_core_terms(scene_id: str) -> set[str]:
    policy = scene_policy(scene_id)
    return set(policy.get("stable_core_terms") or policy.get("allow_terms") or [])


def trial_modern_core_terms(scene_id: str) -> set[str]:
    policy = scene_policy(scene_id)
    return set(policy.get("trial_modern_core_terms") or policy.get("modern_core_terms") or [])


def scene_deny_terms(scene_id: str) -> set[str]:
    return set(scene_policy(scene_id).get("deny_terms", []))


def example_mentions_terms(example: dict[str, Any], terms: set[str]) -> bool:
    if not terms:
        return False
    haystacks = [
        clean_wz(str(example.get("wz_sentence") or "")),
        str(example.get("zh_sentence") or ""),
        str(example.get("definition") or ""),
        str(example.get("wz_word") or ""),
    ]
    return any(term and any(term in haystack for haystack in haystacks) for term in terms)


def prompt_examples_for_task(
    scene_id: str,
    examples: list[dict[str, Any]],
    banned_terms: set[str],
) -> tuple[list[dict[str, Any]], bool]:
    if not examples:
        return [], False
    blocked_terms = banned_terms
    if scene_id == "shopping_payment":
        blocked_terms = set(blocked_terms) | SHOPPING_PROMPT_BLOCKED_TERMS
    clean_examples = [example for example in examples if not example_mentions_terms(example, blocked_terms)]
    if len(clean_examples) >= 2:
        return clean_examples[:4], False
    if scene_id == "shopping_payment":
        # Old money terms in shopping examples strongly leak into generated output.
        # If we cannot form a clean prompt block, prefer no examples over contaminated ones.
        return clean_examples[:1], len(clean_examples) != len(examples)
    return examples[:4], False


# ===================== DATA LOADING =====================

def load_wz_word_set() -> set[str]:
    words = set()
    with open(CLEANED_RECORDS, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            w = row.get("wz_word_train", "")
            if w and 2 <= len(w) <= 4:
                words.add(w)
    return words


def load_primary_scene_by_word() -> dict[str, str]:
    scene_map: dict[str, str] = {}
    if not REPLACEABLE_LEXICON.exists():
        return scene_map
    with open(REPLACEABLE_LEXICON, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            wz_word = str(row.get("wz_word") or "").strip()
            scene_id = str(row.get("primary_scene_id") or "").strip()
            if wz_word and scene_id and wz_word not in scene_map:
                scene_map[wz_word] = scene_id
    return scene_map


def choose_scene_for_example(row: dict[str, Any], word_scene_map: dict[str, str]) -> str:
    source_wz_word = str(row.get("wz_word") or "").strip()
    lex_scene = word_scene_map.get(source_wz_word, "")
    payload_scene = build_scene_payload(
        row.get("wz_sentence", ""),
        row.get("zh_sentence", ""),
        row.get("wz_word", ""),
        row.get("definition", ""),
    ).get("primary_scene_id", "daily_chat")

    score_inputs = (
        row.get("wz_word", ""),
        row.get("definition", ""),
        row.get("zh_sentence", ""),
        row.get("wz_sentence", ""),
    )
    lex_score = scene_match_score(lex_scene, *score_inputs) if lex_scene else 0
    payload_score = scene_match_score(payload_scene, *score_inputs)

    if lex_scene and lex_score > 0 and lex_score >= payload_score:
        return lex_scene
    if payload_scene:
        return payload_scene
    return lex_scene or "daily_chat"


def choose_scene_for_word(row: dict[str, Any]) -> str:
    primary_scene = str(row.get("primary_scene_id") or "daily_chat").strip()
    score_inputs = (
        row.get("wz_word", ""),
        row.get("mandarin_headword", ""),
        row.get("definition", ""),
    )
    primary_score = scene_match_score(primary_scene, *score_inputs)
    if primary_score > 0:
        return primary_scene

    ranked_alts = [
        (
            scene_id,
            scene_match_score(scene_id, *score_inputs),
        )
        for scene_id in PRIORITY_SCENES
    ]
    ranked_alts.sort(key=lambda item: (item[1], item[0]), reverse=True)
    if ranked_alts and ranked_alts[0][1] > 0:
        return ranked_alts[0][0]
    return primary_scene


def load_examples_by_scene() -> dict[str, list[dict]]:
    word_scene_map = load_primary_scene_by_word()
    by_scene: dict[str, list[dict]] = defaultdict(list)
    with open(EXTRACTED_SHORT, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            scene = choose_scene_for_example(row, word_scene_map)
            row["scene_id"] = scene
            by_scene[scene].append(row)
    if MODERN_ANCHOR_EXAMPLES.exists():
        with open(MODERN_ANCHOR_EXAMPLES, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                scene = str(row.get("scene_id") or "").strip()
                if not scene:
                    continue
                by_scene[scene].append(
                    {
                        "scene_id": scene,
                        "wz_word": str(row.get("wz_word") or "").strip(),
                        "definition": str(row.get("definition") or "").strip(),
                        "wz_sentence": str(row.get("wz_sentence") or "").strip(),
                        "zh_sentence": str(row.get("zh_sentence") or "").strip(),
                        "source_file": str(row.get("source_file") or ""),
                    }
                )
    return dict(by_scene)


def append_scene_word(
    by_scene: dict[str, list[dict]],
    seen: set[tuple[str, str]],
    *,
    scene: str,
    wz_word: str,
    definition: str,
    semantic_class: str,
    slot_kind: str,
    source_type: str,
    modern_priority: int,
) -> None:
    key = (scene, wz_word)
    if (
        not scene
        or not wz_word
        or len(wz_word) < 2
        or len(wz_word) > 4
        or not definition
        or key in seen
    ):
        return
    seen.add(key)
    by_scene[scene].append(
        {
            "wz_word": wz_word,
            "definition": definition[:40],
            "scene_id": scene,
            "semantic_class": semantic_class or "unknown",
            "slot_kind": slot_kind or "unknown",
            "source_type": source_type,
            "is_modern": modern_priority > 0,
            "modern_priority": modern_priority,
        }
    )


def load_modern_words_from_assets(
    by_scene: dict[str, list[dict]],
    seen: set[tuple[str, str]],
) -> None:
    if DENSE_SCENE_SEEDS.exists():
        seed_data = json.loads(DENSE_SCENE_SEEDS.read_text(encoding="utf-8"))
        for scene, rows in seed_data.items():
            for row in rows:
                semantic_class = str(row.get("semantic_class") or "unknown")
                append_scene_word(
                    by_scene,
                    seen,
                    scene=scene,
                    wz_word=str(row.get("term") or "").strip(),
                    definition=str(row.get("term") or "").strip(),
                    semantic_class=semantic_class,
                    slot_kind=slot_kind_from_semantic(semantic_class),
                    source_type="scene_seed",
                    modern_priority=2,
                )

    if not EXTERNAL_TERM_CATALOG.exists():
        return

    data = json.loads(EXTERNAL_TERM_CATALOG.read_text(encoding="utf-8"))

    for row in data.get("example_mined_terms", []):
        semantic_class = str(row.get("semantic_class") or "unknown")
        target_scenes = row.get("scene_tags") or [row.get("primary_topic_scene") or "daily_chat"]
        for scene in target_scenes:
            append_scene_word(
                by_scene,
                seen,
                scene=str(scene),
                wz_word=str(row.get("term") or "").strip(),
                definition=str(row.get("term") or "").strip(),
                semantic_class=semantic_class,
                slot_kind=slot_kind_from_semantic(semantic_class),
                source_type="example_mined",
                modern_priority=3,
            )

    for row in data.get("place_names", []):
        target_scenes = row.get("scene_tags") or [row.get("primary_scene_id") or "transport_trip"]
        for scene in target_scenes:
            append_scene_word(
                by_scene,
                seen,
                scene=str(scene),
                wz_word=str(row.get("term") or "").strip(),
                definition="地点名词",
                semantic_class="place",
                slot_kind="place",
                source_type="place_name",
                modern_priority=3,
            )


def load_words_by_scene() -> dict[str, list[dict]]:
    lex_path = REPLACEABLE_LEXICON
    by_scene: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()

    load_modern_words_from_assets(by_scene, seen)

    if DENSE_WHITELIST.exists():
        with open(DENSE_WHITELIST, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                w = row.get("wz_word") or row.get("term") or ""
                d = row.get("mandarin_headword", "") or row.get("definition", "")
                scene = row.get("dense_whitelist_scene_id") or row.get("primary_scene_id") or "daily_chat"
                key = (scene, w)
                if (
                    scene in FOCUS_SCENES
                    and w
                    and 2 <= len(w) <= 4
                    and d
                    and key not in seen
                ):
                    seen.add(key)
                    by_scene[scene].append(
                        {
                            "wz_word": w,
                            "definition": d[:40],
                            "scene_id": scene,
                            "semantic_class": row.get("semantic_class", "unknown"),
                            "slot_kind": row.get("slot_kind", "unknown"),
                            "source_type": "dense_whitelist",
                            "is_modern": False,
                            "modern_priority": 1,
                        }
                    )

    if lex_path.exists():
        with open(lex_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                w = row.get("wz_word", "")
                d = row.get("mandarin_headword", "") or row.get("definition", "")
                scene = choose_scene_for_word(row)
                key = (scene, w)
                if w and 2 <= len(w) <= 4 and d and key not in seen:
                    if scene in FOCUS_SCENES and DENSE_WHITELIST.exists():
                        continue
                    seen.add(key)
                    by_scene[scene].append(
                        {
                            "wz_word": w,
                            "definition": d[:40],
                            "scene_id": scene,
                            "semantic_class": row.get("semantic_class", "unknown"),
                            "slot_kind": row.get("slot_kind", "unknown"),
                            "source_type": "replaceable_lexicon",
                            "is_modern": False,
                            "modern_priority": 0,
                        }
                    )

    if not by_scene:
        with open(CLEANED_RECORDS, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                w = row.get("wz_word_train", "")
                d = row.get("definition_norm", "")
                if w and 2 <= len(w) <= 4 and d and w not in seen:
                    seen.add(w)
                    by_scene["daily_chat"].append(
                        {
                            "wz_word": w,
                            "definition": d[:40],
                            "scene_id": "daily_chat",
                            "semantic_class": "unknown",
                            "slot_kind": "unknown",
                            "source_type": "cleaned_records",
                            "is_modern": False,
                            "modern_priority": 0,
                        }
                    )
    return dict(by_scene)


def keyword_vocab() -> list[str]:
    vocab: set[str] = set()
    for scene in SCENE_CATALOG:
        vocab.update(scene["zh_keywords"])
        vocab.update(scene["wz_keywords"])
    for hints in SEMANTIC_CLASS_HINTS.values():
        vocab.update(hints)
    return sorted({item for item in vocab if 2 <= len(item) <= 6}, key=lambda text: (-len(text), text))


KEYWORD_VOCAB = keyword_vocab()
SCENE_KEYWORDS = {
    scene["scene_id"]: sorted(
        {
            keyword
            for keyword in scene["zh_keywords"] + scene["wz_keywords"]
            if 1 <= len(keyword) <= 8
        },
        key=lambda text: (-len(text), text),
    )
    for scene in SCENE_CATALOG
}
CORE_POLICY = load_core_policy()
MODERN_ANCHOR_GATE = load_modern_anchor_gate()


def extract_keywords(*texts: str | None) -> list[str]:
    merged = " ".join(str(text or "") for text in texts if text)
    keywords: list[str] = []
    seen: set[str] = set()

    for keyword in KEYWORD_VOCAB:
        if keyword in merged and keyword not in seen:
            keywords.append(keyword)
            seen.add(keyword)

    for text in texts:
        raw = str(text or "")
        for token in TOKEN_RE.findall(raw):
            token = token.strip()
            if 2 <= len(token) <= 6 and token not in STOPWORDS and token not in seen:
                keywords.append(token)
                seen.add(token)

    return keywords[:12]


def scene_match_score(scene_id: str, *texts: str | None) -> int:
    merged = " ".join(str(text or "") for text in texts if text)
    if not merged:
        return 0
    return sum(1 for keyword in SCENE_KEYWORDS.get(scene_id, []) if keyword in merged)


def looks_abstract_time(word: str, definition: str) -> bool:
    merged = f"{word} {definition}"
    return any(hint in merged for hint in ABSTRACT_TIME_HINTS)


def core_is_blocked(
    scene_id: str,
    wz_word: str,
    definition: str,
    source_file: str = "",
) -> bool:
    global_policy = CORE_POLICY.get("global", {})
    scoped_policy = scene_policy(scene_id)

    if wz_word in set(global_policy.get("deny_terms", [])):
        return True
    if wz_word in set(scoped_policy.get("deny_terms", [])):
        return True
    if any(hint in definition for hint in global_policy.get("deny_definition_hints", [])):
        return True
    if source_file and any(hint in source_file for hint in global_policy.get("deny_source_files", [])):
        return True
    return False


def core_is_allowed(scene_id: str, wz_word: str) -> bool:
    allowed = stable_core_terms(scene_id)
    modern_terms = trial_modern_core_terms(scene_id)
    if not allowed and not modern_terms:
        return True
    return wz_word in allowed or wz_word in modern_terms


def modern_core_terms(scene_id: str) -> set[str]:
    return trial_modern_core_terms(scene_id)


def support_priority(scene_id: str, core_surface: str, word: dict[str, Any]) -> int:
    wz_word = str(word.get("wz_word") or "").strip()
    semantic_class = str(word.get("semantic_class") or "")
    slot_kind = str(word.get("slot_kind") or "")
    score = 0

    if scene_id == "food_dining":
        if wz_word in FOOD_SUPPORT_PRIORITY_TERMS:
            score += 3
        if semantic_class == "food":
            score += 2
        if slot_kind == "noun":
            score += 1
        if core_surface == "焯菜":
            if wz_word in FOOD_ACTION_TERMS:
                score += 3
            if wz_word in FOOD_STAPLE_TERMS:
                score += 1
        if core_surface == "涌烫":
            if wz_word in FOOD_BEVERAGE_TERMS or wz_word in FOOD_STAPLE_TERMS:
                score += 2
        return score

    scene_priority_terms = {
        "home_life": HOME_SUPPORT_PRIORITY_TERMS,
        "shopping_payment": SHOPPING_SUPPORT_PRIORITY_TERMS,
        "transport_trip": TRANSPORT_SUPPORT_PRIORITY_TERMS,
        "weather_safety": WEATHER_SUPPORT_PRIORITY_TERMS,
    }.get(scene_id, set())

    if wz_word in scene_priority_terms:
        score += 3
    if scene_id == "home_life" and semantic_class in {"device", "place", "action"}:
        score += 1
    if scene_id == "shopping_payment" and semantic_class in {"device", "institution", "action", "adjective", "amount"}:
        score += 1
    if scene_id == "shopping_payment" and wz_word in SHOPPING_AMOUNT_TERMS:
        score += 3
    if scene_id == "transport_trip" and semantic_class in {"transport", "place", "device", "action"}:
        score += 1
    if scene_id == "weather_safety" and semantic_class in {"device", "action", "adjective", "place"}:
        score += 1
    return score


def shared_char_count(left: str, right: str) -> int:
    return len(set(left) & set(right))


def word_looks_usable(word: dict[str, Any]) -> bool:
    wz_word = str(word.get("wz_word") or "").strip()
    definition = str(word.get("definition") or "").strip()
    if not wz_word or not definition:
        return False
    if has_sense_marker(wz_word):
        return False
    if any(hint in definition for hint in BAD_DEFINITION_HINTS):
        return False
    if any(hint in definition for hint in NON_DAILY_DEFINITION_HINTS):
        return False
    if definition_looks_archaic(definition):
        return False
    return True


def example_looks_usable(example: dict[str, Any]) -> bool:
    wz_word = str(example.get("wz_word") or "").strip()
    definition = str(example.get("definition") or "").strip()
    if not wz_word or not definition:
        return False
    if len(wz_word) < 2 or len(wz_word) > 4:
        return False
    if has_sense_marker(wz_word):
        return False
    if any(hint in definition for hint in BAD_DEFINITION_HINTS):
        return False
    if any(hint in definition for hint in NON_DAILY_DEFINITION_HINTS):
        return False
    if definition_looks_archaic(definition):
        return False
    return True


def core_word_looks_usable(word: dict[str, Any], scene_id: str, anchor: dict[str, Any] | None = None) -> bool:
    if not word_looks_usable(word):
        return False
    wz_word = str(word.get("wz_word") or "").strip()
    definition = str(word.get("definition") or "").strip()
    if not core_is_allowed(scene_id, wz_word):
        return False
    if looks_abstract_time(wz_word, definition):
        return False
    source_file = str(anchor.get("source_file") or "") if anchor is not None else ""
    if core_is_blocked(scene_id, wz_word, definition, source_file):
        return False
    score_inputs = [wz_word, definition]
    if anchor is not None:
        score_inputs.extend(
            [
                str(anchor.get("zh_sentence") or ""),
                str(anchor.get("wz_sentence") or ""),
            ]
        )
    if scene_id in FOCUS_SCENES and scene_match_score(scene_id, *score_inputs) <= 0:
        if not (
            word.get("is_modern")
            and scene_id == "food_dining"
            and str(word.get("semantic_class") or "") == "food"
        ):
            return False
    return True


def example_signature(example: dict[str, Any]) -> list[str]:
    return extract_keywords(
        example.get("wz_word"),
        example.get("definition"),
        example.get("zh_sentence"),
    )


def word_signature(word: dict[str, Any]) -> list[str]:
    return extract_keywords(word.get("wz_word"), word.get("definition"))


def overlap_score(left: list[str], right: list[str]) -> int:
    return len(set(left) & set(right))


def related_examples(anchor: dict[str, Any], scene_examples: list[dict], limit: int = 4) -> list[dict]:
    anchor_sig = example_signature(anchor)
    ranked: list[tuple[tuple[int, int, str], dict[str, Any]]] = []
    for example in scene_examples:
        example_sig = example_signature(example)
        overlap = overlap_score(anchor_sig, example_sig)
        if example.get("wz_word") != anchor.get("wz_word") and overlap <= 0:
            continue
        ranked.append(
            (
                (
                    overlap,
                    int(example.get("wz_word") == anchor.get("wz_word")),
                    example.get("wz_sentence", ""),
                ),
                example,
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[dict[str, Any]] = []
    seen_sentences: set[str] = set()
    for _, example in ranked:
        sentence = example.get("wz_sentence", "")
        if sentence in seen_sentences:
            continue
        selected.append(example)
        seen_sentences.add(sentence)
        if len(selected) >= limit:
            break
    return selected


def select_core_and_support_words(
    anchor: dict[str, Any],
    scene_words: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    scene_id: str,
    rng: random.Random | None = None,
    forced_core_word: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    anchor_sig = example_signature(anchor)
    core_word: dict[str, Any] | None = None

    if forced_core_word is not None:
        candidate = {
            "wz_word": forced_core_word.get("wz_word", ""),
            "definition": forced_core_word.get("definition", ""),
            "scene_id": forced_core_word.get("scene_id", scene_id),
            "semantic_class": forced_core_word.get("semantic_class", "unknown"),
            "slot_kind": forced_core_word.get("slot_kind", "unknown"),
            "source_type": forced_core_word.get("source_type", ""),
            "is_modern": forced_core_word.get("is_modern", False),
            "modern_priority": forced_core_word.get("modern_priority", 0),
        }
        if 2 <= len(str(candidate["wz_word"])) <= 4 and core_word_looks_usable(candidate, scene_id, anchor):
            core_word = candidate
            if scene_id == "food_dining" and candidate.get("is_modern"):
                return core_word, []
    else:
        anchor_word = {
            "wz_word": anchor.get("wz_word", ""),
            "definition": anchor.get("definition", ""),
            "scene_id": anchor.get("scene_id", "daily_chat"),
            "semantic_class": "unknown",
            "slot_kind": "unknown",
        }
        if 2 <= len(str(anchor_word["wz_word"])) <= 4 and core_word_looks_usable(anchor_word, scene_id, anchor):
            core_word = anchor_word

    ranked_support: list[tuple[tuple[int, int, int, float, str], dict[str, Any]]] = []
    example_keywords = extract_keywords(*(example.get("zh_sentence") for example in examples))
    combined_sig = anchor_sig + [keyword for keyword in example_keywords if keyword not in set(anchor_sig)]
    seen_words: set[str] = {str(core_word.get("wz_word"))} if core_word else set()
    core_surface = str(core_word.get("wz_word") or "") if core_word else ""

    for word in scene_words:
        wz_word = str(word.get("wz_word") or "").strip()
        if wz_word in seen_words or not word_looks_usable(word):
            continue
        if wz_word in scene_deny_terms(scene_id):
            continue
        if scene_id == "food_dining" and core_surface and shared_char_count(core_surface, wz_word) > 0:
            if str(word.get("semantic_class") or "") in {"food", "action"}:
                continue
        if scene_id in FOCUS_SCENES and scene_match_score(scene_id, wz_word, word.get("definition", "")) <= 0:
            if not (scene_id == "shopping_payment" and wz_word in SHOPPING_AMOUNT_TERMS):
                continue
        sig = word_signature(word)
        score = overlap_score(combined_sig, sig)
        if scene_id == "shopping_payment" and wz_word in SHOPPING_AMOUNT_TERMS:
            score = max(score, 1)
        if score <= 0 and not word.get("is_modern", False):
            continue
        scene_priority = support_priority(scene_id, core_surface, word)
        if scene_id == "food_dining" and core_surface == "焯菜" and wz_word in FOOD_BEVERAGE_TERMS:
            continue
        ranked_support.append(
            (
                (
                    scene_priority,
                    score,
                    int(word.get("modern_priority", 0)),
                    int(word.get("semantic_class") != "unknown"),
                    int(word.get("slot_kind") != "unknown"),
                    float(len(wz_word)),
                    wz_word,
                ),
                word,
            )
        )

    ranked_support.sort(key=lambda item: item[0], reverse=True)
    support_words: list[dict[str, Any]] = []
    support_candidates = [word for _, word in ranked_support]
    if len(support_candidates) > 1 and rng is not None:
        pool_size = SCENE_SUPPORT_POOL_SIZES.get(scene_id, 4)
        pool = support_candidates[: min(pool_size, len(support_candidates))]
        support_candidates = [rng.choice(pool) if rng is not None else pool[0]] + [
            word for word in support_candidates if word not in pool
        ]

    for word in support_candidates:
        wz_word = str(word.get("wz_word") or "").strip()
        if wz_word in seen_words:
            continue
        support_words.append(word)
        seen_words.add(wz_word)
        if len(support_words) >= 1:
            break

    if forced_core_word is None and core_word is None and support_words and core_word_looks_usable(support_words[0], scene_id, anchor):
        core_word = support_words.pop(0)
    return core_word, support_words


# ===================== TASK BUILDING =====================

SYSTEM_PROMPT = """你是温州话句子生成器。你的任务是根据提供的温州话词典例句和词汇，生成自然的温州话长句。

规则：
1. 每句必须 20-30 个字（含标点）
2. 每句必须使用给定的"核心词汇"
3. 辅助词汇只有在非常自然时才可加入，最多加入 1 个，不要硬塞
3. 句子要像温州人日常说话的口语，不是书面语
4. 保持例句中展示的方言特征，但不要为了像方言而乱拼功能词
5. 不要写成普通话
6. 每句要有完整的语义，适合语音训练朗读
7. 生成 5 句，每句独立
8. 优先围绕核心词展开一个完整、日常的小情境，不要把不相关词硬拼进一句
9. 如果辅助词是新事物、现代地点或设备名称，只在真正自然时带进去
10. 如果任务里列了“禁止词汇”，即使参考例句里出现了也绝对不要复用
11. 不要混入其他吴语区常见词形；只能跟参考例句、本地词表和给定词汇走，不会说就换成本地更稳的说法
12. 功能词语法必须比“像不像方言”更优先；拿不准时，宁可少用 `爻 / 罢 / 著埭 / 起 / 落去`
13. 不要自己发明新的两字到四字词；除给定词和参考例句能支持的说法外，拿不准就改写成来源里已有的稳妥表达
""" + "\n\n" + GRAMMAR_PROMPT_RULES + "\n\n只输出 JSON：\n" + '{"sentences": [{"wz": "温州话句子", "zh": "普通话翻译"}]}'

def build_task(
    examples: list[dict[str, Any]],
    core_word: dict[str, Any],
    support_words: list[dict[str, Any]],
    scene_id: str,
    task_id: str,
    lane: str,
    core_tier: str,
) -> dict[str, Any]:
    banned_terms = global_deny_terms() | scene_deny_terms(scene_id) | (SHOPPING_OLD_MONEY_TERMS if scene_id == "shopping_payment" else set())
    prompt_examples, examples_contaminated = prompt_examples_for_task(scene_id, examples, banned_terms)
    example_block = "\n".join(
        f"  {i+1}. 温州话：{ex['wz_sentence']}\n     普通话：{ex['zh_sentence']}"
        for i, ex in enumerate(prompt_examples)
    ) or "  - 本任务不展示旧例句，只保留核心词和辅助词，请直接生成自然口语句子。"
    support_block = "\n".join(
        f"  - {w['wz_word']}（{w['definition']}）"
        for w in support_words
    ) or "  - 无"
    banned_block = "\n".join(f"  - {term}" for term in sorted(banned_terms)) or "  - 无"
    contamination_note = ""
    if examples_contaminated:
        contamination_note = "\n注意：旧材料里有过时说法，只学句式，不要复用旧词。"
    shopping_amount_note = ""
    if scene_id == "shopping_payment" and any(str(w.get("wz_word") or "").strip() in SHOPPING_AMOUNT_TERMS for w in support_words):
        shopping_amount_note = "\n如果句子里提到价钱，优先直接用给定的具体金额说法，不要改成泛泛的“钞票”，也不要写“银圆”。"
    user_msg = f"""场景：{scene_id}

参考温州话例句（注意学习其中的方言风格和用词习惯）：
{example_block}

核心词汇（每句必须使用）：
  - {core_word['wz_word']}（{core_word['definition']}）

辅助词汇（只有自然时才用，最多用 1 个）：
{support_block}

禁止词汇（即使参考例句出现，也绝对不要写进新句子）：
{banned_block}{contamination_note}{shopping_amount_note}

请额外遵守这些温州话语法约束：
{GRAMMAR_USER_RULES}

如果一句话里需要额外内容词，优先复用参考例句和本地来源里已经出现过的说法，不要自己新造两字到四字词。

请生成 5 个 20-30 字的温州话口语长句。"""

    approved_modern_terms = sorted(
        {
            str(word.get("wz_word") or "").strip()
            for word in [core_word, *support_words]
            if str(word.get("wz_word") or "").strip()
            and (
                int(word.get("modern_priority", 0)) > 0
                or str(word.get("wz_word") or "").strip() in modern_core_terms(scene_id)
            )
        }
    )
    banned_terms_sorted = sorted(banned_terms)

    return {
        "task_id": task_id,
        "scene_id": scene_id,
        "lane": lane,
        "core_tier": core_tier,
        "policy_version": POLICY_VERSION,
        "example_sentences": [ex["wz_sentence"] for ex in prompt_examples],
        "example_sources": [
            {"wz_word": ex.get("wz_word", ""), "source_file": ex.get("source_file", "")}
            for ex in prompt_examples
        ],
        "core_word": core_word["wz_word"],
        "core_word_def": core_word["definition"],
        "support_words": [w["wz_word"] for w in support_words],
        "support_word_defs": {w["wz_word"]: w["definition"] for w in support_words},
        "target_words": [core_word["wz_word"]] + [w["wz_word"] for w in support_words],
        "approved_modern_terms": approved_modern_terms,
        "banned_terms": banned_terms_sorted,
        "prompt_examples_contaminated": examples_contaminated,
        "prompt_system": SYSTEM_PROMPT,
        "prompt_user": user_msg,
    }


def create_tasks_balanced(
    examples_by_scene: dict[str, list[dict]],
    words_by_scene: dict[str, list[dict]],
    num_tasks: int,
    seed: int = 42,
    scene_filter: list[str] | None = None,
    food_modern_trial_ratio: float = 0.25,
) -> list[dict]:
    """Create tasks with scene balance, anchor examples, and core-word-driven prompts."""
    rng = random.Random(seed)
    tasks = []
    used_task_signatures: set[tuple[str, str, frozenset[str]]] = set()

    # Determine which scenes have enough data
    allowed_scenes = set(scene_filter or PRIORITY_SCENES)
    viable_scenes = [
        s for s in PRIORITY_SCENES
        if s in allowed_scenes
        if s in examples_by_scene and len(examples_by_scene[s]) >= 3
        and s in words_by_scene and len(words_by_scene[s]) >= 3
        and (
            s not in MODERN_SIDECAR_SCENES
            or MODERN_ANCHOR_GATE.get(s, {}).get("threshold_met", False)
        )
    ]

    if not viable_scenes:
        # Fallback: any scene with enough data
        viable_scenes = [
            s for s in examples_by_scene
            if s in allowed_scenes
            if len(examples_by_scene[s]) >= 3
            and (
                s not in MODERN_SIDECAR_SCENES
                or MODERN_ANCHOR_GATE.get(s, {}).get("threshold_met", False)
            )
        ]
    if not viable_scenes:
        return []

    # Allocate tasks per scene (round-robin with priority)
    scene_quotas: dict[str, int] = defaultdict(int)
    for i in range(num_tasks):
        scene = viable_scenes[i % len(viable_scenes)]
        scene_quotas[scene] += 1

    max_retries_per_task = 20

    for scene in viable_scenes:
        quota = scene_quotas[scene]
        sidecar_gate = MODERN_ANCHOR_GATE.get(scene, {})
        allowed_anchor_words = sidecar_gate.get("allowed_words", set())
        allowed_anchor_sentences = sidecar_gate.get("allowed_sentences", set())
        scene_examples = examples_by_scene[scene]
        if scene in MODERN_SIDECAR_SCENES and sidecar_gate.get("threshold_met", False):
            gated_examples = [
                example
                for example in scene_examples
                if clean_wz(str(example.get("wz_sentence") or "")) in allowed_anchor_sentences
            ]
            if gated_examples:
                scene_examples = gated_examples
        scene_words = words_by_scene.get(scene, [])
        scene_word_set = {str(word.get("wz_word") or "").strip() for word in scene_words}
        scene_modern_core_terms = modern_core_terms(scene)
        modern_core_candidates = [
            word
            for word in scene_words
            if str(word.get("wz_word") or "").strip() in scene_modern_core_terms
            and word_looks_usable(word)
        ]
        food_trial_target = 0
        food_trial_done = 0
        used_trial_core_surfaces: set[str] = set()
        if scene == "food_dining" and modern_core_candidates and quota > 0:
            food_trial_target = max(1, round(quota * food_modern_trial_ratio))
            food_trial_target = min(food_trial_target, quota)
        word_example_counts = Counter(
            str(example.get("wz_word") or "").strip() for example in scene_examples
        )
        anchor_candidates = [
            example
            for example in scene_examples
            if example_looks_usable(example)
            and core_is_allowed(scene, str(example.get("wz_word") or "").strip())
            and not core_is_blocked(
                scene,
                str(example.get("wz_word") or "").strip(),
                str(example.get("definition") or "").strip(),
                str(example.get("source_file") or ""),
            )
            and str(example.get("wz_word") or "").strip() in scene_word_set
            and word_example_counts[str(example.get("wz_word") or "").strip()] >= 2
            and scene_match_score(
                scene,
                example.get("wz_word", ""),
                example.get("definition", ""),
                example.get("zh_sentence", ""),
                example.get("wz_sentence", ""),
            ) > 0
            and (
                scene not in MODERN_SIDECAR_SCENES
                or (
                    str(example.get("wz_word") or "").strip() in allowed_anchor_words
                    and clean_wz(str(example.get("wz_sentence") or "")) in allowed_anchor_sentences
                )
            )
        ]
        if not anchor_candidates:
            fallback_examples = [
                example
                for example in scene_examples
                if example_looks_usable(example)
                and (
                    scene not in MODERN_SIDECAR_SCENES
                    or (
                        str(example.get("wz_word") or "").strip() in allowed_anchor_words
                        and clean_wz(str(example.get("wz_sentence") or "")) in allowed_anchor_sentences
                    )
                )
            ]
            anchor_candidates = [
                example for example in scene_examples
                if example_looks_usable(example)
                and core_is_allowed(scene, str(example.get("wz_word") or "").strip())
                and not core_is_blocked(
                    scene,
                    str(example.get("wz_word") or "").strip(),
                    str(example.get("definition") or "").strip(),
                    str(example.get("source_file") or ""),
                )
                and word_example_counts[str(example.get("wz_word") or "").strip()] >= 2
                and scene_match_score(
                    scene,
                    example.get("wz_word", ""),
                    example.get("definition", ""),
                    example.get("zh_sentence", ""),
                    example.get("wz_sentence", ""),
                ) > 0
                and (
                    scene not in MODERN_SIDECAR_SCENES
                    or (
                        str(example.get("wz_word") or "").strip() in allowed_anchor_words
                        and clean_wz(str(example.get("wz_sentence") or "")) in allowed_anchor_sentences
                    )
                )
            ] or fallback_examples
        if not anchor_candidates:
            continue

        for _ in range(quota):
            task = None
            for _retry in range(max_retries_per_task):
                anchor = rng.choice(anchor_candidates)
                exs = related_examples(anchor, scene_examples, limit=4)
                if not exs:
                    exs = [anchor]
                if scene in FOCUS_SCENES and len(exs) < 2:
                    continue
                forced_core_word = None
                want_modern_trial = False
                if scene == "food_dining" and modern_core_candidates and food_trial_done < food_trial_target:
                    remaining_slots = quota - len([t for t in tasks if t["scene_id"] == scene])
                    remaining_trial = food_trial_target - food_trial_done
                    if remaining_slots <= remaining_trial or rng.random() < food_modern_trial_ratio:
                        want_modern_trial = True
                        unseen_modern_candidates = [
                            word
                            for word in modern_core_candidates
                            if str(word.get("wz_word") or "").strip() not in used_trial_core_surfaces
                        ]
                        pick_pool = unseen_modern_candidates or modern_core_candidates
                        forced_core_word = rng.choice(pick_pool)
                core_word, support_words = select_core_and_support_words(
                    anchor,
                    scene_words,
                    exs,
                    scene,
                    rng=rng,
                    forced_core_word=forced_core_word,
                )
                if core_word is None:
                    continue
                core_word_surface = str(core_word.get("wz_word") or "").strip()
                is_trial_modern = scene == "food_dining" and core_word_surface in scene_modern_core_terms
                if scene == "food_dining":
                    if want_modern_trial and not is_trial_modern:
                        continue
                    if not want_modern_trial and is_trial_modern:
                        continue
                    if is_trial_modern:
                        support_words = []
                lane = "food_modern_trial" if is_trial_modern else "mainline"
                core_tier = "trial_modern" if is_trial_modern else "stable"
                signature = (
                    scene,
                    core_word_surface,
                    frozenset(word["wz_word"] for word in support_words),
                )
                if signature in used_task_signatures:
                    continue
                used_task_signatures.add(signature)
                tid = f"fs_{hashlib.md5(f'{scene}_{len(tasks)}_{seed}'.encode()).hexdigest()[:10]}"
                task = build_task(exs, core_word, support_words, scene, tid, lane, core_tier)
                if lane == "food_modern_trial":
                    food_trial_done += 1
                    used_trial_core_surfaces.add(core_word_surface)
                break
            if task is not None:
                tasks.append(task)

    # Shuffle to interleave scenes (avoid hitting one API pattern too long)
    rng.shuffle(tasks)
    return tasks


# ===================== GENERATION =====================

def request_generation(client, model, task):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": task["prompt_system"]},
                {"role": "user", "content": task["prompt_user"]},
            ],
            temperature=0.6,
            response_format={"type": "json_object"},
            timeout=120.0,
        )
        text = resp.choices[0].message.content or ""
        data = json.loads(text)
        sentences = data.get("sentences", [])
        return sentences if isinstance(sentences, list) else []
    except Exception as e:
        print(f"  [ERROR] {e}")
        return []


def repair_sentence_grammar(client, model, task, wz_sentence: str, zh_sentence: str, reasons: list[str]) -> dict[str, str] | None:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": GRAMMAR_REPAIR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_grammar_repair_user_prompt(
                        scene_id=str(task.get("scene_id") or ""),
                        core_word=str(task.get("core_word") or ""),
                        support_words=list(task.get("support_words") or []),
                        wz_sentence=wz_sentence,
                        zh_sentence=zh_sentence,
                        reasons=reasons,
                    ),
                },
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            timeout=120.0,
        )
        text = resp.choices[0].message.content or ""
        data = json.loads(text)
        wz = str(data.get("wz") or "").strip()
        zh = str(data.get("zh") or "").strip()
        if not wz:
            return None
        return {"wz": wz, "zh": zh}
    except Exception as e:
        print(f"  [GRAMMAR REPAIR ERROR] {e}")
        return None


# ===================== VALIDATION =====================

MANDARIN_MARKERS = [
    "的话", "然后", "但是", "因为", "所以", "而且", "虽然", "如果",
    "可是", "或者", "不过", "已经", "正在", "刚才",
    "什么", "怎么", "为什么", "哪里", "这里", "那里",
    "他们", "她们", "我们", "你们",
    "非常", "特别", "真的是",
]


def validate_sentence(
    wz,
    zh,
    known_words,
    core_word,
    support_words,
    existing_sentences,
    *,
    scene_id: str,
    lane: str,
    core_tier: str,
    approved_modern_terms: list[str],
    banned_terms: list[str],
):
    reasons = []
    wz_clean = clean_wz(wz)
    char_len = len(wz_clean)
    grammar_reasons = grammar_validation_reasons(wz_clean)
    grammar_spec_sections = relevant_spec_labels(wz_clean, grammar_reasons)
    unsupported_surface_terms = detect_unsupported_surface_terms(
        wz_clean,
        protected_terms=[core_word, *support_words, *approved_modern_terms],
    )

    if char_len < 20:
        reasons.append(f"too_short:{char_len}")
    elif char_len > 30:
        reasons.append(f"too_long:{char_len}")

    found_known = sorted({w for w in known_words if w in wz_clean})
    core_present = core_word in wz_clean if core_word else False
    found_support = sorted({w for w in support_words if w in wz_clean})
    approved_modern_hits = sorted({w for w in approved_modern_terms if w in wz_clean})
    banned_term_hits = sorted({w for w in banned_terms if w in wz_clean})
    coverage_hits = len(set(found_known) | set(approved_modern_hits))
    if not core_present:
        reasons.append("core_word_missing")
    coverage_floor = 1 if lane == "food_modern_trial" else 2
    if coverage_hits < coverage_floor:
        reasons.append(f"low_wz_word_coverage:{coverage_hits}")
    if len(found_support) > 1:
        reasons.append(f"support_words_overused:{len(found_support)}")
    if banned_term_hits:
        reasons.append(f"banned_terms:{','.join(banned_term_hits)}")
    for finding in unsupported_surface_terms:
        reasons.append(f"source_surface_missing:{finding['surface']}")

    mandarin_hits = [m for m in MANDARIN_MARKERS if m in wz_clean]
    if len(mandarin_hits) >= 2:
        reasons.append(f"mandarin_markers:{','.join(mandarin_hits[:3])}")

    if wz_clean in existing_sentences:
        reasons.append("duplicate")

    if not zh or not zh.strip():
        reasons.append("no_translation")
    for reason in grammar_reasons:
        reasons.append(f"grammar:{reason}")

    return {
        "pass": len(reasons) == 0,
        "reasons": reasons,
        "char_len": char_len,
        "found_known_words": len(found_known),
        "coverage_hits": coverage_hits,
        "core_word_present": core_present,
        "found_support_words": len(found_support),
        "approved_modern_hits": approved_modern_hits,
        "banned_term_hits": banned_term_hits,
        "lane": lane,
        "core_tier": core_tier,
        "mandarin_markers": mandarin_hits,
        "grammar_reasons": grammar_reasons,
        "grammar_spec_path": str(GRAMMAR_SPEC_PATH),
        "grammar_spec_sections": grammar_spec_sections,
        "unsupported_surface_terms": unsupported_surface_terms,
    }


def write_review_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "序号\tlane\tcore_tier\t温州话句子\t普通话翻译\t字数\t场景\t核心词\t辅助词\tapproved_modern_hits\tbanned_term_hits\t骨架来源词条\t骨架例句\t专家判定\t备注\n"
        )
        for idx, r in enumerate(rows, 1):
            v = r["validation"]
            src_words = "|".join(s.get("wz_word", "") for s in r.get("example_sources", []))
            src_sentences = " / ".join(r.get("example_sentences", []))
            f.write(
                f"{idx}\t{r.get('lane','mainline')}\t{r.get('core_tier','stable')}\t"
                f"{r['wz_sentence']}\t{r['zh_sentence']}\t{v['char_len']}\t{r['scene_id']}\t"
                f"{r.get('core_word','')}\t{'|'.join(r.get('support_words',[]))}\t"
                f"{'|'.join(v.get('approved_modern_hits',[]))}\t{'|'.join(v.get('banned_term_hits',[]))}\t"
                f"{src_words}\t{src_sentences}\t\t\n"
            )


# ===================== CHECKPOINT =====================

def load_completed_ids(results_path: Path) -> tuple[set[str], set[str]]:
    """Load already-completed task_ids and existing sentences from results file."""
    done_ids = set()
    seen_sentences = set()
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done_ids.add(row.get("task_id", ""))
                seen_sentences.add(row.get("wz_sentence", ""))
    return done_ids, seen_sentences


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(description="Batch few-shot long sentence generation with checkpoint.")
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument("--provider", choices=["primary", "secondary", "qwen", "deepseek"], default="deepseek")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument(
        "--scenes",
        type=str,
        default=",".join(DEFAULT_SCENES),
        help="Comma-separated scene ids to include",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional compatibility mirror directory for legacy-style batch files",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--run-id", type=str, default="", help="Explicit run id; required for reliable resume")
    parser.add_argument(
        "--food-modern-trial-ratio",
        type=float,
        default=0.25,
        help="Share of food_dining tasks that use trial modern cores",
    )
    parser.add_argument(
        "--disable-grammar-repair",
        action="store_true",
        help="Skip the second-pass grammar repair step for sentences with high-risk function words.",
    )
    args = parser.parse_args()

    if args.resume and not args.run_id:
        raise SystemExit("--resume requires --run-id so the pipeline can reopen the same run directory.")

    client, model, _ = build_client(args.provider)
    print(f"Provider: {args.provider}, Model: {model}")
    run_id = args.run_id or make_run_id(
        pipeline_name=PIPELINE_NAME,
        provider=args.provider,
        model=model,
        policy_version=POLICY_VERSION,
        seed=args.seed,
    )
    layout = prepare_run_layout(pipeline_name=PIPELINE_NAME, run_id=run_id)
    results_path = layout.results_path
    rule_gate_path = layout.rule_gate_path
    config = {
        "tasks": args.tasks,
        "provider": args.provider,
        "model": model,
        "seed": args.seed,
        "sleep": args.sleep,
        "scenes": [],
        "food_modern_trial_ratio": args.food_modern_trial_ratio,
        "grammar_repair_enabled": not args.disable_grammar_repair,
        "policy_version": POLICY_VERSION,
        "compatibility_output_dir": str(args.output_dir) if args.output_dir else "",
    }

    print("Loading data...")
    known_words = load_wz_word_set()
    examples_by_scene = load_examples_by_scene()
    words_by_scene = load_words_by_scene()
    selected_scenes = normalize_scene_list(args.scenes)
    config["scenes"] = selected_scenes or DEFAULT_SCENES

    print(f"Known WZ words (2-4 chars): {len(known_words)}")
    print(f"Selected scenes: {selected_scenes or DEFAULT_SCENES}")

    # Load existing sentences for dedup
    existing = set()
    for path in [EXTRACTED_SHORT, EXTRACTED_LONG]:
        if path.exists():
            with open(path) as f:
                for line in f:
                    existing.add(json.loads(line)["wz_sentence"])

    # Resume support
    done_ids = set()
    if args.resume:
        done_ids, prev_sentences = load_completed_ids(results_path)
        existing.update(prev_sentences)
        print(f"Resuming: {len(done_ids)} tasks already done, {len(prev_sentences)} sentences in dedup pool")

    print(f"Existing sentences for dedup: {len(existing)}")

    print(f"\nCreating {args.tasks} tasks (scene-balanced, core-word-driven)...")
    tasks = create_tasks_balanced(
        examples_by_scene,
        words_by_scene,
        args.tasks,
        args.seed,
        scene_filter=selected_scenes,
        food_modern_trial_ratio=args.food_modern_trial_ratio,
    )

    # Filter out already-done tasks
    pending = [t for t in tasks if t["task_id"] not in done_ids]
    print(f"Pending tasks: {len(pending)} (skipped {len(tasks) - len(pending)} already done)")

    if not pending:
        print("All tasks already completed. Use a different --seed or increase --tasks.")
        return
    write_json(layout.config_path, config)

    # Open results file in append mode
    results_f = open(results_path, "a", encoding="utf-8")
    rule_gate_f = open(rule_gate_path, "a", encoding="utf-8")
    stats = Counter()
    t_start = time.time()

    try:
        for i, task in enumerate(pending):
            elapsed = time.time() - t_start
            rate = (i / elapsed * 60) if elapsed > 0 and i > 0 else 0
            print(f"\n[{i+1}/{len(pending)}] {task['task_id']} ({task['scene_id']}) | {rate:.1f} tasks/min")
            print(f"  Core: {task['core_word']} | Support: {task['support_words']}")

            sentences = request_generation(client, model, task)
            stats["raw_sentences"] += len(sentences)

            for sent in sentences:
                wz = sent.get("wz", "")
                zh = sent.get("zh", "")
                if not wz:
                    stats["empty"] += 1
                    continue
                original_wz = clean_wz(wz)
                original_zh = zh
                repair_trigger_reasons: list[str] = []
                repair_applied = False
                wz_precheck = original_wz
                precheck_reasons = grammar_validation_reasons(wz_precheck)
                if not args.disable_grammar_repair and should_attempt_grammar_repair(wz_precheck):
                    repair_trigger_reasons = precheck_reasons or ["grammar_marker_review"]
                    repaired = repair_sentence_grammar(
                        client,
                        model,
                        task,
                        wz_precheck,
                        zh,
                        repair_trigger_reasons,
                    )
                    if repaired:
                        wz = repaired.get("wz", wz)
                        zh = repaired.get("zh", zh)
                        repair_applied = clean_wz(wz) != original_wz or zh != original_zh

                val = validate_sentence(
                    wz,
                    zh,
                    known_words,
                    task["core_word"],
                    task["support_words"],
                    existing,
                    scene_id=task["scene_id"],
                    lane=task.get("lane", "mainline"),
                    core_tier=task.get("core_tier", "stable"),
                    approved_modern_terms=task.get("approved_modern_terms", []),
                    banned_terms=task.get("banned_terms", []),
                )
                result = {
                    "task_id": task["task_id"],
                    "scene_id": task["scene_id"],
                    "lane": task.get("lane", "mainline"),
                    "core_tier": task.get("core_tier", "stable"),
                    "policy_version": task.get("policy_version", POLICY_VERSION),
                    "wz_sentence": clean_wz(wz),
                    "zh_sentence": zh,
                    "core_word": task["core_word"],
                    "core_word_def": task.get("core_word_def", ""),
                    "support_words": task["support_words"],
                    "support_word_defs": task.get("support_word_defs", {}),
                    "target_words": task["target_words"],
                    "approved_modern_terms": task.get("approved_modern_terms", []),
                    "banned_terms": task.get("banned_terms", []),
                    "example_sources": task.get("example_sources", []),
                    "example_sentences": task.get("example_sentences", []),
                    "grammar_repair_applied": repair_applied,
                    "grammar_repair_trigger_reasons": repair_trigger_reasons,
                    "validation": val,
                }
                if repair_applied:
                    result["pre_repair_wz_sentence"] = original_wz
                    result["pre_repair_zh_sentence"] = original_zh
                rule_gated_result = apply_contract(
                    result,
                    source_type="generated_candidate",
                    origin_run_id=run_id,
                    pipeline_name=PIPELINE_NAME,
                    policy_version=str(result.get("policy_version") or POLICY_VERSION),
                    scene_id=str(result.get("scene_id") or ""),
                    lane=str(result.get("lane") or ""),
                    core_word=str(result.get("core_word") or ""),
                    support_words=result.get("support_words", []),
                    rule_gate_status="pass" if val["pass"] else "fail",
                    human_review_status="pending",
                    trust_tier="candidate",
                    upstream_eligible=False,
                    sample_prefix="gen",
                )

                results_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                rule_gate_f.write(json.dumps(rule_gated_result, ensure_ascii=False) + "\n")
                results_f.flush()
                rule_gate_f.flush()
                existing.add(clean_wz(wz))

                if val["pass"]:
                    stats["pass"] += 1
                    print(f"  PASS [{val['char_len']}字]: {clean_wz(wz)}")
                else:
                    stats["fail"] += 1

            stats["tasks_done"] += 1

            if args.sleep > 0 and i < len(pending) - 1:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Saving progress...")
    finally:
        results_f.close()
        rule_gate_f.close()

    all_raw_results = read_jsonl(results_path)
    all_rule_results = read_jsonl(rule_gate_path)
    passed = [row for row in all_rule_results if row.get("rule_gate_status") == "pass"]

    export_review_tsv(layout.review_tsv_path, passed)
    write_jsonl(layout.promotion_candidates_path, [])

    total_elapsed = time.time() - t_start
    lane_pass_counts = Counter(r.get("lane", "mainline") for r in passed)
    core_tier_pass_counts = Counter(r.get("core_tier", "stable") for r in passed)
    scene_pass_counts = Counter(r.get("scene_id", "") for r in passed)
    banned_term_fail_counts = Counter()
    grammar_fail_counts = Counter()
    for row in all_rule_results:
        for reason in row["validation"].get("grammar_reasons", []):
            grammar_fail_counts[reason] += 1
        if row.get("rule_gate_status") == "pass":
            continue
        for term in row["validation"].get("banned_term_hits", []):
            banned_term_fail_counts[term] += 1
    trial_core_pass_counts = Counter(
        r.get("core_word", "")
        for r in passed
        if r.get("core_tier") == "trial_modern"
    )
    summary = build_summary(
        pipeline_name=PIPELINE_NAME,
        run_id=run_id,
        config=config,
        raw_count=len(all_raw_results),
        rule_pass_count=len(passed),
        reviewed_count=0,
        human_pass_count=0,
        scene_human_pass_rate={},
        lane_human_pass_rate={},
        extra={
            "tasks_total": len(tasks),
            "tasks_done": len({row.get("task_id", "") for row in all_raw_results if row.get("task_id")}),
            "rule_fail_count": len(all_rule_results) - len(passed),
            "scene_rule_pass_counts": dict(scene_pass_counts),
            "lane_rule_pass_counts": dict(lane_pass_counts),
            "core_tier_rule_pass_counts": dict(core_tier_pass_counts),
            "banned_term_fail_counts": dict(banned_term_fail_counts),
            "grammar_fail_counts": dict(grammar_fail_counts),
            "trial_core_pass_counts": dict(trial_core_pass_counts),
            "policy_version": POLICY_VERSION,
            "provider": args.provider,
            "model": model,
            "elapsed_seconds": round(total_elapsed, 1),
            "output_files": {
                "config": str(layout.config_path),
                "results": str(layout.results_path),
                "rule_gate": str(layout.rule_gate_path),
                "review": str(layout.review_tsv_path),
                "promotion_candidates": str(layout.promotion_candidates_path),
                "summary": str(layout.summary_path),
            },
        },
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
            "provider": args.provider,
            "model": model,
        }
    )

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        passed_path = args.output_dir / "batch_passed.jsonl"
        summary_path = args.output_dir / "batch_summary.json"
        review_path = args.output_dir / "human_review.tsv"
        mainline_review_path = args.output_dir / "mainline_human_review.tsv"
        trial_review_path = args.output_dir / "food_modern_trial_human_review.tsv"
        write_jsonl(args.output_dir / "batch_results.jsonl", all_raw_results)
        write_jsonl(passed_path, passed)
        export_review_tsv(review_path, passed)
        export_review_tsv(mainline_review_path, [r for r in passed if r.get("lane") == "mainline"])
        export_review_tsv(trial_review_path, [r for r in passed if r.get("lane") == "food_modern_trial"])
        compatibility_summary = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "tasks_total": len(tasks),
            "tasks_done": len({row.get("task_id", "") for row in all_raw_results if row.get("task_id")}),
            "raw_sentences": len(all_raw_results),
            "passed": len(passed),
            "failed": len(all_rule_results) - len(passed),
            "pass_rate": round(len(passed) / max(1, len(all_rule_results)), 4),
            "scene_pass_counts": dict(scene_pass_counts),
            "lane_pass_counts": dict(lane_pass_counts),
            "core_tier_pass_counts": dict(core_tier_pass_counts),
            "banned_term_fail_counts": dict(banned_term_fail_counts),
            "trial_core_pass_counts": dict(trial_core_pass_counts),
            "policy_version": POLICY_VERSION,
            "provider": args.provider,
            "model": model,
            "elapsed_seconds": round(total_elapsed, 1),
            "output_files": {
                "results": str(args.output_dir / "batch_results.jsonl"),
                "passed": str(passed_path),
                "human_review_tsv": str(review_path),
                "mainline_human_review_tsv": str(mainline_review_path),
                "food_modern_trial_human_review_tsv": str(trial_review_path),
                "summary": str(summary_path),
            },
        }
        write_json(summary_path, compatibility_summary)

    print(f"\n{'='*60}")
    print("BATCH RESULTS")
    print(f"Run id: {run_id}")
    print(f"Tasks done: {len({row.get('task_id', '') for row in all_raw_results if row.get('task_id')})}/{len(tasks)}")
    print(f"Raw sentences: {len(all_raw_results)}")
    print(f"Rule-pass rows: {len(passed)} ({summary['machine_metrics']['rule_pass_rate']*100:.1f}%)")
    print(f"Rule-fail rows: {len(all_rule_results) - len(passed)}")
    print(f"Time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"Scene distribution: {dict(scene_pass_counts)}")
    print(f"\nOutput files:")
    print(f"  Config: {layout.config_path}")
    print(f"  Results: {layout.results_path}")
    print(f"  Rule gate: {layout.rule_gate_path}")
    print(f"  Review TSV: {layout.review_tsv_path}")
    print(f"  Summary: {layout.summary_path}")


if __name__ == "__main__":
    main()
