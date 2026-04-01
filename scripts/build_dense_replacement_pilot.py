#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from build_controlled_generation_assets import (
    ROOT,
    SCENE_CATALOG,
    SEMANTIC_CLASS_HINTS,
    scene_compatible,
)


DEFAULT_SKELETONS = (
    ROOT
    / "data"
    / "controlled_generation"
    / "skeletons"
    / "dense_reusable_llm_reviewed.jsonl"
)
DEFAULT_LEXICON = (
    ROOT / "data" / "controlled_generation" / "assets" / "dense_slot_lexicon_whitelist.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "controlled_generation"
    / "tasks"
    / "dense_replacement_pilot_tasks.jsonl"
)
DEFAULT_SUMMARY = (
    ROOT
    / "data"
    / "controlled_generation"
    / "reports"
    / "dense_replacement_pilot_summary.json"
)
SYSTEM_PROMPT = (
    "你是温州话高密度实词替换助手。"
    "你必须保留固定块、语气、标点和句法顺序。"
    "不要擅自增删或改写 `爻`、`罢`、`著埭`、`起`、`落去`、`不/未/冇` 这类功能词。"
    "你只能替换 eligible content slots，且所有 eligible content slots 都必须替换。"
    "每个槽位只能使用该槽位池中的词，不能跨槽位交换。"
    "不允许增加槽位池之外的新核心实词。"
    "输出必须是 JSON object，并返回 wz_sentence、slot_values、naturalness_note。"
)

LOCKED_HEADWORD_HINTS = (
    "后置于",
    "前置于",
    "用于",
    "用在",
    "置于",
    "俗语",
    "助词",
    "量词",
    "副词",
    "介词",
    "连词",
    "语气词",
    "代词",
    "补语",
    "状语",
    "作副词",
    "在动词后",
    "尊敬的口吻",
)
GENERIC_GLOSS_TOKENS = {
    "这个",
    "那个",
    "这里",
    "那里",
    "东西",
    "事情",
    "一下",
    "一个",
    "一种",
    "有点",
    "什么",
    "别人",
    "大家",
}
GLOSS_STOPWORDS = {
    "的",
    "了",
    "着",
    "过",
    "把",
    "被",
    "和",
    "跟",
    "在",
    "是",
    "有",
    "个",
    "种",
    "这",
    "那",
    "去",
    "来",
    "再",
    "先",
    "后",
    "时",
    "候",
}
TIME_DISALLOWED_KEYWORDS = {
    "昨天",
    "今天",
    "明天",
    "中午",
    "夜里",
    "半夜",
    "早起",
    "黄昏",
    "辰光",
    "钟头",
    "时候",
    "以前",
    "后来",
    "夏天",
}
STATE_DISALLOWED_KEYWORDS = {
    "疲倦",
    "模糊",
    "不走运",
    "发霉",
    "害羞",
    "羞怯",
    "冷酷",
    "热闹",
    "不灵活",
    "滚热",
    "体温很高",
}
FIXED_EXPRESSION_DISALLOWED_KEYWORDS = {
    "俗语",
    "成语",
    "作副词",
    "在动词后",
    "用于",
    "用在",
    "后置于",
    "前置于",
    "尊敬的口吻",
    "通常用于",
}
STRICT_SEMANTIC_COMPATIBILITY: dict[str, set[str]] = {
    "place": {"place", "institution"},
    "institution": {"institution", "place"},
    "food": {"food"},
    "device": {"device"},
    "transport": {"transport"},
    "person": {"person"},
    "action": {"action"},
    "adjective": {"adjective"},
    "medicine": {"medicine"},
    "document": {"document"},
    "service": {"service"},
}
CHINESE_SEGMENT_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")


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


def build_user_prompt(task_payload: dict[str, Any]) -> str:
    return "根据下面 JSON 生成句子，只返回 JSON：\n" + json.dumps(
        task_payload, ensure_ascii=False, indent=2
    )


def slot_is_eligible(slot: dict[str, Any], skeleton: dict[str, Any]) -> bool:
    llm_slots = set(skeleton.get("llm_dense_eligible_slot_ids") or [])
    if llm_slots:
        return slot.get("slot_id") in llm_slots
    return bool(slot.get("dense_replaceable_rule"))


def term_looks_locked(row: dict[str, Any]) -> bool:
    headword = str(row.get("mandarin_headword") or "").strip()
    if not headword:
        return False
    return any(hint in headword for hint in LOCKED_HEADWORD_HINTS)


def target_eligible_slot_count(skeleton: dict[str, Any]) -> int:
    sentence_length = len(str(skeleton.get("source_wz_sentence") or ""))
    if sentence_length <= 12:
        return 2
    if sentence_length <= 18:
        return 3
    return 4


def target_pool_size_for_slot(slot: dict[str, Any], sentence_length: int) -> int:
    priority = slot.get("dense_priority") or "low"
    pool_total = int(slot.get("dense_candidate_pool_total") or 0)
    if priority == "high":
        base = 6 if sentence_length >= 16 else 5
    elif priority == "medium":
        base = 5 if sentence_length >= 14 else 4
    else:
        base = 4
    if pool_total > 0:
        base = min(base, pool_total)
    return max(3, min(base, 6))


def slot_rank_key(slot: dict[str, Any]) -> tuple[int, int, float, str]:
    priority_order = {"high": 2, "medium": 1, "low": 0}
    return (
        priority_order.get(slot.get("dense_priority") or "low", 0),
        int(slot.get("dense_candidate_pool_total") or 0),
        float(slot.get("confidence", 0.0)),
        slot.get("slot_id", ""),
    )


def rank_eligible_slots(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(slots, key=slot_rank_key, reverse=True)


def normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def extract_slot_context(skeleton_template: str, slot_id: str, window: int = 6) -> tuple[str, str]:
    marker = f"[{slot_id}]"
    if marker not in skeleton_template:
        return "", ""
    left, right = skeleton_template.split(marker, 1)
    return left[-window:], right[:window]


def gloss_keyword_vocab() -> list[str]:
    vocab: set[str] = set()
    for scene in SCENE_CATALOG:
        vocab.update(kw for kw in scene["zh_keywords"] if kw and len(kw) <= 6)
    for hints in SEMANTIC_CLASS_HINTS.values():
        vocab.update(hint for hint in hints if hint and len(hint) <= 6)
    vocab.update(
        {
            "回来",
            "带走",
            "坐车",
            "拿走",
            "放好",
            "放下",
            "买菜",
            "做饭",
            "吃饭",
            "付款",
            "结账",
            "快递",
            "打车",
            "公交",
            "地铁",
            "天气",
            "下雨",
            "台风",
            "行李",
            "皮箱",
            "房子",
            "屋里",
            "空调",
            "洗衣机",
            "奶茶",
            "外卖",
        }
    )
    return sorted(vocab, key=lambda item: (-len(item), item))


GLOSS_KEYWORD_VOCAB = gloss_keyword_vocab()


def extract_gloss_keywords(*texts: str | None) -> list[str]:
    merged_text = " ".join(str(text or "") for text in texts if text)
    keywords: list[str] = []
    seen: set[str] = set()

    for keyword in GLOSS_KEYWORD_VOCAB:
        if keyword in merged_text and keyword not in seen:
            keywords.append(keyword)
            seen.add(keyword)

    for text in texts:
        raw = str(text or "")
        for segment in CHINESE_SEGMENT_RE.findall(raw):
            if len(segment) <= 4 and segment not in GLOSS_STOPWORDS and segment not in GENERIC_GLOSS_TOKENS:
                if segment not in seen:
                    keywords.append(segment)
                    seen.add(segment)
            if len(segment) > 4:
                for keyword in GLOSS_KEYWORD_VOCAB:
                    if keyword in segment and keyword not in seen:
                        keywords.append(keyword)
                        seen.add(keyword)

    keywords = [keyword for keyword in keywords if keyword not in GLOSS_STOPWORDS and keyword not in GENERIC_GLOSS_TOKENS]
    return keywords[:8]


def derive_disallowed_gloss_keywords(slot: dict[str, Any]) -> list[str]:
    slot_kind = str(slot.get("slot_kind") or "unknown")
    semantic_class = str(slot.get("semantic_class") or "unknown")
    disallowed: set[str] = set(TIME_DISALLOWED_KEYWORDS) | set(FIXED_EXPRESSION_DISALLOWED_KEYWORDS)
    if slot_kind in {"verb", "adjective"} or semantic_class in {"action", "adjective"}:
        disallowed |= STATE_DISALLOWED_KEYWORDS
    if slot_kind == "noun" or semantic_class in {"food", "device", "transport", "person", "place", "institution"}:
        disallowed |= {"作副词", "在动词后", "通常用于"}
    return sorted(disallowed)


def build_slot_signature(slot: dict[str, Any], skeleton: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    slot_id = slot.get("slot_id")
    surface_wz = str(slot.get("surface_wz") or "").strip()
    surface_zh = str(slot.get("surface_zh") or "").strip()
    slot_kind = str(slot.get("slot_kind") or "unknown")
    semantic_class = str(slot.get("semantic_class") or "unknown")
    if not slot_id:
        return None, "slot_semantics_unclear"
    if slot_kind in {"unknown", "function", "time"} or semantic_class == "unknown":
        return None, "slot_semantics_unclear"
    if not surface_wz or not surface_zh:
        return None, "slot_semantics_unclear"
    if any(hint in surface_zh for hint in LOCKED_HEADWORD_HINTS):
        return None, "fixed_expression_core"
    source_gloss_keywords = extract_gloss_keywords(
        surface_zh,
        surface_zh.replace("（", " ").replace("）", " "),
    )
    if len(source_gloss_keywords) < 1:
        source_gloss_keywords = extract_gloss_keywords(
            surface_zh,
            surface_zh.replace("（", " ").replace("）", " "),
            skeleton.get("source_zh_sentence"),
        )
    if not source_gloss_keywords:
        return None, "slot_semantics_unclear"
    left_context, right_context = extract_slot_context(
        str(skeleton.get("skeleton_template") or ""),
        slot_id,
    )
    signature = {
        "slot_id": slot_id,
        "slot_kind": slot_kind,
        "semantic_class": semantic_class,
        "source_surface_wz": surface_wz,
        "source_surface_zh": surface_zh,
        "left_fixed_context": left_context,
        "right_fixed_context": right_context,
        "source_gloss_keywords": source_gloss_keywords,
        "disallowed_gloss_keywords": derive_disallowed_gloss_keywords(slot),
    }
    return signature, None


def slot_kind_compatible(term_kind: str, slot_kind: str) -> bool:
    return term_kind == slot_kind


def semantic_class_compatible(term_class: str, slot_class: str) -> bool:
    allowed = STRICT_SEMANTIC_COMPATIBILITY.get(slot_class, {slot_class})
    return term_class in allowed


def term_headword_text(row: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in [
            str(row.get("mandarin_headword") or "").strip(),
            str(row.get("wz_word") or row.get("term") or "").strip(),
        ]
        if part
    )


def positive_gloss_match_score(headword_text: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword and keyword in headword_text)


def contains_disallowed_gloss(headword_text: str, disallowed_keywords: list[str]) -> bool:
    return any(keyword in headword_text for keyword in disallowed_keywords if keyword)


def build_candidate_pool(
    slot: dict[str, Any],
    signature: dict[str, Any],
    skeleton: dict[str, Any],
    lexicon_rows: list[dict[str, Any]],
    pool_size: int,
    used_terms: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_wz = signature["source_surface_wz"]
    scene_id = skeleton.get("scene_id")
    slot_kind = signature["slot_kind"]
    semantic_class = signature["semantic_class"]
    source_keywords = signature["source_gloss_keywords"]
    disallowed_keywords = signature["disallowed_gloss_keywords"]

    ranked: list[tuple[tuple[int, int, int, int, float, str], dict[str, Any]]] = []
    rejected = Counter()
    seen_terms: set[str] = set()

    for row in lexicon_rows:
        if not row.get("usable_as_slot"):
            rejected["not_usable_as_slot"] += 1
            continue
        term = str(row.get("wz_word") or row.get("term") or "").strip()
        if not term or term == source_wz or term in seen_terms or term in used_terms:
            rejected["duplicate_or_source"] += 1
            continue
        if term_looks_locked(row):
            rejected["fixed_expression_core"] += 1
            continue
        if not scene_compatible(row.get("scene_tags") or [], scene_id):
            rejected["scene_mismatch"] += 1
            continue
        row_kind = str(row.get("slot_kind") or "unknown")
        if not slot_kind_compatible(row_kind, slot_kind):
            rejected["slot_kind_mismatch"] += 1
            continue
        row_class = str(row.get("semantic_class") or "unknown")
        if not semantic_class_compatible(row_class, semantic_class):
            rejected["semantic_class_mismatch"] += 1
            continue
        headword_text = term_headword_text(row)
        if contains_disallowed_gloss(headword_text, disallowed_keywords):
            rejected["disallowed_gloss"] += 1
            continue
        positive_match = positive_gloss_match_score(headword_text, source_keywords)
        if positive_match <= 0:
            rejected["gloss_keyword_mismatch"] += 1
            continue
        score_key = (
            positive_match,
            int(row_class == semantic_class),
            int(row_kind == slot_kind),
            int(row.get("primary_scene_id") == scene_id),
            round(float(row.get("confidence", 0.0)), 4),
            term,
        )
        ranked.append((score_key, row))
        seen_terms.add(term)

    ranked.sort(key=lambda item: item[0], reverse=True)
    pool: list[dict[str, Any]] = []
    for _, row in ranked:
        term = str(row.get("wz_word") or row.get("term") or "").strip()
        if term in used_terms:
            continue
        used_terms.add(term)
        pool.append(
            {
                "term": term,
                "mandarin_headword": row.get("mandarin_headword"),
                "semantic_class": row.get("semantic_class", "unknown"),
                "slot_kind": row.get("slot_kind", "unknown"),
                "source_file": row.get("source_file"),
                "record_type": row.get("record_type"),
                "confidence": round(float(row.get("confidence", 0.0)), 4),
            }
        )
        if len(pool) >= pool_size:
            break

    note = {
        "source_gloss_keywords": source_keywords,
        "disallowed_gloss_keywords": disallowed_keywords,
        "selected_terms": [item["term"] for item in pool],
        "rejected_reason_counts": dict(rejected),
    }
    return pool, note


def pool_jaccard(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    left_terms = {item.get("term") for item in left if item.get("term")}
    right_terms = {item.get("term") for item in right if item.get("term")}
    if not left_terms and not right_terms:
        return 0.0
    return round(len(left_terms & right_terms) / max(len(left_terms | right_terms), 1), 4)


def enumerate_plans(
    ordered_slot_ids: list[str],
    slot_candidate_pools: dict[str, list[dict[str, Any]]],
    max_plans: int,
) -> list[dict[str, Any]]:
    pool_term_lists = [
        [item["term"] for item in slot_candidate_pools.get(slot_id, [])]
        for slot_id in ordered_slot_ids
    ]
    plans: list[dict[str, Any]] = []
    if not ordered_slot_ids or not all(pool_term_lists):
        return plans
    for index, combo in enumerate(itertools.product(*pool_term_lists), start=1):
        if len(set(combo)) != len(combo):
            continue
        plans.append(
            {
                "plan_id": f"plan_{index:04d}",
                "slot_values": {
                    slot_id: term for slot_id, term in zip(ordered_slot_ids, combo)
                },
                "required_slot_replacements": ordered_slot_ids,
                "target_slot_changes": len(ordered_slot_ids),
            }
        )
        if len(plans) >= max_plans:
            break
    return plans


def build_prompt_payload_from_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_type": "dense_slot_replacement",
        "scene_id": task.get("scene_id"),
        "scene_label": task.get("scene_label"),
        "speech_function": task.get("primary_speech_function"),
        "source_wz_sentence": task.get("source_wz_sentence"),
        "source_zh_sentence": task.get("source_zh_sentence"),
        "skeleton_template": task.get("skeleton_template"),
        "fixed_chunks": task.get("fixed_chunks") or [],
        "eligible_slot_ids": task.get("eligible_slot_ids") or [],
        "replace_all_eligible_content_slots": True,
        "slot_candidate_pools": task.get("slot_candidate_pools") or {},
        "replacement_plans": task.get("replacement_plans") or [],
        "output_schema": {
            "candidate_sentences": [
                {
                    "wz_sentence": "温州话句子",
                    "slot_values": {"SLOT_X": "chosen term"},
                    "plan_id": "plan_0001",
                    "naturalness_note": "short Chinese note",
                }
            ]
        },
    }


def rebuild_task_prompt_bundle(task: dict[str, Any]) -> dict[str, Any]:
    prompt_payload = build_prompt_payload_from_task(task)
    task["generation_count"] = len(task.get("replacement_plans") or [])
    task["prompt_bundle"] = {
        "system_prompt": SYSTEM_PROMPT,
        "user_payload": prompt_payload,
        "user_prompt": build_user_prompt(prompt_payload),
    }
    return task


def build_dense_task(
    skeleton: dict[str, Any],
    lexicon_rows: list[dict[str, Any]],
    max_plans_per_skeleton: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    sentence_length = len(str(skeleton.get("source_wz_sentence") or ""))
    target_count = target_eligible_slot_count(skeleton)
    candidate_slots = rank_eligible_slots(
        [slot for slot in skeleton.get("slots") or [] if slot_is_eligible(slot, skeleton)]
    )

    slot_signatures: dict[str, dict[str, Any]] = {}
    slot_candidate_pools: dict[str, list[dict[str, Any]]] = {}
    slot_target_pool_sizes: dict[str, int] = {}
    slot_pool_build_notes: dict[str, Any] = {}
    selected_slots: list[dict[str, Any]] = []
    locked_slot_reasons: dict[str, str] = {}
    task_build_issues: list[str] = []
    used_terms: set[str] = set()

    for slot in candidate_slots:
        if len(selected_slots) >= target_count:
            break
        signature, lock_reason = build_slot_signature(slot, skeleton)
        if lock_reason:
            locked_slot_reasons[slot["slot_id"]] = lock_reason
            task_build_issues.append(lock_reason)
            continue
        slot_signatures[slot["slot_id"]] = signature
        target_pool_size = target_pool_size_for_slot(slot, sentence_length)
        pool, note = build_candidate_pool(
            slot,
            signature,
            skeleton,
            lexicon_rows,
            pool_size=target_pool_size,
            used_terms=used_terms,
        )
        slot_pool_build_notes[slot["slot_id"]] = note
        if len(pool) < 3:
            locked_slot_reasons[slot["slot_id"]] = "insufficient_candidates"
            task_build_issues.append("insufficient_candidates")
            continue
        selected_slots.append(slot)
        slot_candidate_pools[slot["slot_id"]] = pool
        slot_target_pool_sizes[slot["slot_id"]] = target_pool_size

    if len(selected_slots) < 2:
        return None, {
            "task_build_decision": "insufficient_dense_support",
            "task_build_issues": task_build_issues or ["insufficient_dense_support"],
            "locked_slot_reasons": locked_slot_reasons,
            "pool_collision_pairs": [],
        }

    pool_collision_pairs: list[dict[str, Any]] = []
    ordered_slot_ids = [slot["slot_id"] for slot in selected_slots]
    for left_index, left_slot_id in enumerate(ordered_slot_ids):
        for right_slot_id in ordered_slot_ids[left_index + 1 :]:
            score = pool_jaccard(
                slot_candidate_pools.get(left_slot_id, []),
                slot_candidate_pools.get(right_slot_id, []),
            )
            if score > 0.34:
                pool_collision_pairs.append(
                    {
                        "slot_ids": [left_slot_id, right_slot_id],
                        "jaccard": score,
                    }
                )

    if pool_collision_pairs:
        weaker_slot_id = min(
            (slot["slot_id"] for slot in selected_slots),
            key=lambda slot_id: (
                slot_rank_key(next(slot for slot in selected_slots if slot["slot_id"] == slot_id)),
                slot_id,
            ),
        )
        locked_slot_reasons[weaker_slot_id] = "pool_collision"
        task_build_issues.append("pool_collision")
        selected_slots = [slot for slot in selected_slots if slot["slot_id"] != weaker_slot_id]
        slot_candidate_pools.pop(weaker_slot_id, None)
        slot_target_pool_sizes.pop(weaker_slot_id, None)
        slot_signatures.pop(weaker_slot_id, None)

    ordered_slot_ids = [slot["slot_id"] for slot in selected_slots]
    replacement_plans = enumerate_plans(
        ordered_slot_ids,
        slot_candidate_pools,
        max_plans=max_plans_per_skeleton,
    )
    if len(ordered_slot_ids) < 2 or len(replacement_plans) < 6:
        issue = "insufficient_dense_support"
        if len(replacement_plans) < 6:
            task_build_issues.append("too_few_replacement_plans")
        return None, {
            "task_build_decision": issue,
            "task_build_issues": task_build_issues or [issue],
            "locked_slot_reasons": locked_slot_reasons,
            "pool_collision_pairs": pool_collision_pairs,
        }

    task = {
        "task_id": f"dense::{skeleton['skeleton_id']}",
        "task_type": "dense_slot_replacement",
        "task_build_decision": "ready_for_generation",
        "task_build_issues": sorted(set(task_build_issues)),
        "skeleton_id": skeleton["skeleton_id"],
        "scene_id": skeleton.get("scene_id"),
        "scene_label": skeleton.get("scene_label"),
        "primary_speech_function": skeleton.get("primary_speech_function"),
        "source_wz_sentence": skeleton.get("source_wz_sentence"),
        "source_zh_sentence": skeleton.get("source_zh_sentence"),
        "skeleton_template": skeleton.get("skeleton_template"),
        "fixed_chunks": skeleton.get("fixed_chunks") or [],
        "slots": skeleton.get("slots") or [],
        "slot_signatures": slot_signatures,
        "slot_candidate_pools": slot_candidate_pools,
        "slot_pool_build_notes": slot_pool_build_notes,
        "locked_slot_reasons": locked_slot_reasons,
        "pool_collision_pairs": pool_collision_pairs,
        "eligible_slot_ids": ordered_slot_ids,
        "eligible_slot_count_target": target_count,
        "slot_target_pool_sizes": slot_target_pool_sizes,
        "required_slot_replacements": ordered_slot_ids,
        "replace_all_eligible_slots": True,
        "replacement_plans": replacement_plans,
        "generation_count": len(replacement_plans),
        "llm_dense_tier": skeleton.get("llm_dense_tier"),
        "llm_dense_note": skeleton.get("llm_dense_note"),
        "dense_combination_upper_bound_rule": skeleton.get("dense_combination_upper_bound", 0),
    }
    return rebuild_task_prompt_bundle(task), {
        "task_build_decision": "ready_for_generation",
        "task_build_issues": sorted(set(task_build_issues)),
        "locked_slot_reasons": locked_slot_reasons,
        "pool_collision_pairs": pool_collision_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dense slot replacement pilot tasks from reviewed skeletons."
    )
    parser.add_argument("--skeletons", type=Path, default=DEFAULT_SKELETONS)
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-plans-per-skeleton", type=int, default=16)
    args = parser.parse_args()

    skeleton_rows = load_jsonl(args.skeletons)
    lexicon_rows = load_jsonl(args.lexicon)

    filtered_skeletons = [
        row
        for row in skeleton_rows
        if row.get("llm_dense_pass") and row.get("llm_dense_tier") in {"dense_reusable", "medium_reusable"}
    ]
    filtered_skeletons.sort(
        key=lambda row: (
            float(row.get("dense_reuse_rule_score", 0.0)),
            float(row.get("skeleton_quality_score", 0.0)),
            int(row.get("dense_combination_upper_bound", 0)),
            row.get("skeleton_id", ""),
        ),
        reverse=True,
    )
    if args.limit > 0:
        filtered_skeletons = filtered_skeletons[: args.limit]

    task_rows: list[dict[str, Any]] = []
    skipped_reasons = Counter()
    locked_slot_reason_counts = Counter()
    pool_collision_task_count = 0
    insufficient_dense_support_count = 0

    for skeleton in filtered_skeletons:
        task, meta = build_dense_task(
            skeleton,
            lexicon_rows,
            max_plans_per_skeleton=args.max_plans_per_skeleton,
        )
        for reason in (meta.get("locked_slot_reasons") or {}).values():
            locked_slot_reason_counts[reason] += 1
        if meta.get("pool_collision_pairs"):
            pool_collision_task_count += 1
        if meta.get("task_build_decision") == "insufficient_dense_support":
            insufficient_dense_support_count += 1
            for issue in meta.get("task_build_issues") or ["insufficient_dense_support"]:
                skipped_reasons[str(issue)] += 1
        if task is None:
            continue
        task_rows.append(task)

    write_jsonl(args.output, task_rows)
    write_json(
        args.summary,
        {
            "input_reviewed_skeletons": len(skeleton_rows),
            "llm_dense_reusable_input": len(filtered_skeletons),
            "dense_task_count": len(task_rows),
            "scene_counts": dict(Counter(row["scene_id"] for row in task_rows)),
            "eligible_slot_count_distribution": dict(Counter(len(row["eligible_slot_ids"]) for row in task_rows)),
            "eligible_slot_target_distribution": dict(Counter(row["eligible_slot_count_target"] for row in task_rows)),
            "slot_target_pool_size_distribution": dict(
                Counter(
                    size
                    for row in task_rows
                    for size in (row.get("slot_target_pool_sizes") or {}).values()
                )
            ),
            "replacement_plan_count_distribution": dict(Counter(len(row["replacement_plans"]) for row in task_rows)),
            "total_replacement_plans": sum(len(row["replacement_plans"]) for row in task_rows),
            "locked_slot_reason_counts": dict(locked_slot_reason_counts),
            "pool_collision_task_count": pool_collision_task_count,
            "insufficient_dense_support_count": insufficient_dense_support_count,
            "skipped_reasons": dict(skipped_reasons),
            "output_path": str(args.output),
        },
    )

    print(f"Reviewed skeleton rows: {len(skeleton_rows)}")
    print(f"LLM dense reusable input rows: {len(filtered_skeletons)}")
    print(f"Dense pilot tasks: {len(task_rows)}")
    print(f"Total replacement plans: {sum(len(row['replacement_plans']) for row in task_rows)}")
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
