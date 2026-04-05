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
import math
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

from build_controlled_generation_assets import SCENE_CATALOG, SEMANTIC_CLASS_HINTS, build_scene_payload
from generate_controlled_sentences import build_client
from wz_pipeline.contracts import apply_contract
from wz_pipeline.dialect import ACTIVE_DIALECT_CONFIG
from wz_pipeline.domain_config import build_domain_context, normalize_domain_ids
from wz_pipeline.failure_taxonomy import classify_rule_fail_buckets, classify_rule_reason_bucket, reason_key
from wz_pipeline.grammar_config import load_speech_act_types
from wz_pipeline.grammar_guardrails import (
    build_generation_grammar_prompt_rules,
    build_generation_grammar_user_rules,
    build_grammar_repair_system_prompt,
    build_grammar_repair_user_prompt,
    grammar_validation_reasons,
    should_attempt_grammar_repair,
)
from wz_pipeline.paths import DATA_DIR, GRAMMAR_SPEC_PATH
from wz_pipeline.grammar_spec import relevant_spec_labels
from wz_pipeline.jsonl import read_jsonl, write_jsonl
from wz_pipeline.registry import register_run
from wz_pipeline.review import export_review_tsv
from wz_pipeline.runs import build_summary, make_run_id, prepare_run_layout, write_json
from wz_pipeline.scene_policy import DEFAULT_SCENES, FOCUS_SCENES, MODERN_SIDECAR_SCENES, PRIORITY_SCENES
from wz_pipeline.source_surface_guardrails import detect_unsupported_surface_terms
from wz_pipeline.task_feedback import build_feedback_plan, load_feedback_summary

# ---- paths ----
CLEANED_RECORDS = DATA_DIR / "cleaned" / "cleaned_records_primary.jsonl"
EXTRACTED_SHORT = DATA_DIR / "extracted_training_sentences" / "short_8_20.jsonl"
EXTRACTED_LONG = DATA_DIR / "extracted_training_sentences" / "long_20_30.jsonl"
OUT_DIR = DATA_DIR / "generated_long_sentences" / "fewshot_batch"
REPLACEABLE_LEXICON = DATA_DIR / "controlled_generation" / "assets" / "replaceable_lexicon.jsonl"
DENSE_WHITELIST = DATA_DIR / "controlled_generation" / "assets" / "dense_slot_lexicon_whitelist.jsonl"
EXTERNAL_TERM_CATALOG = DATA_DIR / "controlled_generation" / "assets" / "external_term_catalog.json"
DENSE_SCENE_SEEDS = DATA_DIR / "controlled_generation" / "assets" / "dense_scene_seed_terms.json"
FEWSHOT_CORE_POLICY = DATA_DIR / "controlled_generation" / "assets" / "fewshot_core_policy.json"
MODERN_ANCHOR_CANDIDATES = DATA_DIR / "generated_long_sentences" / "reports" / "modern_anchor_candidates.json"
MODERN_ANCHOR_EXAMPLES = DATA_DIR / "generated_long_sentences" / "reports" / "modern_anchor_examples.jsonl"
PIPELINE_NAME = "fewshot_batch"

PAREN_RE = re.compile(r"[（(][^）)]{1,4}[）)]")
TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
DIALECT_SETTINGS = ACTIVE_DIALECT_CONFIG.raw
SENTENCE_LENGTH = DIALECT_SETTINGS.get("sentence_length") or {}
MIN_SENTENCE_LENGTH = int(SENTENCE_LENGTH.get("min") or 20)
MAX_SENTENCE_LENGTH = int(SENTENCE_LENGTH.get("max") or 30)
DIALECT_NAME = ACTIVE_DIALECT_CONFIG.dialect_name
STANDARD_LANGUAGE_LABEL = ACTIVE_DIALECT_CONFIG.standard_language_label

BAD_DEFINITION_HINTS = tuple(DIALECT_SETTINGS.get("bad_definition_hints") or (
    "后置于", "前置于", "用于", "用在", "俗语", "助词", "量词", "副词", "介词", "连词", "语气词", "代词", "作副词", "在动词后",
))
NON_DAILY_DEFINITION_HINTS = tuple(DIALECT_SETTINGS.get("non_daily_definition_hints") or (
    "比喻", "借指", "引申", "戏称", "绰号", "一种舞", "一类人",
))
ARCHAIC_DEFINITION_HINTS = tuple(DIALECT_SETTINGS.get("archaic_definition_hints") or (
    "旧俗", "旧时", "旧称", "古代", "古时", "旧式", "老式", "银圆", "酒筵", "传统木结构", "旧社会",
))
ARCHAIC_ALLOWED_HINTS = tuple(DIALECT_SETTINGS.get("archaic_allowed_hints") or ("今指", "现在也指", "现指"))
ABSTRACT_TIME_HINTS = tuple(DIALECT_SETTINGS.get("abstract_time_hints") or (
    "明后天", "明天", "后天", "昨天", "今天", "以前", "以后", "现在", "刚才", "过去", "过些日子",
))
STOPWORDS = set(DIALECT_SETTINGS.get("stopwords") or (
    "一个", "一种", "一件", "这个", "那个", "这里", "那里", "什么", "事情", "东西", "个", "了", "的", "着", "过", "在", "是", "有", "很", "真",
))
POLICY_VERSION = "track_b_mainline_v4_2026-04-05"
SCENE_TERMS = DIALECT_SETTINGS.get("scene_terms") or {}
DIGITAL_PARENT_SCENE = "digital_chat"
DIGITAL_SUBSCENES = (
    "digital_ai_assistant",
    "digital_messaging_call",
    "digital_device_trouble",
    "digital_app_operation",
)
SCENE_PARENT_OVERRIDES = {
    scene_id: DIGITAL_PARENT_SCENE
    for scene_id in DIGITAL_SUBSCENES
}
SCENE_TURN_ROLE_OVERRIDES = {
    scene_id: "user_query"
    for scene_id in DIGITAL_SUBSCENES
}


def _scene_term_config(scene_id: str) -> dict[str, Any]:
    payload = SCENE_TERMS.get(scene_id) or {}
    if not payload:
        parent_scene = SCENE_PARENT_OVERRIDES.get(scene_id, "")
        payload = SCENE_TERMS.get(parent_scene) or {}
    return payload if isinstance(payload, dict) else {}


def _scene_term_set(scene_id: str, field: str) -> set[str]:
    values = _scene_term_config(scene_id).get(field)
    if not isinstance(values, (list, tuple, set)):
        return set()
    return {str(value).strip() for value in values if str(value).strip()}


def _scene_support_pool_size(scene_id: str, default: int) -> int:
    value = _scene_term_config(scene_id).get("support_pool_size")
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


SHOPPING_OLD_MONEY_TERMS = _scene_term_set("shopping_payment", "old_money_terms")
SHOPPING_PROMPT_BLOCKED_TERMS = _scene_term_set("shopping_payment", "prompt_blocked_terms")
SHOPPING_AMOUNT_TERMS = _scene_term_set("shopping_payment", "amount_terms")
FOOD_SUPPORT_PRIORITY_TERMS = _scene_term_set("food_dining", "support_priority_terms")
FOOD_BEVERAGE_TERMS = _scene_term_set("food_dining", "beverage_terms")
FOOD_STAPLE_TERMS = _scene_term_set("food_dining", "staple_terms")
FOOD_ACTION_TERMS = _scene_term_set("food_dining", "action_terms")
HOME_SUPPORT_PRIORITY_TERMS = _scene_term_set("home_life", "support_priority_terms")
SHOPPING_SUPPORT_PRIORITY_TERMS = _scene_term_set("shopping_payment", "support_priority_terms")
TRANSPORT_SUPPORT_PRIORITY_TERMS = _scene_term_set("transport_trip", "support_priority_terms")
WEATHER_SUPPORT_PRIORITY_TERMS = _scene_term_set("weather_safety", "support_priority_terms")
SCENE_SUPPORT_POOL_SIZES = {
    "food_dining": _scene_support_pool_size("food_dining", 10),
    "home_life": _scene_support_pool_size("home_life", 12),
    "shopping_payment": _scene_support_pool_size("shopping_payment", 12),
    "transport_trip": _scene_support_pool_size("transport_trip", 12),
    "weather_safety": _scene_support_pool_size("weather_safety", 12),
    "health_medical": _scene_support_pool_size("health_medical", 8),
    "digital_chat": _scene_support_pool_size("digital_chat", 8),
    "digital_ai_assistant": _scene_support_pool_size("digital_ai_assistant", 8),
    "digital_messaging_call": _scene_support_pool_size("digital_messaging_call", 8),
    "digital_device_trouble": _scene_support_pool_size("digital_device_trouble", 8),
    "digital_app_operation": _scene_support_pool_size("digital_app_operation", 8),
    "work_study": _scene_support_pool_size("work_study", 8),
}

TASK_BALANCE_SETTINGS = DIALECT_SETTINGS.get("task_balance") or {}


def _task_balance_int(field: str, default: int) -> int:
    value = TASK_BALANCE_SETTINGS.get(field)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _task_balance_float(field: str, default: float) -> float:
    value = TASK_BALANCE_SETTINGS.get(field)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


MAX_PROMPT_VARIANTS_PER_WORD_COMBO = max(1, _task_balance_int("max_prompt_variants_per_word_combo", 1))
CORE_USAGE_CAP_RATIO = min(max(_task_balance_float("core_usage_cap_ratio", 0.35), 0.0), 1.0)
PLACE_SUPPORT_RATIO = min(max(_task_balance_float("place_support_ratio", 0.5), 0.0), 1.0)
PLACE_SUPPORT_CAP_RATIO = min(max(_task_balance_float("place_support_cap_ratio", 0.15), 0.0), 1.0)
FOCUS_SCENE_REPLACEABLE_CAP = max(0, _task_balance_int("focus_scene_replaceable_cap", 48))
PER_SEMANTIC_CLASS_CAP = max(1, _task_balance_int("per_semantic_class_cap", 12))
FOCUS_REPLACEABLE_ALLOWED_CATEGORIES = {"noun", "action", "adjective", "device", "amount", "transport"}
CORE_SOURCE_PRIORITY = {
    "anchor_example": 5,
    "prompt_example": 4,
    "scene_seed": 4,
    "example_mined": 3,
    "dense_whitelist": 2,
    "replaceable_lexicon": 1,
}


def domain_term_priority(
    surface: str,
    *,
    definition: str = "",
    domain_context: dict[str, Any] | None = None,
) -> int:
    context = domain_context or {}
    required_terms = set(context.get("required_terms") or [])
    preferred_terms = set(context.get("preferred_terms") or [])
    clean_surface = str(surface or "").strip()
    clean_definition = str(definition or "").strip()
    if clean_surface in required_terms:
        return 3
    if clean_surface in preferred_terms:
        return 2
    if clean_definition and any(term and term in clean_definition for term in required_terms):
        return 2
    if clean_definition and any(term and term in clean_definition for term in preferred_terms):
        return 1
    return 0


def domain_text_score(domain_context: dict[str, Any] | None, *texts: str | None) -> int:
    context = domain_context or {}
    required_terms = set(context.get("required_terms") or [])
    preferred_terms = set(context.get("preferred_terms") or [])
    if not required_terms and not preferred_terms:
        return 0
    merged = " ".join(str(text or "") for text in texts if text)
    score = 0
    for term in required_terms:
        if term and term in merged:
            score += 3
    for term in preferred_terms:
        if term and term in merged:
            score += 1
    return score


def task_word_combo_signature(
    scene_id: str,
    core_word_surface: str,
    support_words: list[dict[str, Any]] | list[str],
) -> tuple[str, str, frozenset[str]]:
    normalized_support_words: set[str] = set()
    for word in support_words:
        if isinstance(word, dict):
            surface = str(word.get("wz_word") or "").strip()
        else:
            surface = str(word).strip()
        if surface:
            normalized_support_words.add(surface)
    return (
        scene_id,
        core_word_surface,
        frozenset(normalized_support_words),
    )


def task_prompt_signature(
    scene_id: str,
    core_word_surface: str,
    support_words: list[dict[str, Any]] | list[str],
    examples: list[dict[str, Any]],
    speech_acts: list[dict[str, Any]] | list[str] | None = None,
) -> tuple[str, str, frozenset[str], tuple[str, ...], tuple[str, ...]]:
    prompt_context = tuple(
        clean_wz(str(example.get("wz_sentence") or "")) or str(example.get("wz_word") or "").strip()
        for example in examples[:4]
    )
    speech_act_ids = normalized_speech_act_ids(speech_acts)
    return task_word_combo_signature(scene_id, core_word_surface, support_words) + (prompt_context, speech_act_ids)


def normalized_speech_act_ids(
    speech_acts: list[dict[str, Any]] | list[str] | None,
) -> tuple[str, ...]:
    normalized: list[str] = []
    for act in speech_acts or []:
        if isinstance(act, dict):
            act_id = str(act.get("id") or "").strip()
        else:
            act_id = str(act).strip()
        if act_id:
            normalized.append(act_id)
    return tuple(normalized)


def speech_act_window(
    all_speech_acts: list[dict[str, str]],
    offset: int,
    *,
    window_size: int = 3,
) -> list[dict[str, str]]:
    if not all_speech_acts:
        return []
    n_acts = len(all_speech_acts)
    bounded_window = max(1, min(window_size, n_acts))
    return [all_speech_acts[(offset + i) % n_acts] for i in range(bounded_window)]


def ordered_speech_act_offsets(
    all_speech_acts: list[dict[str, str]],
    global_offset_counts: Counter[int] | None,
    scene_offset_counts: Counter[int] | None = None,
) -> list[int]:
    if not all_speech_acts:
        return [0]
    global_counts = global_offset_counts or Counter()
    local_counts = scene_offset_counts or Counter()
    return sorted(
        range(len(all_speech_acts)),
        key=lambda offset: (global_counts[offset], local_counts[offset], offset),
    )


def scene_has_prompt_material(
    scene_id: str,
    examples_by_scene: dict[str, list[dict[str, Any]]],
    words_by_scene: dict[str, list[dict[str, Any]]],
) -> bool:
    del words_by_scene
    return len(examples_by_scene.get(scene_id, [])) >= 3


def default_turn_role(scene_id: str) -> str:
    return SCENE_TURN_ROLE_OVERRIDES.get(scene_id, "speaker_utterance")


def digital_subscene_scores(*texts: str | None) -> list[tuple[str, int]]:
    return [
        (scene_id, scene_match_score(scene_id, *texts))
        for scene_id in DIGITAL_SUBSCENES
    ]


def expand_scene_targets(
    scene_ids: list[str] | tuple[str, ...],
    *texts: str | None,
    expand_digital: bool = False,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_scene in scene_ids:
        scene_id = str(raw_scene or "").strip()
        if not scene_id:
            continue
        targets = [scene_id]
        if scene_id == DIGITAL_PARENT_SCENE and expand_digital:
            scored = digital_subscene_scores(*texts)
            positives = [subscene for subscene, score in scored if score > 0]
            targets = [scene_id] + (positives or list(DIGITAL_SUBSCENES))
        for target in targets:
            if target and target not in seen:
                seen.add(target)
                ordered.append(target)
    return ordered


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


def normalized_focus_replaceable_category(word: dict[str, Any]) -> str:
    semantic_class = str(word.get("semantic_class") or "").strip()
    slot_kind = str(word.get("slot_kind") or "").strip()
    if semantic_class == "action" or slot_kind == "verb":
        return "action"
    if semantic_class == "adjective" or slot_kind == "adjective":
        return "adjective"
    if semantic_class == "device":
        return "device"
    if semantic_class == "amount":
        return "amount"
    if semantic_class == "transport":
        return "transport"
    if slot_kind == "noun":
        return "noun"
    return ""


def word_is_place_like(word: dict[str, Any]) -> bool:
    return (
        str(word.get("slot_kind") or "").strip() == "place"
        or str(word.get("semantic_class") or "").strip() == "place"
    )


def core_source_priority(word: dict[str, Any]) -> int:
    return CORE_SOURCE_PRIORITY.get(str(word.get("source_type") or "").strip(), 0)


def merge_word_candidate(
    by_surface: dict[str, dict[str, Any]],
    candidate: dict[str, Any],
) -> None:
    surface = str(candidate.get("wz_word") or "").strip()
    if not surface:
        return
    existing = by_surface.get(surface)
    if existing is None:
        by_surface[surface] = dict(candidate)
        return
    candidate_rank = (
        core_source_priority(candidate),
        int(candidate.get("modern_priority", 0)),
        str(candidate.get("definition") or ""),
    )
    existing_rank = (
        core_source_priority(existing),
        int(existing.get("modern_priority", 0)),
        str(existing.get("definition") or ""),
    )
    if candidate_rank > existing_rank:
        by_surface[surface] = dict(candidate)


def external_place_target_scenes(row: dict[str, Any]) -> list[str]:
    requested = normalize_scene_list(",".join(str(item) for item in (row.get("scene_tags") or [])))
    ordered: list[str] = []
    for scene in ("transport_trip", "digital_messaging_call"):
        if scene == "transport_trip":
            ordered.append(scene)
            continue
        if requested and "transport_trip" not in requested and scene not in requested:
            continue
        ordered.append(scene)
    return ordered


def task_prefers_place_support(
    scene_id: str,
    anchor: dict[str, Any],
    examples: list[dict[str, Any]],
) -> bool:
    if scene_id == "transport_trip":
        return True
    if scene_id != "digital_messaging_call":
        return False
    score_inputs = [
        anchor.get("wz_word", ""),
        anchor.get("definition", ""),
        anchor.get("zh_sentence", ""),
        anchor.get("wz_sentence", ""),
    ]
    for example in examples:
        score_inputs.extend([example.get("zh_sentence", ""), example.get("wz_sentence", "")])
    return scene_match_score("transport_trip", *score_inputs) > 0


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
    scene_policies = CORE_POLICY.get("scene_policies", {})
    if scene_id in scene_policies:
        return scene_policies.get(scene_id, {})
    parent_scene = SCENE_PARENT_OVERRIDES.get(scene_id, "")
    if parent_scene:
        return scene_policies.get(parent_scene, {})
    return {}


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


def classify_sentence_type(wz: str) -> str:
    """Heuristic sentence-type classifier for diversity-aware example selection."""
    wz_clean = wz.strip()
    if wz_clean.endswith("\uff1f") or wz_clean.endswith("?"):
        return "question"
    if wz_clean.endswith("\uff01") or wz_clean.endswith("!"):
        return "exclamation"
    if wz_clean.startswith("\u4f60") and any(m in wz_clean[:8] for m in ("\u8985", "\u6162\u6162", "\U000279df")):
        return "imperative"
    return "declarative"


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
    if len(clean_examples) > 4:
        by_type: dict[str, list[dict[str, Any]]] = {}
        for ex in clean_examples:
            t = classify_sentence_type(str(ex.get("wz_sentence") or ""))
            by_type.setdefault(t, []).append(ex)
        diverse_selection: list[dict[str, Any]] = []
        for t in ("question", "exclamation", "imperative", "declarative"):
            if t in by_type and by_type[t]:
                diverse_selection.append(by_type[t][0])
        seen_ids = {id(ex) for ex in diverse_selection}
        for ex in clean_examples:
            if id(ex) not in seen_ids and len(diverse_selection) < 4:
                diverse_selection.append(ex)
                seen_ids.add(id(ex))
        return diverse_selection[:4], False
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
            target_scenes = expand_scene_targets(
                [scene],
                row.get("wz_sentence", ""),
                row.get("zh_sentence", ""),
                row.get("wz_word", ""),
                row.get("definition", ""),
                expand_digital=True,
            )
            for target_scene in target_scenes:
                row_copy = dict(row)
                row_copy["scene_id"] = target_scene
                by_scene[target_scene].append(row_copy)
    if MODERN_ANCHOR_EXAMPLES.exists():
        with open(MODERN_ANCHOR_EXAMPLES, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                scene = str(row.get("scene_id") or "").strip()
                if not scene:
                    continue
                target_scenes = expand_scene_targets(
                    [scene],
                    row.get("wz_sentence", ""),
                    row.get("zh_sentence", ""),
                    row.get("wz_word", ""),
                    row.get("definition", ""),
                    expand_digital=True,
                )
                for target_scene in target_scenes:
                    by_scene[target_scene].append(
                        {
                            "scene_id": target_scene,
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
        target_scenes = expand_scene_targets(
            row.get("scene_tags") or [row.get("primary_topic_scene") or "daily_chat"],
            row.get("term", ""),
            row.get("definition", ""),
            *(row.get("evidence_examples") or []),
            expand_digital=True,
        )
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
        target_scenes = external_place_target_scenes(row)
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
    focus_scene_replaceable_counts: Counter[str] = Counter()
    focus_scene_category_counts: dict[str, Counter[str]] = defaultdict(Counter)

    load_modern_words_from_assets(by_scene, seen)

    if DENSE_WHITELIST.exists():
        with open(DENSE_WHITELIST, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                w = row.get("wz_word") or row.get("term") or ""
                d = row.get("mandarin_headword", "") or row.get("definition", "")
                target_scenes = expand_scene_targets(
                    [row.get("dense_whitelist_scene_id") or row.get("primary_scene_id") or "daily_chat"],
                    w,
                    d,
                    expand_digital=True,
                )
                for scene in target_scenes:
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
                target_scenes = expand_scene_targets(
                    [choose_scene_for_word(row)],
                    w,
                    d,
                    row.get("zh_sentence", ""),
                    expand_digital=True,
                )
                for scene in target_scenes:
                    key = (scene, w)
                    if w and 2 <= len(w) <= 4 and d and key not in seen:
                        semantic_class = str(row.get("semantic_class") or "unknown")
                        slot_kind = str(row.get("slot_kind") or "unknown")
                        candidate = {
                            "wz_word": w,
                            "definition": d[:40],
                            "scene_id": scene,
                            "semantic_class": semantic_class,
                            "slot_kind": slot_kind,
                            "source_type": "replaceable_lexicon",
                            "is_modern": False,
                            "modern_priority": 0,
                        }
                        if scene in FOCUS_SCENES:
                            if word_is_place_like(candidate):
                                continue
                            category = normalized_focus_replaceable_category(candidate)
                            if category not in FOCUS_REPLACEABLE_ALLOWED_CATEGORIES:
                                continue
                            if (
                                FOCUS_SCENE_REPLACEABLE_CAP > 0
                                and focus_scene_replaceable_counts[scene] >= FOCUS_SCENE_REPLACEABLE_CAP
                            ):
                                continue
                            if focus_scene_category_counts[scene][category] >= PER_SEMANTIC_CLASS_CAP:
                                continue
                        seen.add(key)
                        by_scene[scene].append(candidate)
                        if scene in FOCUS_SCENES:
                            focus_scene_replaceable_counts[scene] += 1
                            focus_scene_category_counts[scene][category] += 1

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
SCENE_COMMUNICATIVE_FUNCTIONS: dict[str, list[str]] = {
    scene["scene_id"]: list(scene.get("communicative_functions") or [])
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
    score = sum(1 for keyword in SCENE_KEYWORDS.get(scene_id, []) if keyword in merged)
    if score > 0:
        return score
    parent_scene = SCENE_PARENT_OVERRIDES.get(scene_id, "")
    if parent_scene:
        parent_score = sum(1 for keyword in SCENE_KEYWORDS.get(parent_scene, []) if keyword in merged)
        if parent_score > 0:
            return 1
    return 0


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
    # Place names (地名) get a general boost across all scenes
    if slot_kind == "place" and str(word.get("source_type") or "") in ("place_name", "example_mined"):
        score += 2
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


def prioritize_anchor_candidates(
    anchor_candidates: list[dict[str, Any]],
    *,
    domain_context: dict[str, Any] | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not anchor_candidates:
        return []
    scored = [
        (
            (
                domain_text_score(
                    domain_context,
                    example.get("wz_word", ""),
                    example.get("definition", ""),
                    example.get("zh_sentence", ""),
                    example.get("wz_sentence", ""),
                ),
                scene_match_score(
                    str(example.get("scene_id") or ""),
                    example.get("wz_word", ""),
                    example.get("definition", ""),
                    example.get("zh_sentence", ""),
                    example.get("wz_sentence", ""),
                ),
                str(example.get("wz_word") or ""),
            ),
            example,
        )
        for example in anchor_candidates
    ]
    rng.shuffle(scored)
    scored.sort(key=lambda item: item[0], reverse=True)
    return [example for _, example in scored]


def allocate_scene_quotas(
    viable_scenes: list[str],
    num_tasks: int,
    *,
    rng: random.Random,
    scene_weights: dict[str, float] | None = None,
) -> dict[str, int]:
    if not viable_scenes or num_tasks <= 0:
        return {}
    ordered_scenes = list(viable_scenes)
    rng.shuffle(ordered_scenes)
    normalized_weights: dict[str, float] = {}
    for scene in ordered_scenes:
        raw_weight = (scene_weights or {}).get(scene, 1.0)
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            weight = 1.0
        normalized_weights[scene] = max(weight, 0.0)
    if not any(weight > 0 for weight in normalized_weights.values()):
        normalized_weights = {scene: 1.0 for scene in ordered_scenes}
    total_weight = sum(normalized_weights.values())
    current_weights = {scene: 0.0 for scene in ordered_scenes}
    quotas: dict[str, int] = defaultdict(int)
    for _ in range(num_tasks):
        for scene in ordered_scenes:
            current_weights[scene] += normalized_weights[scene]
        chosen_scene = max(ordered_scenes, key=lambda scene: current_weights[scene])
        quotas[chosen_scene] += 1
        current_weights[chosen_scene] -= total_weight
    # Enforce minimum floor: each scene gets at least ~8% of total tasks
    if len(ordered_scenes) > 1 and num_tasks >= len(ordered_scenes) * 2:
        min_per_scene = max(1, num_tasks // (len(ordered_scenes) * 2))
        deficit_scenes = [s for s in ordered_scenes if quotas[s] < min_per_scene]
        surplus_scenes = sorted(
            [s for s in ordered_scenes if quotas[s] > min_per_scene],
            key=lambda s: quotas[s],
            reverse=True,
        )
        for s in deficit_scenes:
            needed = min_per_scene - quotas[s]
            for donor in surplus_scenes:
                give = min(needed, quotas[donor] - min_per_scene)
                if give > 0:
                    quotas[donor] -= give
                    quotas[s] += give
                    needed -= give
                if needed <= 0:
                    break
    return quotas


def select_core_and_support_words(
    anchor: dict[str, Any],
    scene_words: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    scene_id: str,
    rng: random.Random | None = None,
    forced_core_word: dict[str, Any] | None = None,
    domain_context: dict[str, Any] | None = None,
    scene_core_usage_counts: Counter[str] | None = None,
    scene_support_usage_counts: Counter[str] | None = None,
    scene_place_support_counts: Counter[str] | None = None,
    scene_last_support_surface: str = "",
    available_core_surfaces: set[str] | None = None,
    core_usage_cap_count: int = 0,
    force_place_support: bool = False,
    allow_place_support: bool = True,
    place_support_cap_count: int = 0,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    anchor_sig = example_signature(anchor)
    example_keywords = extract_keywords(*(example.get("zh_sentence") for example in examples))
    combined_sig = anchor_sig + [keyword for keyword in example_keywords if keyword not in set(anchor_sig)]
    scene_core_counts = scene_core_usage_counts or Counter()
    scene_support_counts = scene_support_usage_counts or Counter()
    scene_place_counts = scene_place_support_counts or Counter()
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
        if (
            2 <= len(str(candidate["wz_word"])) <= 4
            and not word_is_place_like(candidate)
            and core_word_looks_usable(candidate, scene_id, anchor)
        ):
            core_word = candidate
            if scene_id == "food_dining" and candidate.get("is_modern"):
                return core_word, []
    else:
        core_candidates_by_surface: dict[str, dict[str, Any]] = {}
        anchor_word = {
            "wz_word": anchor.get("wz_word", ""),
            "definition": anchor.get("definition", ""),
            "scene_id": scene_id,
            "semantic_class": "unknown",
            "slot_kind": "unknown",
            "source_type": "anchor_example",
            "is_modern": False,
            "modern_priority": 0,
        }
        if (
            2 <= len(str(anchor_word["wz_word"])) <= 4
            and not word_is_place_like(anchor_word)
            and core_word_looks_usable(anchor_word, scene_id, anchor)
        ):
            merge_word_candidate(core_candidates_by_surface, anchor_word)

        for example in examples:
            example_word = {
                "wz_word": example.get("wz_word", ""),
                "definition": example.get("definition", ""),
                "scene_id": scene_id,
                "semantic_class": "unknown",
                "slot_kind": "unknown",
                "source_type": "prompt_example",
                "is_modern": False,
                "modern_priority": 0,
            }
            if (
                2 <= len(str(example_word["wz_word"])) <= 4
                and not word_is_place_like(example_word)
                and core_word_looks_usable(example_word, scene_id, example)
            ):
                merge_word_candidate(core_candidates_by_surface, example_word)

        for word in scene_words:
            if word_is_place_like(word):
                continue
            if core_word_looks_usable(word, scene_id, anchor):
                merge_word_candidate(core_candidates_by_surface, word)

        ranked_core_candidates: list[tuple[tuple[int, int, int, int, int, int, str], dict[str, Any]]] = []
        for candidate in core_candidates_by_surface.values():
            wz_word = str(candidate.get("wz_word") or "").strip()
            definition = str(candidate.get("definition") or "")
            ranked_core_candidates.append(
                (
                    (
                        domain_term_priority(
                            wz_word,
                            definition=definition,
                            domain_context=domain_context,
                        ),
                        int(wz_word == str(anchor.get("wz_word") or "").strip()),
                        overlap_score(combined_sig, word_signature(candidate)),
                        scene_match_score(
                            scene_id,
                            wz_word,
                            definition,
                            anchor.get("zh_sentence", ""),
                            anchor.get("wz_sentence", ""),
                        ),
                        core_source_priority(candidate),
                        int(candidate.get("modern_priority", 0)),
                        wz_word,
                    ),
                    candidate,
                )
            )
        ranked_core_candidates.sort(key=lambda item: item[0], reverse=True)
        core_candidates = [candidate for _, candidate in ranked_core_candidates]
        if (
            available_core_surfaces
            and len(available_core_surfaces) >= 4
            and core_usage_cap_count > 0
        ):
            uncapped_candidates = [
                candidate
                for candidate in core_candidates
                if scene_core_counts[str(candidate.get("wz_word") or "").strip()] < core_usage_cap_count
            ]
            if uncapped_candidates:
                core_candidates = uncapped_candidates
        if core_candidates:
            core_pool = core_candidates[: min(6, len(core_candidates))]
            min_core_usage = min(
                scene_core_counts[str(candidate.get("wz_word") or "").strip()]
                for candidate in core_pool
            )
            least_used_candidates = [
                candidate
                for candidate in core_pool
                if scene_core_counts[str(candidate.get("wz_word") or "").strip()] == min_core_usage
            ]
            core_choice_pool = least_used_candidates[: min(3, len(least_used_candidates))]
            core_word = (
                rng.choice(core_choice_pool)
                if rng is not None and len(core_choice_pool) > 1
                else core_choice_pool[0]
            )

    ranked_support: list[tuple[tuple[int, int, int, float, str], dict[str, Any]]] = []
    seen_words: set[str] = {str(core_word.get("wz_word"))} if core_word else set()
    core_surface = str(core_word.get("wz_word") or "") if core_word else ""

    for word in scene_words:
        wz_word = str(word.get("wz_word") or "").strip()
        if wz_word in seen_words or not word_looks_usable(word):
            continue
        if wz_word in scene_deny_terms(scene_id):
            continue
        if word_is_place_like(word) and not allow_place_support:
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
        is_place = str(word.get("slot_kind") or "") == "place"
        if score <= 0 and not word.get("is_modern", False) and not is_place:
            continue
        scene_priority = support_priority(scene_id, core_surface, word)
        domain_priority = domain_term_priority(
            wz_word,
            definition=str(word.get("definition") or ""),
            domain_context=domain_context,
        )
        if scene_id == "food_dining" and core_surface == "焯菜" and wz_word in FOOD_BEVERAGE_TERMS:
            continue
        ranked_support.append(
            (
                (
                    domain_priority,
                    scene_priority,
                    score,
                    int(word.get("modern_priority", 0)),
                    core_source_priority(word),
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
    required_support_candidates = [
        word
        for word in support_candidates
        if domain_term_priority(
            str(word.get("wz_word") or ""),
            definition=str(word.get("definition") or ""),
            domain_context=domain_context,
        )
        >= 3
    ]
    if required_support_candidates and domain_term_priority(
        core_surface,
        definition=str(core_word.get("definition") or "") if core_word else "",
        domain_context=domain_context,
    ) < 3:
        support_candidates = required_support_candidates + [
            word for word in support_candidates if word not in required_support_candidates
        ]

    max_support = 2 if len(support_candidates) >= 3 else 1
    pool_size = SCENE_SUPPORT_POOL_SIZES.get(scene_id, 4)

    def pick_support_candidate(ordered_candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not ordered_candidates:
            return None
        pool = list(ordered_candidates[: min(pool_size, len(ordered_candidates))])
        under_place_cap = [
            candidate
            for candidate in pool
            if not (
                word_is_place_like(candidate)
                and place_support_cap_count > 0
                and scene_place_counts[str(candidate.get("wz_word") or "").strip()] >= place_support_cap_count
            )
        ]
        if under_place_cap:
            pool = under_place_cap
        if scene_last_support_surface:
            non_repeat_pool = [
                candidate
                for candidate in pool
                if str(candidate.get("wz_word") or "").strip() != scene_last_support_surface
            ]
            if non_repeat_pool:
                pool = non_repeat_pool
        min_usage = min(scene_support_counts[str(candidate.get("wz_word") or "").strip()] for candidate in pool)
        usage_pool = [
            candidate
            for candidate in pool
            if scene_support_counts[str(candidate.get("wz_word") or "").strip()] == min_usage
        ]
        least_used_place_pool = usage_pool
        if any(word_is_place_like(candidate) for candidate in usage_pool):
            min_place_usage = min(
                scene_place_counts[str(candidate.get("wz_word") or "").strip()]
                for candidate in usage_pool
                if word_is_place_like(candidate)
            )
            least_used_place_pool = [
                candidate
                for candidate in usage_pool
                if not word_is_place_like(candidate)
                or scene_place_counts[str(candidate.get("wz_word") or "").strip()] == min_place_usage
            ]
        return (
            rng.choice(least_used_place_pool)
            if rng is not None and len(least_used_place_pool) > 1
            else least_used_place_pool[0]
        )

    if force_place_support:
        place_candidates = [word for word in support_candidates if word_is_place_like(word)]
        chosen_place = pick_support_candidate(place_candidates)
        if chosen_place is not None:
            support_words.append(chosen_place)
            seen_words.add(str(chosen_place.get("wz_word") or "").strip())

    while len(support_words) < max_support:
        remaining_candidates = [
            word
            for word in support_candidates
            if str(word.get("wz_word") or "").strip() not in seen_words
        ]
        if not remaining_candidates:
            break
        chosen_support = pick_support_candidate(remaining_candidates)
        if chosen_support is None:
            break
        support_words.append(chosen_support)
        seen_words.add(str(chosen_support.get("wz_word") or "").strip())

    if (
        forced_core_word is None
        and core_word is None
        and support_words
        and not word_is_place_like(support_words[0])
        and core_word_looks_usable(support_words[0], scene_id, anchor)
    ):
        core_word = support_words.pop(0)
    return core_word, support_words


# ===================== TASK BUILDING =====================

SYSTEM_PROMPT = f"""你是{DIALECT_NAME}句子生成器。你的任务是根据提供的{DIALECT_NAME}词典例句和词汇，生成自然的{DIALECT_NAME}长句。

规则：
1. 每句必须 {MIN_SENTENCE_LENGTH}-{MAX_SENTENCE_LENGTH} 个字（含标点）
2. 每句必须使用给定的"核心词汇"
3. 每句必须对应任务指定的"语气类型"标签（提问/抱怨/请求/叙述/评价），不要全部写成叙述或命令
4. 句子要像当地人真实说话的口语：问别人事情、抱怨天气、催家人做事、跟朋友讲八卦，不要写空泛旁白
5. 辅助词汇只有在非常自然时才可加入，最多加入 1 个，不要硬塞
6. 保持例句中展示的方言特征，但不要为了像方言而乱拼功能词
7. 不要写成{STANDARD_LANGUAGE_LABEL}
8. 每句要有完整的语义，适合语音训练朗读
9. 生成 3 句，每句独立，每句对应一种不同的语气类型
10. 优先围绕核心词展开一个完整、日常的小情境，不要把不相关词硬拼进一句
11. 如果辅助词是新事物、现代地点或设备名称，只在真正自然时带进去
12. 如果任务里列了"禁止词汇"，即使参考例句里出现了也绝对不要复用
13. 不要混入其他吴语区常见词形；只能跟参考例句、本地词表和给定词汇走，不会说就换成本地更稳的说法
14. 功能词语法必须比"像不像方言"更优先；拿不准时，宁可少用 `爳 / 罢 / 著埭 / 起 / 落去`
15. 不要自己发明新的两字到四字词；除给定词和参考例句能支持的说法外，拿不准就改写成来源里已有的稳妥表达
""" + "\n\n" + build_generation_grammar_prompt_rules() + f"\n\n只输出 JSON：\n" + f'{{"sentences": [{{"wz": "{DIALECT_NAME}句子", "zh": "{STANDARD_LANGUAGE_LABEL}翻译"}}]}}'

def build_task(
    examples: list[dict[str, Any]],
    core_word: dict[str, Any],
    support_words: list[dict[str, Any]],
    scene_id: str,
    task_id: str,
    lane: str,
    core_tier: str,
    domain_ids: list[str],
    domain_context: dict[str, Any] | None = None,
    speech_acts: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    domain_context = domain_context or build_domain_context(domain_ids, scene_id=scene_id)
    turn_role = default_turn_role(scene_id)
    core_word_meta = {
        "definition": str(core_word.get("definition") or ""),
        "semantic_class": str(core_word.get("semantic_class") or "unknown"),
        "slot_kind": str(core_word.get("slot_kind") or "unknown"),
        "source_type": str(core_word.get("source_type") or ""),
        "is_modern": bool(core_word.get("is_modern", False)),
        "modern_priority": int(core_word.get("modern_priority", 0) or 0),
    }
    support_word_meta = {
        str(word.get("wz_word") or ""): {
            "definition": str(word.get("definition") or ""),
            "semantic_class": str(word.get("semantic_class") or "unknown"),
            "slot_kind": str(word.get("slot_kind") or "unknown"),
            "source_type": str(word.get("source_type") or ""),
            "is_modern": bool(word.get("is_modern", False)),
            "modern_priority": int(word.get("modern_priority", 0) or 0),
        }
        for word in support_words
        if str(word.get("wz_word") or "").strip()
    }
    banned_terms = (
        global_deny_terms()
        | scene_deny_terms(scene_id)
        | set(domain_context.get("blocked_terms") or [])
        | (SHOPPING_OLD_MONEY_TERMS if scene_id == "shopping_payment" else set())
    )
    prompt_examples, examples_contaminated = prompt_examples_for_task(scene_id, examples, banned_terms)
    example_block = "\n".join(
        f"  {i+1}. {DIALECT_NAME}：{ex['wz_sentence']}\n     {STANDARD_LANGUAGE_LABEL}：{ex['zh_sentence']}"
        for i, ex in enumerate(prompt_examples)
    ) or "  - 本任务不展示旧例句，只保留核心词和辅助词，请直接生成自然口语句子。"
    support_block = "\n".join(
        f"  - {w['wz_word']}（{w['definition']}）"
        for w in support_words
    ) or "  - 无"
    banned_block = "\n".join(f"  - {term}" for term in sorted(banned_terms)) or "  - 无"
    domain_required_block = "\n".join(
        f"  - {term}" for term in (domain_context.get("required_terms") or [])
    ) or "  - 无"
    domain_preferred_block = "\n".join(
        f"  - {term}" for term in (domain_context.get("preferred_terms") or [])
    ) or "  - 无"
    domain_notes = domain_context.get("prompt_notes") or []
    domain_note_block = "\n".join(f"  - {note}" for note in domain_notes) or "  - 无"
    contamination_note = ""
    if examples_contaminated:
        contamination_note = "\n注意：旧材料里有过时说法，只学句式，不要复用旧词。"
    shopping_amount_note = ""
    if scene_id == "shopping_payment" and any(str(w.get("wz_word") or "").strip() in SHOPPING_AMOUNT_TERMS for w in support_words):
        shopping_amount_note = "\n如果句子里提到价钱，优先直接用给定的具体金额说法，不要改成泛泛的“钞票”，也不要写“银圆”。"
    grammar_user_rules = build_generation_grammar_user_rules()
    # Pick a communicative function for this task (deterministic by task_id)
    scene_comm_funcs = SCENE_COMMUNICATIVE_FUNCTIONS.get(scene_id, [])
    comm_func_line = ""
    if scene_comm_funcs:
        comm_func_rng = random.Random(task_id)
        picked_func = comm_func_rng.choice(scene_comm_funcs)
        comm_func_line = f"\n本任务的交际场景重点：{picked_func}\n"
    turn_role_note = ""
    if turn_role == "user_query":
        turn_role_note = "\n这批只生成用户对手机、软件或 AI 发起的一句话，不要写 AI 回复，也不要写成来回对话。"
    user_msg = f"""场景：{scene_id}{comm_func_line}

参考{DIALECT_NAME}例句（注意学习其中的方言风格和用词习惯）：
{example_block}

核心词汇（每句必须使用）：
  - {core_word['wz_word']}（{core_word['definition']}）

辅助词汇（只有自然时才用，最多用 1 个）：
{support_block}

领域必用词（如果给了，就至少自然地用 1 个，不要硬塞）：
{domain_required_block}

领域优先词（自然时优先考虑，可以不用全带）：
{domain_preferred_block}

禁止词汇（即使参考例句出现，也绝对不要写进新句子）：
{banned_block}{contamination_note}{shopping_amount_note}

领域补充说明：
{domain_note_block}

请额外遵守这些{DIALECT_NAME}语法约束：
{grammar_user_rules}

如果一句话里需要额外内容词，优先复用参考例句和本地来源里已经出现过的说法，不要自己新造两字到四字词。
{turn_role_note}
"""
    # Build speech act instruction block
    if speech_acts and len(speech_acts) >= 1:
        act_lines = "\n".join(
            f"  {i+1}. {act['label']}：{act['hint']}"
            for i, act in enumerate(speech_acts)
        )
        n_sentences = len(speech_acts)
        user_msg += f"""请生成 {n_sentences} 个 {MIN_SENTENCE_LENGTH}-{MAX_SENTENCE_LENGTH} 字的{DIALECT_NAME}口语长句，每句对应下面指定的语气类型：
{act_lines}"""
    else:
        user_msg += f"""请生成 3 个 {MIN_SENTENCE_LENGTH}-{MAX_SENTENCE_LENGTH} 字的{DIALECT_NAME}口语长句。"""

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
        "turn_role": turn_role,
        "support_words": [w["wz_word"] for w in support_words],
        "support_word_defs": {w["wz_word"]: w["definition"] for w in support_words},
        "domain_ids": list(domain_context.get("domain_ids") or []),
        "domain_labels": list(domain_context.get("domain_labels") or []),
        "domain_required_terms": list(domain_context.get("required_terms") or []),
        "domain_preferred_terms": list(domain_context.get("preferred_terms") or []),
        "domain_blocked_terms": list(domain_context.get("blocked_terms") or []),
        "domain_prompt_notes": list(domain_context.get("prompt_notes") or []),
        "domain_review_notes": list(domain_context.get("review_notes") or []),
        "target_words": [core_word["wz_word"]] + [w["wz_word"] for w in support_words],
        "approved_modern_terms": approved_modern_terms,
        "banned_terms": banned_terms_sorted,
        "prompt_examples_contaminated": examples_contaminated,
        "speech_acts": [act["id"] for act in (speech_acts or [])],
        "prompt_system": SYSTEM_PROMPT,
        "prompt_user": user_msg,
        "core_word_meta": core_word_meta,
        "support_word_meta": support_word_meta,
    }


def create_tasks_balanced(
    examples_by_scene: dict[str, list[dict]],
    words_by_scene: dict[str, list[dict]],
    num_tasks: int,
    seed: int = 42,
    scene_filter: list[str] | None = None,
    food_modern_trial_ratio: float = 0.25,
    domain_ids: list[str] | None = None,
    feedback_plan: dict[str, Any] | None = None,
) -> list[dict]:
    """Create tasks with scene balance, anchor examples, and core-word-driven prompts."""
    rng = random.Random(seed)
    tasks = []
    used_prompt_signatures: set[tuple[str, str, frozenset[str], tuple[str, ...], tuple[str, ...]]] = set()
    word_combo_variant_counts: Counter[tuple[str, str, frozenset[str]]] = Counter()
    selected_domain_ids = normalize_domain_ids(domain_ids)
    feedback = feedback_plan or {}
    feedback_scene_weights = feedback.get("scene_weights") if isinstance(feedback.get("scene_weights"), dict) else {}
    feedback_anchor_top_k = int(feedback.get("anchor_pool_top_k") or 0)
    feedback_domain_anchor_top_k = int(feedback.get("domain_anchor_pool_top_k") or 0)
    scene_domain_contexts = {
        scene_id: build_domain_context(selected_domain_ids, scene_id=scene_id)
        for scene_id in PRIORITY_SCENES
    }

    # Determine which scenes have enough data
    allowed_scenes = set(scene_filter or PRIORITY_SCENES)
    viable_scenes = [
        s for s in PRIORITY_SCENES
        if s in allowed_scenes
        if scene_has_prompt_material(s, examples_by_scene, words_by_scene)
        and (
            not selected_domain_ids
            or bool(scene_domain_contexts.get(s, {}).get("domain_ids"))
        )
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
            if scene_has_prompt_material(s, examples_by_scene, words_by_scene)
            and (
                not selected_domain_ids
                or bool(scene_domain_contexts.get(s, {}).get("domain_ids"))
            )
            and (
                s not in MODERN_SIDECAR_SCENES
                or MODERN_ANCHOR_GATE.get(s, {}).get("threshold_met", False)
            )
        ]
    if not viable_scenes:
        return []

    # Allocate tasks per scene (round-robin with priority)
    scene_quotas = allocate_scene_quotas(
        viable_scenes,
        num_tasks,
        rng=rng,
        scene_weights=feedback_scene_weights,
    )

    max_retries_per_task = 20
    all_speech_acts = load_speech_act_types()
    max_prompt_variants = max(MAX_PROMPT_VARIANTS_PER_WORD_COMBO, len(all_speech_acts) or 1)
    global_speech_act_offset_counts: Counter[int] = Counter()
    scene_speech_act_offset_counts: dict[str, Counter[int]] = defaultdict(Counter)
    scene_states: dict[str, dict[str, Any]] = {}
    scene_order = list(viable_scenes)
    rng.shuffle(scene_order)

    for scene in viable_scenes:
        quota = scene_quotas[scene]
        scene_domain_context = scene_domain_contexts.get(scene) or build_domain_context(
            selected_domain_ids,
            scene_id=scene,
        )
        if selected_domain_ids and not scene_domain_context.get("domain_ids"):
            continue
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
            and word_example_counts[str(example.get("wz_word") or "").strip()] >= 1
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
                and word_example_counts[str(example.get("wz_word") or "").strip()] >= 1
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
        should_rank_anchors = bool(scene_domain_context.get("domain_ids")) or feedback_anchor_top_k > 0
        if should_rank_anchors:
            anchor_candidates = prioritize_anchor_candidates(
                anchor_candidates,
                domain_context=scene_domain_context,
                rng=rng,
            )
        available_core_candidates: dict[str, dict[str, Any]] = {}
        for word in scene_words:
            if word_is_place_like(word):
                continue
            if core_word_looks_usable(word, scene):
                merge_word_candidate(available_core_candidates, word)
        for example in scene_examples:
            example_word = {
                "wz_word": example.get("wz_word", ""),
                "definition": example.get("definition", ""),
                "scene_id": scene,
                "semantic_class": "unknown",
                "slot_kind": "unknown",
                "source_type": "prompt_example",
                "is_modern": False,
                "modern_priority": 0,
            }
            if not word_is_place_like(example_word) and core_word_looks_usable(
                example_word,
                scene,
                example,
            ):
                merge_word_candidate(available_core_candidates, example_word)
        available_core_surfaces = set(available_core_candidates)
        core_usage_cap_count = 0
        if (
            available_core_surfaces
            and len(available_core_surfaces) >= 4
            and quota > 0
            and CORE_USAGE_CAP_RATIO > 0
        ):
            core_usage_cap_count = max(1, math.ceil(quota * CORE_USAGE_CAP_RATIO))
        available_place_support_surfaces = {
            str(word.get("wz_word") or "").strip()
            for word in scene_words
            if word_is_place_like(word) and word_looks_usable(word)
        }
        place_support_target = 0
        place_support_cap_count = 0
        if (
            scene == "transport_trip"
            and available_place_support_surfaces
            and quota > 0
            and PLACE_SUPPORT_RATIO > 0
        ):
            place_support_target = min(quota, max(1, round(quota * PLACE_SUPPORT_RATIO)))
            if PLACE_SUPPORT_CAP_RATIO > 0:
                place_support_cap_count = max(1, math.ceil(quota * PLACE_SUPPORT_CAP_RATIO))
        scene_states[scene] = {
            "quota": quota,
            "scene_domain_context": scene_domain_context,
            "scene_examples": scene_examples,
            "scene_words": scene_words,
            "scene_modern_core_terms": scene_modern_core_terms,
            "modern_core_candidates": modern_core_candidates,
            "food_trial_target": food_trial_target,
            "food_trial_done": 0,
            "used_trial_core_surfaces": set(),
            "anchor_candidates": anchor_candidates,
            "available_core_surfaces": available_core_surfaces,
            "core_usage_cap_count": core_usage_cap_count,
            "available_place_support_surfaces": available_place_support_surfaces,
            "place_support_target": place_support_target,
            "place_support_done": 0,
            "place_support_cap_count": place_support_cap_count,
        }

    if not scene_states:
        return []

    scene_order = [scene for scene in scene_order if scene in scene_states]
    scene_order_index = {scene: index for index, scene in enumerate(scene_order)}
    scene_task_counts: Counter[str] = Counter()
    scene_core_usage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scene_support_usage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scene_place_support_usage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scene_place_task_counts: Counter[str] = Counter()
    scene_last_support_surface: dict[str, str] = {}
    scene_exhausted: set[str] = set()

    def attempt_task_for_scene(scene: str) -> dict[str, Any] | None:
        state = scene_states.get(scene)
        if not state or scene in scene_exhausted:
            return None
        scene_domain_context = state["scene_domain_context"]
        scene_examples = state["scene_examples"]
        scene_words = state["scene_words"]
        modern_core_candidates = state["modern_core_candidates"]
        scene_modern_core_terms = state["scene_modern_core_terms"]
        anchor_candidates = state["anchor_candidates"]
        available_core_surfaces = state["available_core_surfaces"]
        core_usage_cap_count = int(state["core_usage_cap_count"])
        available_place_support_surfaces = state["available_place_support_surfaces"]
        place_support_target = int(state["place_support_target"])
        place_support_cap_count = int(state["place_support_cap_count"])
        quota = int(state["quota"])
        for _retry in range(max_retries_per_task):
            anchor_top_k = 0
            if scene_domain_context.get("domain_ids"):
                anchor_top_k = feedback_domain_anchor_top_k or 4
            elif feedback_anchor_top_k > 0:
                anchor_top_k = feedback_anchor_top_k
            if anchor_top_k > 0:
                anchor_pool = anchor_candidates[: min(anchor_top_k, len(anchor_candidates))]
                anchor = rng.choice(anchor_pool)
            else:
                anchor = rng.choice(anchor_candidates)
            exs = related_examples(anchor, scene_examples, limit=4)
            if not exs:
                exs = [anchor]
            if scene in FOCUS_SCENES and len(exs) < 2:
                continue
            allow_place_support = bool(available_place_support_surfaces) and task_prefers_place_support(
                scene,
                anchor,
                exs,
            )
            force_place_support = False
            if allow_place_support and scene == "transport_trip" and place_support_target > state["place_support_done"]:
                remaining_slots = quota - scene_task_counts[scene]
                remaining_place_needed = place_support_target - state["place_support_done"]
                if remaining_slots <= remaining_place_needed or rng.random() < PLACE_SUPPORT_RATIO:
                    force_place_support = True
            forced_core_word = None
            want_modern_trial = False
            if (
                scene == "food_dining"
                and modern_core_candidates
                and state["food_trial_done"] < state["food_trial_target"]
                and scene_task_counts[scene] < quota
            ):
                remaining_slots = quota - scene_task_counts[scene]
                remaining_trial = state["food_trial_target"] - state["food_trial_done"]
                if remaining_slots <= remaining_trial or rng.random() < food_modern_trial_ratio:
                    want_modern_trial = True
                    unseen_modern_candidates = [
                        word
                        for word in modern_core_candidates
                        if str(word.get("wz_word") or "").strip() not in state["used_trial_core_surfaces"]
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
                domain_context=scene_domain_context,
                scene_core_usage_counts=scene_core_usage_counts[scene],
                scene_support_usage_counts=scene_support_usage_counts[scene],
                scene_place_support_counts=scene_place_support_usage_counts[scene],
                scene_last_support_surface=scene_last_support_surface.get(scene, ""),
                available_core_surfaces=available_core_surfaces,
                core_usage_cap_count=core_usage_cap_count,
                force_place_support=force_place_support,
                allow_place_support=allow_place_support,
                place_support_cap_count=place_support_cap_count if allow_place_support else 0,
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
            word_combo_signature = task_word_combo_signature(scene, core_word_surface, support_words)
            if word_combo_variant_counts[word_combo_signature] >= max_prompt_variants:
                continue

            selected_speech_acts: list[dict[str, str]] = []
            speech_act_offset = 0
            prompt_signature = None
            for candidate_offset in ordered_speech_act_offsets(
                all_speech_acts,
                global_speech_act_offset_counts,
                scene_speech_act_offset_counts[scene],
            ):
                candidate_speech_acts = speech_act_window(all_speech_acts, candidate_offset, window_size=3)
                candidate_signature = task_prompt_signature(
                    scene,
                    core_word_surface,
                    support_words,
                    exs,
                    candidate_speech_acts,
                )
                if candidate_signature in used_prompt_signatures:
                    continue
                selected_speech_acts = candidate_speech_acts
                speech_act_offset = candidate_offset
                prompt_signature = candidate_signature
                break
            if prompt_signature is None and not all_speech_acts:
                prompt_signature = task_prompt_signature(
                    scene,
                    core_word_surface,
                    support_words,
                    exs,
                    [],
                )
            if prompt_signature is None or prompt_signature in used_prompt_signatures:
                continue
            used_prompt_signatures.add(prompt_signature)
            word_combo_variant_counts[word_combo_signature] += 1

            tid = f"fs_{hashlib.md5(f'{scene}_{len(tasks)}_{seed}'.encode()).hexdigest()[:10]}"
            task = build_task(
                exs,
                core_word,
                support_words,
                scene,
                tid,
                lane,
                core_tier,
                list(scene_domain_context.get("domain_ids") or []),
                domain_context=scene_domain_context,
                speech_acts=selected_speech_acts,
            )
            task["temperature"] = 0.5 + 0.1 * (len(tasks) % 3)
            task["speech_act_offset"] = speech_act_offset
            scene_core_usage_counts[scene][core_word_surface] += 1
            has_place_support = False
            for support_word in support_words:
                support_surface = str(support_word.get("wz_word") or "").strip()
                if not support_surface:
                    continue
                scene_support_usage_counts[scene][support_surface] += 1
                if word_is_place_like(support_word):
                    has_place_support = True
                    scene_place_support_usage_counts[scene][support_surface] += 1
            if support_words:
                scene_last_support_surface[scene] = str(support_words[0].get("wz_word") or "").strip()
            if has_place_support:
                scene_place_task_counts[scene] += 1
                if scene == "transport_trip":
                    state["place_support_done"] += 1
            task["has_place_support"] = has_place_support
            if lane == "food_modern_trial":
                state["food_trial_done"] += 1
                state["used_trial_core_surfaces"].add(core_word_surface)
            if all_speech_acts:
                global_speech_act_offset_counts[speech_act_offset] += 1
                scene_speech_act_offset_counts[scene][speech_act_offset] += 1
            return task
        scene_exhausted.add(scene)
        return None

    for scene in scene_order:
        target_quota = scene_quotas.get(scene, 0)
        while scene_task_counts[scene] < target_quota:
            task = attempt_task_for_scene(scene)
            if task is None:
                break
            tasks.append(task)
            scene_task_counts[scene] += 1

    active_scenes = [scene for scene in scene_order if scene not in scene_exhausted]
    while len(tasks) < num_tasks and active_scenes:
        progress = False
        refill_pool = sorted(
            active_scenes,
            key=lambda scene: (
                scene_task_counts[scene] / max(1, scene_quotas.get(scene, 1)),
                scene_task_counts[scene],
                scene_order_index.get(scene, len(scene_order_index)),
            ),
        )
        for scene in refill_pool:
            if len(tasks) >= num_tasks:
                break
            task = attempt_task_for_scene(scene)
            if task is None:
                continue
            tasks.append(task)
            scene_task_counts[scene] += 1
            progress = True
        active_scenes = [scene for scene in active_scenes if scene not in scene_exhausted]
        if not progress:
            break

    # Shuffle to interleave scenes (avoid hitting one API pattern too long)
    rng.shuffle(tasks)
    return tasks


# ===================== GENERATION =====================


def summarize_task_lexical_diversity(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    scene_core_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scene_support_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scene_place_support_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scene_place_task_counts: Counter[str] = Counter()

    for task in tasks:
        scene_id = str(task.get("scene_id") or "").strip()
        if not scene_id:
            continue
        core_word = str(task.get("core_word") or "").strip()
        if core_word:
            scene_core_task_counts[scene_id][core_word] += 1
        support_meta = task.get("support_word_meta")
        if not isinstance(support_meta, dict):
            support_meta = {}
        has_place_support = False
        for support_word in task.get("support_words", []) or []:
            support_surface = str(support_word or "").strip()
            if not support_surface:
                continue
            scene_support_task_counts[scene_id][support_surface] += 1
            meta = support_meta.get(support_surface)
            if isinstance(meta, dict) and word_is_place_like(meta):
                scene_place_support_task_counts[scene_id][support_surface] += 1
                has_place_support = True
        if has_place_support:
            scene_place_task_counts[scene_id] += 1

    def serialize_counter_map(counter_map: dict[str, Counter[str]], *, top_n: int = 0) -> dict[str, dict[str, int]]:
        serialized: dict[str, dict[str, int]] = {}
        for scene_id in sorted(counter_map):
            counter = counter_map[scene_id]
            items = counter.most_common(top_n) if top_n > 0 else sorted(counter.items())
            serialized[scene_id] = {word: count for word, count in items}
        return serialized

    return {
        "scene_core_task_counts": serialize_counter_map(scene_core_task_counts),
        "scene_support_task_counts": serialize_counter_map(scene_support_task_counts),
        "scene_place_support_task_counts": serialize_counter_map(scene_place_support_task_counts),
        "scene_distinct_core_count": {
            scene_id: len(counter)
            for scene_id, counter in sorted(scene_core_task_counts.items())
        },
        "scene_distinct_support_count": {
            scene_id: len(counter)
            for scene_id, counter in sorted(scene_support_task_counts.items())
        },
        "scene_place_task_counts": dict(sorted(scene_place_task_counts.items())),
        "scene_top_core_words": serialize_counter_map(scene_core_task_counts, top_n=10),
        "scene_top_support_words": serialize_counter_map(scene_support_task_counts, top_n=10),
    }


def sentence_skeleton(wz: str, content_words: list[str]) -> str:
    """Replace content words with _ to get a structural pattern for dedup monitoring."""
    s = wz
    for w in sorted(content_words, key=len, reverse=True):
        if w:
            s = s.replace(w, "_")
    return s


def request_generation(client, model, task):
    try:
        task_temperature = task.get("temperature", 0.6)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": task["prompt_system"]},
                {"role": "user", "content": task["prompt_user"]},
            ],
            temperature=task_temperature,
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
                {"role": "system", "content": build_grammar_repair_system_prompt()},
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

MANDARIN_MARKERS = list(
    DIALECT_SETTINGS.get("mandarin_markers")
    or [
        "的话", "然后", "但是", "因为", "所以", "而且", "虽然", "如果",
        "可是", "或者", "不过", "已经", "正在", "刚才",
        "什么", "怎么", "为什么", "哪里", "这里", "那里",
        "他们", "她们", "我们", "你们",
        "非常", "特别", "真的是",
    ]
)


def validate_sentence(
    wz,
    zh,
    known_words,
    core_word,
    support_words,
    existing_sentences,
    existing_skeleton_counts=None,
    candidate_skeleton: str = "",
    *,
    scene_id: str,
    lane: str,
    core_tier: str,
    approved_modern_terms: list[str],
    banned_terms: list[str],
    domain_required_terms: list[str],
    domain_preferred_terms: list[str],
    domain_blocked_terms: list[str],
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

    if char_len < MIN_SENTENCE_LENGTH:
        reasons.append(f"too_short:{char_len}")
    elif char_len > MAX_SENTENCE_LENGTH:
        reasons.append(f"too_long:{char_len}")

    found_known = sorted({w for w in known_words if w in wz_clean})
    core_present = core_word in wz_clean if core_word else False
    found_support = sorted({w for w in support_words if w in wz_clean})
    approved_modern_hits = sorted({w for w in approved_modern_terms if w in wz_clean})
    banned_term_hits = sorted({w for w in banned_terms if w in wz_clean})
    domain_required_hits = sorted({w for w in domain_required_terms if w in wz_clean})
    domain_preferred_hits = sorted({w for w in domain_preferred_terms if w in wz_clean})
    domain_blocked_hits = sorted({w for w in domain_blocked_terms if w in wz_clean})
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
    if domain_blocked_hits:
        reasons.append(f"domain_blocked_terms:{','.join(domain_blocked_hits)}")
    if domain_required_terms and not domain_required_hits:
        reasons.append("domain_required_terms_missing")
    for finding in unsupported_surface_terms:
        reasons.append(f"source_surface_missing:{finding['surface']}")

    mandarin_hits = [m for m in MANDARIN_MARKERS if m in wz_clean]
    if len(mandarin_hits) >= 2:
        reasons.append(f"mandarin_markers:{','.join(mandarin_hits[:3])}")

    if wz_clean in existing_sentences:
        reasons.append("duplicate")
    if candidate_skeleton and existing_skeleton_counts and existing_skeleton_counts.get(candidate_skeleton, 0) > 0:
        reasons.append("structural_duplicate")

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
        "domain_required_hits": domain_required_hits,
        "domain_preferred_hits": domain_preferred_hits,
        "domain_blocked_hits": domain_blocked_hits,
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

def load_completed_ids(results_path: Path) -> tuple[set[str], set[str], Counter[str]]:
    """Load already-completed task_ids, existing sentences, and skeleton counts."""
    done_ids = set()
    seen_sentences = set()
    skeleton_counts: Counter[str] = Counter()
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done_ids.add(row.get("task_id", ""))
                wz_sentence = clean_wz(str(row.get("wz_sentence", "")))
                if wz_sentence:
                    seen_sentences.add(wz_sentence)
                    content_words = [str(row.get("core_word") or "").strip()] + [
                        str(item).strip() for item in (row.get("support_words") or [])
                    ]
                    skeleton_counts[sentence_skeleton(wz_sentence, content_words)] += 1
    return done_ids, seen_sentences, skeleton_counts


def load_sibling_dedup_state(runs_root: Path, current_run_id: str) -> tuple[set[str], Counter[str]]:
    """Load existing sentences and skeleton counts from sibling runs in the same pipeline directory."""
    seen_sentences: set[str] = set()
    skeleton_counts: Counter[str] = Counter()
    if not runs_root.exists():
        return seen_sentences, skeleton_counts
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir() or run_dir.name == current_run_id:
            continue
        results_path = run_dir / "results.jsonl"
        if not results_path.exists():
            continue
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                wz_sentence = clean_wz(str(row.get("wz_sentence", "")))
                if not wz_sentence:
                    continue
                seen_sentences.add(wz_sentence)
                content_words = [str(row.get("core_word") or "").strip()] + [
                    str(item).strip() for item in (row.get("support_words") or [])
                ]
                skeleton_counts[sentence_skeleton(wz_sentence, content_words)] += 1
    return seen_sentences, skeleton_counts


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
    parser.add_argument(
        "--domains",
        type=str,
        default="",
        help="Comma-separated domain ids from configs/domain_catalog.json to inject into generation and review.",
    )
    parser.add_argument(
        "--feedback-run-id",
        type=str,
        default="",
        help="Optional prior run_id whose summary.json should steer scene quotas and anchor sampling.",
    )
    parser.add_argument(
        "--feedback-summary",
        type=Path,
        default=None,
        help="Optional summary.json path from a previous run to use for quota and sampling feedback.",
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
        "dialect": ACTIVE_DIALECT_CONFIG.dialect_key,
        "dialect_name": DIALECT_NAME,
        "tasks": args.tasks,
        "provider": args.provider,
        "model": model,
        "seed": args.seed,
        "sleep": args.sleep,
        "scenes": [],
        "domains": [],
        "food_modern_trial_ratio": args.food_modern_trial_ratio,
        "grammar_repair_enabled": not args.disable_grammar_repair,
        "grammar_spec_path": str(GRAMMAR_SPEC_PATH),
        "policy_version": POLICY_VERSION,
        "compatibility_output_dir": str(args.output_dir) if args.output_dir else "",
    }

    print("Loading data...")
    known_words = load_wz_word_set()
    examples_by_scene = load_examples_by_scene()
    words_by_scene = load_words_by_scene()
    selected_scenes = normalize_scene_list(args.scenes)
    selected_domains = normalize_domain_ids(args.domains)
    try:
        feedback_summary = load_feedback_summary(
            feedback_summary_path=args.feedback_summary,
            feedback_run_id=args.feedback_run_id,
            pipeline_name=PIPELINE_NAME,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))
    feedback_plan = build_feedback_plan(
        feedback_summary,
        scenes=selected_scenes or DEFAULT_SCENES,
        domain_ids=selected_domains,
        default_food_modern_trial_ratio=args.food_modern_trial_ratio,
    )
    effective_food_modern_trial_ratio = float(
        feedback_plan.get("food_modern_trial_ratio", args.food_modern_trial_ratio)
    )
    config["scenes"] = selected_scenes or DEFAULT_SCENES
    config["domains"] = selected_domains
    config["food_modern_trial_ratio"] = effective_food_modern_trial_ratio
    config["feedback"] = {
        "enabled": bool(feedback_plan.get("enabled")),
        "source_run_id": str(feedback_plan.get("source_run_id") or ""),
        "source_summary_path": str(feedback_plan.get("source_summary_path") or ""),
        "primary_bucket": str(feedback_plan.get("primary_bucket") or ""),
        "bucket_counts": dict(feedback_plan.get("bucket_counts") or {}),
        "scene_weights": dict(feedback_plan.get("scene_weights") or {}),
        "anchor_pool_top_k": int(feedback_plan.get("anchor_pool_top_k") or 0),
        "domain_anchor_pool_top_k": int(feedback_plan.get("domain_anchor_pool_top_k") or 0),
        "notes": list(feedback_plan.get("notes") or []),
    }

    print(f"Known WZ words (2-4 chars): {len(known_words)}")
    print(f"Selected scenes: {selected_scenes or DEFAULT_SCENES}")
    print(f"Selected domains: {selected_domains or ['<none>']}")
    if feedback_plan.get("enabled"):
        print(
            "Feedback plan: "
            f"bucket={feedback_plan.get('primary_bucket') or '<none>'}, "
            f"source_run_id={feedback_plan.get('source_run_id') or '<unknown>'}, "
            f"anchor_top_k={feedback_plan.get('anchor_pool_top_k') or 0}, "
            f"food_trial_ratio={effective_food_modern_trial_ratio}"
        )
        print(f"Feedback scene weights: {feedback_plan.get('scene_weights') or {}}")

    # Load existing sentences for dedup
    existing = set()
    for path in [EXTRACTED_SHORT, EXTRACTED_LONG]:
        if path.exists():
            with open(path) as f:
                for line in f:
                    existing.add(json.loads(line)["wz_sentence"])

    # Resume support
    done_ids = set()
    local_skeleton_counts: Counter[str] = Counter()
    if args.resume:
        done_ids, prev_sentences, prev_skeleton_counts = load_completed_ids(results_path)
        existing.update(prev_sentences)
        local_skeleton_counts.update(prev_skeleton_counts)
        print(
            f"Resuming: {len(done_ids)} tasks already done, {len(prev_sentences)} sentences in dedup pool"
        )

    sibling_sentences, sibling_skeleton_counts = load_sibling_dedup_state(layout.root.parent, run_id)
    print(
        "Shared dedup pool: "
        f"{len(sibling_sentences)} sibling sentences, {sum(sibling_skeleton_counts.values())} skeleton rows"
    )

    print(f"Existing sentences for dedup: {len(existing)}")

    planned_scene_contexts = {
        scene_id: build_domain_context(selected_domains, scene_id=scene_id)
        for scene_id in (selected_scenes or DEFAULT_SCENES)
    }
    planned_viable_scenes = [
        scene
        for scene in (selected_scenes or DEFAULT_SCENES)
        if scene_has_prompt_material(scene, examples_by_scene, words_by_scene)
        and (
            not selected_domains
            or bool(planned_scene_contexts.get(scene, {}).get("domain_ids"))
        )
        and (
            scene not in MODERN_SIDECAR_SCENES
            or MODERN_ANCHOR_GATE.get(scene, {}).get("threshold_met", False)
        )
    ]
    scene_quotas = allocate_scene_quotas(
        planned_viable_scenes,
        args.tasks,
        rng=random.Random(args.seed),
        scene_weights=feedback_plan.get("scene_weights") if isinstance(feedback_plan.get("scene_weights"), dict) else None,
    )

    print(f"\nCreating {args.tasks} tasks (scene-balanced, core-word-driven)...")
    tasks = create_tasks_balanced(
        examples_by_scene,
        words_by_scene,
        args.tasks,
        args.seed,
        scene_filter=selected_scenes,
        food_modern_trial_ratio=effective_food_modern_trial_ratio,
        domain_ids=selected_domains,
        feedback_plan=feedback_plan,
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
            sibling_sentences, sibling_skeleton_counts = load_sibling_dedup_state(layout.root.parent, run_id)
            effective_existing_sentences = existing | sibling_sentences
            effective_skeleton_counts = local_skeleton_counts + sibling_skeleton_counts
            elapsed = time.time() - t_start
            rate = (i / elapsed * 60) if elapsed > 0 and i > 0 else 0
            print(f"\n[{i+1}/{len(pending)}] {task['task_id']} ({task['scene_id']}) | {rate:.1f} tasks/min")
            print(f"  Core: {task['core_word']} | Support: {task['support_words']}")

            sentences = request_generation(client, model, task)
            stats["raw_sentences"] += len(sentences)

            task_speech_acts = normalized_speech_act_ids(task.get("speech_acts"))
            for sentence_index, sent in enumerate(sentences):
                wz = sent.get("wz", "")
                zh = sent.get("zh", "")
                if not wz:
                    stats["empty"] += 1
                    continue
                target_speech_act = task_speech_acts[sentence_index] if sentence_index < len(task_speech_acts) else ""
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
                candidate_skeleton = sentence_skeleton(
                    clean_wz(wz),
                    [task["core_word"]] + list(task["support_words"]),
                )

                val = validate_sentence(
                    wz,
                    zh,
                    known_words,
                    task["core_word"],
                    task["support_words"],
                    effective_existing_sentences,
                    effective_skeleton_counts,
                    candidate_skeleton,
                    scene_id=task["scene_id"],
                    lane=task.get("lane", "mainline"),
                    core_tier=task.get("core_tier", "stable"),
                    approved_modern_terms=task.get("approved_modern_terms", []),
                    banned_terms=task.get("banned_terms", []),
                    domain_required_terms=task.get("domain_required_terms", []),
                    domain_preferred_terms=task.get("domain_preferred_terms", []),
                    domain_blocked_terms=task.get("domain_blocked_terms", []),
                )
                result = {
                    "task_id": task["task_id"],
                    "scene_id": task["scene_id"],
                    "lane": task.get("lane", "mainline"),
                    "turn_role": task.get("turn_role", default_turn_role(task["scene_id"])),
                    "core_tier": task.get("core_tier", "stable"),
                    "policy_version": task.get("policy_version", POLICY_VERSION),
                    "wz_sentence": clean_wz(wz),
                    "zh_sentence": zh,
                    "core_word": task["core_word"],
                    "core_word_def": task.get("core_word_def", ""),
                    "support_words": task["support_words"],
                    "support_word_defs": task.get("support_word_defs", {}),
                    "domain_ids": task.get("domain_ids", []),
                    "domain_labels": task.get("domain_labels", []),
                    "domain_required_terms": task.get("domain_required_terms", []),
                    "domain_preferred_terms": task.get("domain_preferred_terms", []),
                    "domain_blocked_terms": task.get("domain_blocked_terms", []),
                    "domain_prompt_notes": task.get("domain_prompt_notes", []),
                    "domain_review_notes": task.get("domain_review_notes", []),
                    "target_words": task["target_words"],
                    "approved_modern_terms": task.get("approved_modern_terms", []),
                    "banned_terms": task.get("banned_terms", []),
                    "example_sources": task.get("example_sources", []),
                    "example_sentences": task.get("example_sentences", []),
                    "task_speech_acts": list(task_speech_acts),
                    "target_speech_act": target_speech_act,
                    "sentence_index": sentence_index,
                    "grammar_repair_applied": repair_applied,
                    "grammar_repair_trigger_reasons": repair_trigger_reasons,
                    "structural_duplicate": "structural_duplicate" in val["reasons"],
                    "structural_skeleton": candidate_skeleton,
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
                local_skeleton_counts[candidate_skeleton] += 1

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
    task_scene_counts = Counter(t.get("scene_id", "") for t in tasks)
    task_speech_act_counts = Counter(
        act for task in tasks for act in normalized_speech_act_ids(task.get("speech_acts"))
    )
    task_speech_act_pattern_counts = Counter(
        "|".join(normalized_speech_act_ids(task.get("speech_acts")))
        for task in tasks
        if normalized_speech_act_ids(task.get("speech_acts"))
    )
    lane_pass_counts = Counter(r.get("lane", "mainline") for r in passed)
    core_tier_pass_counts = Counter(r.get("core_tier", "stable") for r in passed)
    scene_pass_counts = Counter(r.get("scene_id", "") for r in passed)
    pass_target_speech_act_counts = Counter(
        str(r.get("target_speech_act") or "").strip()
        for r in passed
        if str(r.get("target_speech_act") or "").strip()
    )
    scene_pass_target_speech_act_counts: dict[str, dict[str, int]] = {}
    for scene in sorted(scene_pass_counts):
        act_counts = Counter(
            str(row.get("target_speech_act") or "").strip()
            for row in passed
            if row.get("scene_id") == scene and str(row.get("target_speech_act") or "").strip()
        )
        if act_counts:
            scene_pass_target_speech_act_counts[scene] = dict(act_counts)
    scene_task_shortfalls = {
        scene: max(0, int(scene_quotas.get(scene, 0)) - task_scene_counts.get(scene, 0))
        for scene in sorted(scene_quotas)
    }
    task_lexical_summary = summarize_task_lexical_diversity(tasks)
    banned_term_fail_counts = Counter()
    grammar_fail_counts = Counter()
    domain_fail_counts = Counter()
    naturalness_fail_counts = Counter()
    rule_fail_bucket_counts = Counter()
    for row in all_rule_results:
        for reason in row["validation"].get("grammar_reasons", []):
            grammar_fail_counts[reason] += 1
        if row.get("rule_gate_status") == "pass":
            pass
        else:
            validation_reasons = row["validation"].get("reasons", [])
            for bucket in classify_rule_fail_buckets(validation_reasons):
                rule_fail_bucket_counts[bucket] += 1
            for term in row["validation"].get("banned_term_hits", []):
                banned_term_fail_counts[term] += 1
            for reason in validation_reasons:
                bucket = classify_rule_reason_bucket(reason)
                key = reason_key(reason)
                if bucket == "domain":
                    domain_fail_counts[key] += 1
                elif bucket == "naturalness":
                    naturalness_fail_counts[key] += 1
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
            "scene_target_quotas": dict(scene_quotas),
            "scene_task_counts": dict(task_scene_counts),
            "scene_task_shortfalls": scene_task_shortfalls,
            "scene_rule_pass_counts": dict(scene_pass_counts),
            "lane_rule_pass_counts": dict(lane_pass_counts),
            "core_tier_rule_pass_counts": dict(core_tier_pass_counts),
            "task_speech_act_counts": dict(task_speech_act_counts),
            "task_speech_act_pattern_counts": dict(task_speech_act_pattern_counts),
            "pass_target_speech_act_counts": dict(pass_target_speech_act_counts),
            "scene_pass_target_speech_act_counts": scene_pass_target_speech_act_counts,
            **task_lexical_summary,
            "banned_term_fail_counts": dict(banned_term_fail_counts),
            "grammar_fail_counts": dict(grammar_fail_counts),
            "domain_fail_counts": dict(domain_fail_counts),
            "naturalness_fail_counts": dict(naturalness_fail_counts),
            "rule_fail_bucket_counts": dict(rule_fail_bucket_counts),
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
            "scene_target_quotas": dict(scene_quotas),
            "scene_task_counts": dict(task_scene_counts),
            "scene_task_shortfalls": scene_task_shortfalls,
            "scene_pass_counts": dict(scene_pass_counts),
            "lane_pass_counts": dict(lane_pass_counts),
            "core_tier_pass_counts": dict(core_tier_pass_counts),
            "task_speech_act_counts": dict(task_speech_act_counts),
            "task_speech_act_pattern_counts": dict(task_speech_act_pattern_counts),
            "pass_target_speech_act_counts": dict(pass_target_speech_act_counts),
            "scene_pass_target_speech_act_counts": scene_pass_target_speech_act_counts,
            **task_lexical_summary,
            "banned_term_fail_counts": dict(banned_term_fail_counts),
            "domain_fail_counts": dict(domain_fail_counts),
            "naturalness_fail_counts": dict(naturalness_fail_counts),
            "rule_fail_bucket_counts": dict(rule_fail_bucket_counts),
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
    failure_analysis = summary["machine_metrics"].get("failure_analysis") or {}
    if failure_analysis:
        print(f"Fail buckets: {failure_analysis.get('bucket_counts', {})}")
        if failure_analysis.get("top_grammar_reasons"):
            print(f"Top grammar fails: {failure_analysis['top_grammar_reasons'][:3]}")
        if failure_analysis.get("top_domain_reasons"):
            print(f"Top domain fails: {failure_analysis['top_domain_reasons'][:3]}")
        if failure_analysis.get("top_naturalness_reasons"):
            print(f"Top naturalness fails: {failure_analysis['top_naturalness_reasons'][:3]}")
    print(f"Time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"Task scene quotas: {dict(scene_quotas)}")
    print(f"Task scene counts: {dict(task_scene_counts)}")
    print(f"Task speech acts: {dict(task_speech_act_counts)}")
    print(f"Scene distribution: {dict(scene_pass_counts)}")
    if pass_target_speech_act_counts:
        print(f"Pass speech acts: {dict(pass_target_speech_act_counts)}")
    print(f"\nOutput files:")
    print(f"  Config: {layout.config_path}")
    print(f"  Results: {layout.results_path}")
    print(f"  Rule gate: {layout.rule_gate_path}")
    print(f"  Review TSV: {layout.review_tsv_path}")
    print(f"  Summary: {layout.summary_path}")


if __name__ == "__main__":
    main()
