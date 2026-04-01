from __future__ import annotations

import copy
import re
from functools import lru_cache
from typing import Any

import yaml

from .dialect import ACTIVE_DIALECT_CONFIG
from .paths import GRAMMAR_RULES_PATH


WENZHOU_LEGACY_GRAMMAR_CONFIG: dict[str, Any] = {
    "review_markers": ["爻", "罢", "著埭", "落去", "未", "冇", "不"],
    "repair_trigger_pattern": "(爻|罢|著埭|落去|起罢|啊不|啊未|啊冇|未|冇)",
    "completion_marker": "爻",
    "allowed_after_completion": "罢，。！？；,.!?;、再就还也阿沃个",
    "max_completion_count": 1,
    "sentence_final_particle": "罢",
    "max_sentence_final_count": 1,
    "high_risk_patterns": [
        {
            "name": "completion_object_order",
            "pattern": "跳爻舞",
            "description": "完成体宾语语序错误：应为 舞跳爻",
        },
        {
            "name": "stative_progressive",
            "pattern": "著埭(?:坐|晓得|快活)",
            "description": "进行体误配静态动词或状态词",
        },
        {
            "name": "dynamic_postposed_zhedai",
            "pattern": "(?:走|笑|飞)著埭",
            "description": "持续体误配动态动词",
        },
        {
            "name": "qishi_object_order",
            "pattern": "唱歌起罢",
            "description": "起始体宾语语序错误",
        },
        {
            "name": "continuative_object_order",
            "pattern": "一直开会落去",
            "description": "继续体宾语语序错误",
        },
        {
            "name": "mandarin_negation",
            "pattern": "(?:没有|沒有)",
            "description": "普通话否定混入",
        },
        {
            "name": "northern_wu_lexeme_gaotou",
            "pattern": "高头",
            "description": "混入北部吴语常见词形",
        },
    ],
    "modal_verbs_before_completion": {
        "pattern": "(?:想|要|会|會|能|可以|应该|應該|打算|喜欢|曉得|晓得|觉得|覺得|认识|認識|希望|准备|準備|肯|敢)爻"
    },
    "generation_section_titles": [
        "## 2. 生成总原则",
        "### 3.1 完成体：`爻`",
        "### 3.2 进行体：`著埭 + V`",
        "### 3.3 持续体：`V + 著埭`",
        "### 3.4 已然体：重读 `罢` 与句末轻读 `罢`",
        "### 3.5 起始体：`起`",
        "### 3.6 继续体：`落去`",
        "### 9.1 `不`",
        "### 9.2 `未`",
        "### 9.3 `冇` 的否定用法",
    ],
    "marker_to_sections": {
        "爻": ["### 3.1 完成体：`爻`"],
        "罢": ["### 3.4 已然体：重读 `罢` 与句末轻读 `罢`"],
        "著埭": ["### 3.2 进行体：`著埭 + V`", "### 3.3 持续体：`V + 著埭`"],
        "起": ["### 3.5 起始体：`起`"],
        "落去": ["### 3.6 继续体：`落去`"],
        "不": ["### 9.1 `不`"],
        "未": ["### 9.2 `未`"],
        "冇": ["### 9.3 `冇` 的否定用法"],
    },
    "reason_to_sections": {
        "completion_object_order": ["### 3.1 完成体：`爻`"],
        "completion_after_modal": ["### 3.1 完成体：`爻`"],
        "completion_marker_overused": ["### 3.1 完成体：`爻`"],
        "completion_followed_by_clause": ["### 3.1 完成体：`爻`"],
        "sentence_final_ba_overused": ["### 3.4 已然体：重读 `罢` 与句末轻读 `罢`"],
        "stative_progressive": ["### 3.2 进行体：`著埭 + V`"],
        "dynamic_postposed_zhedai": ["### 3.3 持续体：`V + 著埭`"],
        "qishi_object_order": ["### 3.5 起始体：`起`"],
        "continuative_object_order": ["### 3.6 继续体：`落去`"],
        "mandarin_negation": ["### 9.1 `不`", "### 9.2 `未`", "### 9.3 `冇` 的否定用法"],
    },
    "prompt_extra_notes": {
        "generation_user": [
            "不要写北部吴语或上海话类词形，比如 `高头 / 交关 / 贪头`；拿不准就换成更稳的本地方言表达。"
        ]
    },
}

EMPTY_GRAMMAR_CONFIG: dict[str, Any] = {
    "review_markers": [],
    "repair_trigger_pattern": "",
    "completion_marker": "",
    "allowed_after_completion": "",
    "max_completion_count": 0,
    "sentence_final_particle": "",
    "max_sentence_final_count": 0,
    "high_risk_patterns": [],
    "modal_verbs_before_completion": {"pattern": ""},
    "generation_section_titles": [],
    "marker_to_sections": {},
    "reason_to_sections": {},
    "prompt_extra_notes": {},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
            continue
        merged[key] = copy.deepcopy(value)
    return merged


@lru_cache(maxsize=1)
def load_grammar_config() -> dict[str, Any]:
    base = WENZHOU_LEGACY_GRAMMAR_CONFIG if ACTIVE_DIALECT_CONFIG.dialect_key == "wenzhou" else EMPTY_GRAMMAR_CONFIG
    payload = copy.deepcopy(base)
    if GRAMMAR_RULES_PATH.exists():
        loaded = yaml.safe_load(GRAMMAR_RULES_PATH.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Grammar rules must be a mapping: {GRAMMAR_RULES_PATH}")
        payload = _deep_merge(payload, loaded)
    return payload


@lru_cache(maxsize=1)
def compiled_high_risk_patterns() -> list[tuple[str, re.Pattern[str]]]:
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for row in load_grammar_config().get("high_risk_patterns") or []:
        name = str((row or {}).get("name") or "").strip()
        pattern = str((row or {}).get("pattern") or "").strip()
        if not name or not pattern:
            continue
        compiled.append((name, re.compile(pattern)))
    return compiled


@lru_cache(maxsize=1)
def compiled_repair_trigger_pattern() -> re.Pattern[str] | None:
    pattern = str(load_grammar_config().get("repair_trigger_pattern") or "").strip()
    return re.compile(pattern) if pattern else None


@lru_cache(maxsize=1)
def compiled_modal_before_completion_pattern() -> re.Pattern[str] | None:
    payload = load_grammar_config().get("modal_verbs_before_completion") or {}
    pattern = str(payload.get("pattern") or "").strip()
    return re.compile(pattern) if pattern else None
