from __future__ import annotations

from .dialect import ACTIVE_DIALECT_CONFIG
from .grammar_config import (
    compiled_high_risk_patterns,
    compiled_modal_before_completion_pattern,
    compiled_repair_trigger_pattern,
    load_grammar_config,
)
from .grammar_spec import generation_spec_excerpt, relevant_spec_excerpt
from .paths import GRAMMAR_SPEC_PATH


def _prompt_extra_notes(section: str) -> list[str]:
    payload = load_grammar_config().get("prompt_extra_notes") or {}
    notes = payload.get(section) or []
    return [str(note).strip() for note in notes if str(note).strip()]


def build_generation_grammar_prompt_rules() -> str:
    excerpt = generation_spec_excerpt(max_lines=3).strip()
    lines: list[str] = []
    if excerpt:
        lines.append(f"以下规范摘录直接来自 `{GRAMMAR_SPEC_PATH.name}`，生成和校验都要按它走：")
        lines.append(excerpt)
    notes = _prompt_extra_notes("generation_system")
    if notes:
        lines.append("额外提醒：" + " ".join(notes))
    return "\n".join(lines).strip()


def build_generation_grammar_user_rules() -> str:
    excerpt = generation_spec_excerpt(max_lines=2).strip()
    lines: list[str] = []
    if excerpt:
        lines.append(f"请直接参照 `{GRAMMAR_SPEC_PATH.name}` 的这些关键规范摘录：")
        lines.append(excerpt)
    notes = _prompt_extra_notes("generation_user")
    if notes:
        lines.append("额外提醒：" + " ".join(notes))
    return "\n\n".join(lines).strip()


def build_grammar_repair_system_prompt() -> str:
    dialect_name = ACTIVE_DIALECT_CONFIG.dialect_name
    standard_language_label = ACTIVE_DIALECT_CONFIG.standard_language_label
    return f"""你是{dialect_name}语法修订助手。你的任务是最小幅度修改已有{dialect_name}句子，让它更符合{dialect_name}语法规范。

硬规则：
1. 优先修正高风险功能词、体貌、否定和语序问题。
2. 保留原句核心语义、场景和必用词，除非原句的功能词用法明显不对。
3. 如果拿不准某个功能词是否该保留，宁可删掉或改成更稳妥的{dialect_name}说法，不要硬加。
4. 不要把句子改成{standard_language_label}；也不要为了修语法再引入新的高风险功能词。
5. 句子仍需适合日常口语和语音训练朗读。

只输出 JSON：
{{"wz": "修订后的方言句子", "zh": "对应{standard_language_label}翻译"}}"""


GRAMMAR_PROMPT_RULES = build_generation_grammar_prompt_rules()
GRAMMAR_USER_RULES = build_generation_grammar_user_rules()
GRAMMAR_REPAIR_SYSTEM_PROMPT = build_grammar_repair_system_prompt()


def grammar_validation_reasons(text: str) -> list[str]:
    clean = "".join(str(text or "").split())
    grammar_config = load_grammar_config()
    reasons: list[str] = []
    for reason, pattern in compiled_high_risk_patterns():
        if pattern.search(clean):
            reasons.append(reason)
    completion_marker = str(grammar_config.get("completion_marker") or "").strip()
    allowed_after_completion = set(str(grammar_config.get("allowed_after_completion") or ""))
    max_completion_count = int(grammar_config.get("max_completion_count") or 0)
    sentence_final_particle = str(grammar_config.get("sentence_final_particle") or "").strip()
    max_sentence_final_count = int(grammar_config.get("max_sentence_final_count") or 0)
    modal_before_completion = compiled_modal_before_completion_pattern()
    if completion_marker and modal_before_completion and modal_before_completion.search(clean):
        reasons.append("completion_after_modal")
    if completion_marker and allowed_after_completion:
        start = 0
        while True:
            idx = clean.find(completion_marker, start)
            if idx < 0:
                break
            next_idx = idx + len(completion_marker)
            if next_idx < len(clean):
                next_char = clean[next_idx]
                if next_char not in allowed_after_completion:
                    reasons.append(f"completion_followed_by_clause:{next_char}")
                    break
            start = next_idx
    if completion_marker and max_completion_count > 0 and clean.count(completion_marker) > max_completion_count:
        reasons.append("completion_marker_overused")
    if sentence_final_particle and max_sentence_final_count > 0 and clean.count(sentence_final_particle) > max_sentence_final_count:
        if sentence_final_particle == "罢":
            reasons.append("sentence_final_ba_overused")
        else:
            reasons.append(f"sentence_final_particle_overused:{sentence_final_particle}")
    return reasons


def should_attempt_grammar_repair(text: str) -> bool:
    clean = "".join(str(text or "").split())
    repair_trigger = compiled_repair_trigger_pattern()
    return bool(repair_trigger and repair_trigger.search(clean))


def build_grammar_repair_user_prompt(
    *,
    scene_id: str,
    core_word: str,
    support_words: list[str],
    wz_sentence: str,
    zh_sentence: str,
    reasons: list[str],
) -> str:
    support_block = "、".join(support_words) if support_words else "无"
    reasons_block = "、".join(reasons) if reasons else "高风险功能词复核"
    spec_excerpt = relevant_spec_excerpt(wz_sentence, reasons, max_lines=4)
    spec_block = (
        f"请直接按下面这份规范摘录判断：\n{spec_excerpt}"
        if spec_excerpt
        else "当前未配置可引用的语法摘录，请只做最小必要修改。"
    )
    return f"""场景：{scene_id}
核心词：{core_word or "无"}
辅助词：{support_block}

原句：
温州话：{wz_sentence}
普通话：{zh_sentence}

当前需要重点检查的问题：
{reasons_block}

请只做最小必要修改，让句子更符合{ACTIVE_DIALECT_CONFIG.dialect_name}语法。
{spec_block}
"""
