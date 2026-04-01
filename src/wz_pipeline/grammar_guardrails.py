from __future__ import annotations

import re

from .grammar_spec import GRAMMAR_SPEC_PATH, generation_spec_excerpt, relevant_spec_excerpt

GRAMMAR_PROMPT_RULES = (
    f"以下规范摘录直接来自 `{GRAMMAR_SPEC_PATH.name}`，生成和校验都要按它走：\n"
    + generation_spec_excerpt(max_lines=3)
)

GRAMMAR_USER_RULES = (
    f"请直接参照 `{GRAMMAR_SPEC_PATH.name}` 的这些关键规范摘录：\n"
    + generation_spec_excerpt(max_lines=2)
    + "\n\n额外提醒：不要写北部吴语/上海话类词形，比如 `高头 / 交关 / 贪头`；不会说就换成更稳的温州话表达。"
)

GRAMMAR_REPAIR_SYSTEM_PROMPT = """你是温州话语法修订助手。你的任务是最小幅度修改已有温州话句子，让它更符合温州话语法规范。

硬规则：
1. 优先修正 `爻`、`罢`、`著埭`、`起`、`落去`、`不/未/冇` 的误用。
2. 保留原句核心语义、场景和必用词，除非原句的功能词用法明显不对。
3. 如果拿不准某个功能词是否该保留，宁可删掉或改成更稳妥的普通温州话说法，不要硬加。
4. 不要把句子改成普通话；也不要为了修语法再引入新的高风险功能词。
5. 句子仍需适合日常口语和语音训练朗读。

只输出 JSON：
{"wz": "修订后的温州话句子", "zh": "对应普通话翻译"}"""

GRAMMAR_REPAIR_TRIGGER_RE = re.compile(r"(爻|罢|著埭|落去|起罢|啊不|啊未|啊冇|未|冇)")
HIGH_RISK_LITERAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("completion_object_order", re.compile(r"跳爻舞")),
    ("stative_progressive", re.compile(r"著埭(?:坐|晓得|快活)")),
    ("dynamic_postposed_zhedai", re.compile(r"(?:走|笑|飞)著埭")),
    ("qishi_object_order", re.compile(r"唱歌起罢")),
    ("continuative_object_order", re.compile(r"一直开会落去")),
    ("mandarin_negation", re.compile(r"(?:没有|沒有)")),
    ("northern_wu_lexeme_gaotou", re.compile(r"高头")),
]
LIAO_AFTER_MODAL_RE = re.compile(
    r"(?:想|要|会|會|能|可以|应该|應該|打算|喜欢|曉得|晓得|觉得|覺得|认识|認識|希望|准备|準備|肯|敢)爻"
)
ALLOWED_AFTER_LIAO = set("罢，。！？；,.!?;、再就还也阿沃个")


def grammar_validation_reasons(text: str) -> list[str]:
    clean = "".join(str(text or "").split())
    reasons: list[str] = []
    for reason, pattern in HIGH_RISK_LITERAL_PATTERNS:
        if pattern.search(clean):
            reasons.append(reason)
    if LIAO_AFTER_MODAL_RE.search(clean):
        reasons.append("completion_after_modal")
    for idx, char in enumerate(clean[:-1]):
        if char != "爻":
            continue
        next_char = clean[idx + 1]
        if next_char not in ALLOWED_AFTER_LIAO:
            reasons.append(f"completion_followed_by_clause:{next_char}")
            break
    if clean.count("爻") > 1:
        reasons.append("completion_marker_overused")
    if clean.count("罢") > 1:
        reasons.append("sentence_final_ba_overused")
    return reasons


def should_attempt_grammar_repair(text: str) -> bool:
    clean = "".join(str(text or "").split())
    return bool(GRAMMAR_REPAIR_TRIGGER_RE.search(clean))


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
    return f"""场景：{scene_id}
核心词：{core_word or "无"}
辅助词：{support_block}

原句：
温州话：{wz_sentence}
普通话：{zh_sentence}

当前需要重点检查的问题：
{reasons_block}

请只做最小必要修改，让句子更符合温州话语法。请直接按下面这份规范摘录判断：
{spec_excerpt}
"""
