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
    "repair_trigger_pattern": "(爻|罢|著埭|落去|吃起|做起|走起|写起|讲起|用起|显|阿是|訾那沃|阿乜|啊不|啊未|啊冇|未|冇)",
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
        {
            "name": "northern_wu_lexeme_jiaoguan",
            "pattern": "交关",
            "description": "混入北部吴语常见词形",
        },
        {
            "name": "ba_after_request_marker",
            "pattern": "(?:快俫|快点|覅再|着紧|妆紧|趁热|快走|走归|先).{0,10}罢(?:[？?！!。.]|$)",
            "description": "罢 不用于尚未发生的请求/命令或提醒动作",
        },
        {
            "name": "ba_clause_chain_after_completion",
            "pattern": "爻罢[，,].{0,8}(?:等|再|就|然后)",
            "description": "爻罢 不宜直接挂接下一分句，应用更稳妥的已然表达",
        },
        {
            "name": "future_start_marker_after_modal",
            "pattern": "(?:想|要|会|會|打算|准备|準備).{0,10}(?:吃起|做起|走起|写起|讲起|读起|用起|开起)(?:[？?！!。.]|$)",
            "description": "未发生事件不宜直接把 起 当成起始体结果收尾",
        },
        {
            "name": "future_continuative_marker_after_modal",
            "pattern": "(?:想|要|会|會|打算|准备|準備).{0,10}(?:做落去|走落去|开落去|卖落去|吃落去|用落去)(?:[？?！!。.]|$)",
            "description": "未发生事件不宜直接把 落去 当成继续体结果收尾",
        },
        {
            "name": "ba_realis_misuse",
            "pattern": "(?:(?:着|要|会|會|应该|應該|打算|准备|準備).{0,10}罢(?:[？?！!。.]|$)|(?:走归|走出).{0,4}罢(?:[？?！!。.]|$))",
            "description": "罢 只用于事件已成立；提醒、计划、将来动作后不裸加 罢",
        },
        {
            "name": "subjective_emotion_xianxian",
            "pattern": "(?:我|阿拉|我个|自家).{0,3}(?:急|烦|烦闷|心焦|心烦|担心)显(?:急|烦|烦闷|心焦|心烦|担心)",
            "description": "自身主观情感默认不用 X显X",
        },
        {
            "name": "bare_xian_degree",
            "pattern": "(?:困难显(?:显)?|着急显|心焦显|心烦显|麻烦显|难寻显(?:罢)?|好走显(?:罢)?|方便显(?:罢)?|熟显(?:罢)?)(?:[？?！!。.]|$)",
            "description": "X显 不能粗放裸收尾，优先改成更完整稳妥的程度表达",
        },
        {
            "name": "bare_start_marker_qi",
            "pattern": "(?:走起|吃起|做起)(?:[，,。！？]|$)",
            "description": "起 不作粗放句末骨架，优先改成更稳妥的起始或动作表达",
        },
        {
            "name": "completion_ba_motion_or_setup",
            "pattern": "(?:走爻罢|开爻罢|叫爻罢|焯菜爻罢|寻著爻罢)(?:[？?！!。]|$|[，,])",
            "description": "机械式 V爻罢 组合不稳，优先改成更自然的完成/已然表达",
        },
        {
            "name": "completion_marker_nonadverse_action",
            "pattern": "(?:点爻(?:罢)?|焯菜爻(?:罢)?|煮爻(?:罢)?|避雨爻(?:罢)?|不出门爻(?:罢)?|不出去爻(?:罢)?|出门爻(?:罢)?|出去爻(?:罢)?|等雨停爻(?:再走)?)(?:[？?！!。]|$|[，,])",
            "description": "爻 带消极收束意味，不当一般完成体；中性或正向动作后不机械加 爻",
        },
        {
            "name": "preposed_degree_with_xian",
            "pattern": "(?:恁|忒|蛮)\\w{0,5}显",
            "description": "前置程度副词（恁/忒/蛮）与后置显叠加，应去掉其中一个",
        },
        {
            "name": "zhenzhen_xian",
            "pattern": "真真.{0,6}显",
            "description": "真真...显是旧坏骨架，优先改成更朴素稳妥的程度表达",
        },
        {
            "name": "distributive_vv_object_order",
            "pattern": "(?P<v>[\\u4e00-\\u9fff]{1,2})(?:该|许|这|那)[^，。！？,.!?]{1,12}(?P=v)(?:该|许|这|那)",
            "description": "分配式动词重叠语序错误：应为 OBJ+VV，OBJ+VV，不写成 V+OBJ+V+OBJ",
        },
        {
            "name": "mao_for_unfinished_event",
            "pattern": "还冇(?:开声|开始|寄出|发出)",
            "description": "未发生/未完成事件优先用 未，不写 还冇开声/还冇开始 这类格式",
        },
        {
            "name": "a_mie_as_how_question",
            "pattern": "阿乜(?:妆|办|做|讲)",
            "description": "阿乜 表示什么，不替代 怎么/如何；how-question 优先用 訾那",
        },
        {
            "name": "nonhuman_qu_pronoun",
            "pattern": "渠(?:恁|忒|蛮|真).{0,4}(?:厚|高|贵|灵清|方便|要紧)",
            "description": "渠 默认指人，不默认指代物件或非人对象",
        },
        {
            "name": "second_person_subjective_xianxian",
            "pattern": "你.{0,6}(?:烦闷|烦|急|累)显(?:烦闷|烦|急|累)",
            "description": "主观情绪类 X显X 默认不用第二人称承载",
        },
        {
            "name": "postposed_destination_after_qu",
            "pattern": "走去(?:车站|地铁站|高铁站|医院|学校|机场)",
            "description": "位移方向默认用地点前置：走车站去，不写 走去车站",
        },
        {
            "name": "future_completion_ba",
            "pattern": "(?:修起罢|开声爻罢|寄出罢)",
            "description": "未发生或将来事件不应误用完成体/句末罢",
        },
        {
            "name": "xian_non_adjective_base",
            "pattern": "(?:糟塌|支付|走|写|读|讲|说|吃|睡|坐|站|跑|买|卖|修|寄|开声)显(?:糟塌|支付|走|写|读|讲|说|吃|睡|坐|站|跑|买|卖|修|寄|开声)",
            "description": "X显X 只限形容词或状态词，不用于动词、动作结果或耗损过程",
        },
        {
            "name": "object_fronted_reduplication",
            "pattern": "(?:眙眙|问问|听听)(?:该|许|这|那)[^，。！？,.!?]{1,12}",
            "description": "尝试义动词重叠默认宾语前置：OBJ+V+VV，不写 V+VV+OBJ",
        },
        {
            "name": "constructional_core_misanalysis",
            "pattern": "赶不逮记牢|赶不逮写牢|赶不逮背牢",
            "description": "赶不逮 是构式资源，不和 记牢/写牢/背牢 这类结果义补语硬拼",
        },
        {
            "name": "a_shi_template_drift",
            "pattern": "(?<!是)阿是",
            "description": "阿是 不作自由疑问骨架；只有明确 yes/no alternative 才写 是阿是",
        },
        {
            "name": "wo_used_for_scalar_also",
            "pattern": "(?:我|你|渠|伊|阿拉|俫|人家).{0,3}沃(?:还|先|就|才|已经|已)",
            "description": "沃 主要管全称量化；标量或让步语境优先用 阿",
        },
        {
            "name": "a_used_for_quantificational_all",
            "pattern": "(?:这些|这些个|大家|人人|逐个|全部|都个).{0,4}阿(?:还|都|会|要|已经|已)",
            "description": "全体/总括义优先用 沃，不把 quantificational all 写成 阿",
        },
        {
            "name": "choice_question_template",
            "pattern": "(?:是阿是.{0,8}听底罢(?!阿未)|眙着未(?:[？?！!。.]|$)|足也未(?:[？?！!。.]|$))",
            "description": "选择问句模板错误：优先用 V不V / V冇V / V罢阿未 / X也不X",
        },
    ],
    "modal_verbs_before_completion": {
        "pattern": "(?:想|要|会|會|能|可以|应该|應該|打算|喜欢|曉得|晓得|觉得|覺得|认识|認識|希望|准备|準備|肯|敢)爻"
    },
    "generation_full_section_titles": [
        "## 12. 批量生成硬约束清单",
        "## 13. 不推荐直接生成的高风险式",
    ],
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
        "ba_after_request_marker": ["### 3.4 已然体：重读 `罢` 与句末轻读 `罢`"],
        "ba_clause_chain_after_completion": ["### 3.1 完成体：`爻`", "### 3.4 已然体：重读 `罢` 与句末轻读 `罢`"],
        "stative_progressive": ["### 3.2 进行体：`著埭 + V`"],
        "dynamic_postposed_zhedai": ["### 3.3 持续体：`V + 著埭`"],
        "qishi_object_order": ["### 3.5 起始体：`起`"],
        "continuative_object_order": ["### 3.6 继续体：`落去`"],
        "future_start_marker_after_modal": ["### 3.5 起始体：`起`"],
        "future_continuative_marker_after_modal": ["### 3.6 继续体：`落去`"],
        "ba_realis_misuse": ["### 3.4 已然体：重读 `罢` 与句末轻读 `罢`"],
        "mandarin_negation": ["### 9.1 `不`", "### 9.2 `未`", "### 9.3 `冇` 的否定用法"],
        "northern_wu_lexeme_jiaoguan": [],
        "subjective_emotion_xianxian": [],
        "bare_xian_degree": [],
        "bare_start_marker_qi": ["### 3.5 起始体：`起`"],
        "completion_ba_motion_or_setup": ["### 3.1 完成体：`爻`", "### 3.4 已然体：重读 `罢` 与句末轻读 `罢`"],
        "completion_marker_nonadverse_action": ["### 3.1 完成体：`爻`"],
        "preposed_degree_with_xian": ["## 13. 不推荐直接生成的高风险式"],
        "zhenzhen_xian": ["## 13. 不推荐直接生成的高风险式"],
        "distributive_vv_object_order": ["## 12. 批量生成硬约束清单", "## 14. 可复用句式模板"],
        "mao_for_unfinished_event": ["### 9.2 `未`", "### 9.3 `冇` 的否定用法", "### 9.4 三者不可混用"],
        "a_mie_as_how_question": ["## 13. 不推荐直接生成的高风险式"],
        "nonhuman_qu_pronoun": ["## 13. 不推荐直接生成的高风险式"],
        "second_person_subjective_xianxian": ["## 13. 不推荐直接生成的高风险式"],
        "postposed_destination_after_qu": ["## 13. 不推荐直接生成的高风险式"],
        "future_completion_ba": ["### 3.1 完成体：`爻`", "### 3.4 已然体：重读 `罢` 与句末轻读 `罢`"],
        "xian_non_adjective_base": ["## 13. 不推荐直接生成的高风险式"],
        "object_fronted_reduplication": ["## 12. 批量生成硬约束清单", "## 14. 可复用句式模板"],
        "constructional_core_misanalysis": ["## 13. 不推荐直接生成的高风险式"],
        "a_shi_template_drift": [],
        "wo_used_for_scalar_also": [],
        "a_used_for_quantificational_all": [],
        "choice_question_template": [],
    },
    "prompt_extra_notes": {
        "generation_user": [
            "不要写北部吴语或上海话类词形，比如 `高头 / 交关 / 贪头`；拿不准就换成更稳的本地方言表达。",
            "`爻` 带消极、收束性的完结意味，不当一般完成体；中性或正向动作后不要机械加 `爻 / 爻罢`。",
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
