#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECORDS = ROOT / "data" / "cleaned" / "cleaned_records_primary.jsonl"
DEFAULT_HEURISTIC = ROOT / "data" / "llm_review" / "mandarin_headwords_heuristic.jsonl"
DEFAULT_API = ROOT / "data" / "llm_review" / "mandarin_headwords.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "controlled_generation"
DEFAULT_NEW_NOUNS = ROOT / "新事物名词.xlsx"
DEFAULT_PLACE_NAMES = ROOT / "温州地名300个.xlsx"
DEFAULT_EXAMPLE_MINED = ROOT / "data" / "controlled_generation" / "assets" / "example_mined_lexicon.jsonl"
DEFAULT_SAMPLE_SIZE = 200
MIN_LEXICON_CONFIDENCE = 0.68

SYSTEM_PROMPT = (
    "你是温州话句子受控生成助手。"
    "必须保留模板中的固定片段、虚词、语气、标点和句式顺序，只能替换槽位内容。"
    "如果模板里没有 `爻`、`罢`、`著埭`、`起`、`落去`、`不/未/冇` 这类功能词，不要为了像方言而自行新增。"
    "如果模板里已经有这些功能词，也不要改坏其句法位置。"
    "如果 slots 提供了 semantic_class，替换后的内容必须和该槽位角色兼容。"
    "优先替换尽可能多的实词槽位。"
    "如果有两个及以上可替换槽位，默认至少替换两个，不要只改一个槽位就交差。"
    "输出自然的温州话句子，适合日常对话和语音训练。"
    "required_words 必须出现，preferred_words 尽量自然融入。"
    "只输出 JSON。"
)

SCENE_CATALOG: list[dict[str, Any]] = [
    {
        "scene_id": "daily_chat",
        "label": "日常闲聊",
        "intents": ["闲聊", "情绪表达", "轻对话"],
        "zh_keywords": ["说", "讲", "听", "知道", "觉得", "事情", "等下", "回去", "聊天", "脾气"],
        "wz_keywords": ["讲", "晓得", "事干", "走归"],
        "default_required_modern_words": ["手机"],
        "default_preferred_modern_words": ["消息", "视频", "照片"],
    },
    {
        "scene_id": "home_life",
        "label": "居家生活",
        "intents": ["家务", "家庭生活", "家居对话"],
        "zh_keywords": ["家里", "回家", "奶奶", "孩子", "房子", "做饭", "睡", "屋里", "家务", "空调", "洗衣机"],
        "wz_keywords": ["屋里", "娘娘", "细儿", "睏", "屋宕"],
        "default_required_modern_words": ["空调"],
        "default_preferred_modern_words": ["洗衣机", "扫地机器人", "外卖"],
    },
    {
        "scene_id": "digital_chat",
        "label": "数字沟通",
        "intents": ["手机问答", "数字沟通", "设备使用"],
        "zh_keywords": ["电话", "录音", "消息", "手机", "微信", "视频", "打电话", "充电器", "充电宝", "智能手表", "电脑", "笔记本", "平板", "iPad", "密码"],
        "wz_keywords": ["电话", "录音"],
        "default_required_modern_words": ["手机"],
        "default_preferred_modern_words": ["微信", "视频", "消息"],
    },
    {
        "scene_id": "shopping_payment",
        "label": "购物支付",
        "intents": ["网购问答", "支付退款", "快递物流"],
        "zh_keywords": ["买", "卖", "钱", "块钱", "付款", "顾客", "商店", "快递", "网购", "退款", "优惠券", "自动取款机", "拼多多", "淘宝", "支付宝", "专卖店"],
        "wz_keywords": ["买", "钞票", "主顾", "卖"],
        "default_required_modern_words": ["付款码"],
        "default_preferred_modern_words": ["外卖", "快递", "优惠券"],
    },
    {
        "scene_id": "food_dining",
        "label": "饮食吃饭",
        "intents": ["吃饭点单", "口味评价", "外卖问答"],
        "zh_keywords": ["吃", "喝", "饭", "菜", "面", "口味", "点单", "奶茶", "配送", "外卖"],
        "wz_keywords": ["吃", "喝", "饭", "面"],
        "default_required_modern_words": ["外卖"],
        "default_preferred_modern_words": ["奶茶", "点单", "配送"],
    },
    {
        "scene_id": "transport_trip",
        "label": "出行交通",
        "intents": ["交通规划问答", "路线询问", "出行计划"],
        "zh_keywords": ["去", "回来", "车", "公交", "地铁", "高铁", "导航", "路上", "路线", "站台", "网约车", "打车"],
        "wz_keywords": ["走", "车", "路里"],
        "default_required_modern_words": ["导航"],
        "default_preferred_modern_words": ["地铁", "高铁", "打车"],
    },
    {
        "scene_id": "weather_safety",
        "label": "天气安全",
        "intents": ["天气提醒", "安全提醒", "应急准备"],
        "zh_keywords": ["台风", "下雨", "风", "准备", "救生", "口罩", "天气", "预报", "预警", "应急"],
        "wz_keywords": ["台风", "落雨"],
        "default_required_modern_words": ["天气预报"],
        "default_preferred_modern_words": ["预警", "口罩", "应急包"],
    },
    {
        "scene_id": "work_study",
        "label": "工作学习",
        "intents": ["工作协同", "学习问答", "会议文档"],
        "zh_keywords": ["公司", "工作", "负责", "记录", "保存", "学习", "老师", "作业", "会议", "文档", "打卡", "课件", "电脑", "笔记本"],
        "wz_keywords": ["公司", "工作", "负责"],
        "default_required_modern_words": ["会议"],
        "default_preferred_modern_words": ["文档", "打卡", "课件"],
    },
    {
        "scene_id": "health_medical",
        "label": "健康医疗",
        "intents": ["生活知识问答", "健康问答", "看病流程"],
        "zh_keywords": ["医院", "医生", "医师", "病人", "肚子痛", "看病", "看医生", "体检", "挂号", "消毒", "报告", "核酸", "吃药", "药片"],
        "wz_keywords": ["医院", "医师", "挂号", "体检"],
        "default_required_modern_words": ["挂号"],
        "default_preferred_modern_words": ["体检", "核酸", "报告"],
    },
]

SPEECH_FUNCTION_CATALOG: list[dict[str, Any]] = [
    {"function_id": "question", "label": "提问", "keywords": ["?", "吗", "么", "什么", "怎么", "几", "哪", "谁", "会不会", "晓得晓不得"]},
    {"function_id": "reminder", "label": "提醒建议", "keywords": ["要", "着", "应该", "不用", "别", "快", "记得", "最好"]},
    {"function_id": "explanation", "label": "解释说明", "keywords": ["因为", "所以", "照讲", "就是", "意思", "说明", "按理说"]},
    {"function_id": "planning", "label": "计划安排", "keywords": ["准备", "打算", "想", "安排", "明朝", "等下"]},
    {"function_id": "evaluation", "label": "评价吐槽", "keywords": ["真", "显", "烦", "难过", "不好", "好显好", "马马虎虎"]},
    {"function_id": "statement", "label": "陈述", "keywords": []},
]

FUNCTION_DEFINITION_PREFIXES = (
    "量词",
    "助词",
    "语气词",
    "代词",
    "介词",
    "连词",
    "副词",
    "数词",
    "叹词",
)
ADJECTIVE_HINTS = (
    "形容",
    "不灵活",
    "笨拙",
    "坚韧",
    "谨慎",
    "爽快",
    "难过",
    "好",
    "坏",
    "害羞",
    "羞怯",
    "羞",
    "怯",
    "冷酷",
    "热闹",
    "合算",
    "不错",
)
VERB_HINTS = (
    "发觉",
    "拦阻",
    "支撑",
    "顶住",
    "蒸",
    "欺负",
    "看",
    "走",
    "买",
    "吃",
    "喝",
    "讲",
    "录",
    "保存",
    "准备",
)
TIME_HINTS = (
    "昨天",
    "今天",
    "明天",
    "夜里",
    "日里",
    "正月",
    "以前",
    "后来",
    "夏天",
    "暑天",
    "黄昏",
    "钟头",
    "时候",
    "辰光",
    "半夜",
    "早起",
)
PLACE_HINTS = ("屋里", "家里", "公园", "街", "医院", "公司", "学校", "山里", "路上")
NOUN_HINTS = (
    "人",
    "物",
    "东西",
    "植物",
    "动物",
    "帽子",
    "房子",
    "药",
    "鱼",
    "虾",
    "病",
    "顾客",
    "朋友",
    "故事",
    "公园",
    "电话",
    "孩子",
)
BAD_HEADWORDS = {
    "的",
    "了",
    "呢",
    "吗",
    "啊",
    "吧",
    "和",
    "跟",
    "把",
    "被",
    "都",
    "只",
    "再",
    "还",
    "就",
}
BAD_WZ_WORDS = {"个", "阿", "沃", "起", "只", "了", "啊", "吗", "呢"}
DENSE_LOCKED_ZH_WORDS = {
    "我",
    "你",
    "他",
    "她",
    "它",
    "我们",
    "你们",
    "他们",
    "别人",
    "自己",
    "谁",
    "什么",
    "这",
    "那",
    "这里",
    "那里",
    "这个",
    "那个",
    "大家",
}
DENSE_LOCKED_SLOT_KINDS = {"function", "time"}
DENSE_LOCKED_SEMANTIC_CLASSES = {"unknown", "time"}
DENSE_HIGH_PRIORITY_SEMANTIC_CLASSES = {
    "place",
    "institution",
    "food",
    "device",
    "transport",
    "medicine",
    "document",
    "service",
    "person",
}
DENSE_MEDIUM_PRIORITY_SEMANTIC_CLASSES = {"animal", "adjective", "action"}
SLOT_RE = re.compile(r"\[SLOT_(\d+)\]")
SEMANTIC_CLASS_HINTS: dict[str, tuple[str, ...]] = {
    "service": ("人工智能", "人脸识别", "识别", "网站", "游戏", "直播", "快手", "抖音", "京东", "美团", "拼多多", "淘宝", "支付宝", "付款码", "话费", "大数据", "流量"),
    "person": ("老师", "医生", "医师", "顾客", "主顾", "旅客", "孩子", "细儿", "阿伯", "朋友", "人"),
    "place": ("路", "街", "巷", "桥", "村", "山", "江", "站", "机场", "屋里", "家里", "公园"),
    "institution": ("医院", "学校", "公司", "店", "超市", "药店", "工厂", "车间", "奶茶店", "品牌店"),
    "food": ("苹果", "香蕉", "葡萄", "西瓜", "牛奶", "鸡蛋", "奶茶", "米饭", "饭", "菜", "面", "外卖", "饮料"),
    "animal": ("苍蝇", "鸟", "鱼", "虾", "虫", "狗", "猫", "牛", "鸡", "鸭"),
    "device": ("手机", "电脑", "电视", "冰箱", "空调", "风扇", "充电桩", "充电器", "充电宝", "蓝牙耳机", "耳机", "门锁", "洗衣机", "热水器", "扫地机器人"),
    "transport": ("公交", "地铁", "高铁", "飞机", "网约车", "打车", "电动车", "自行车", "导航"),
    "medicine": ("挂号", "体检", "核酸", "消毒", "药片", "吃药", "报告", "口罩"),
    "document": ("文档", "作业", "课件", "会议", "消息", "照片", "视频", "微信"),
}


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


def load_extraction_map(heuristic_path: Path, api_path: Path) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(heuristic_path):
        merged[row["dedup_exact_key"]] = row
    for row in load_jsonl(api_path):
        merged[row["dedup_exact_key"]] = row
    return merged


def normalize_compact(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", "", text)


def stable_id(prefix: str, *parts: str) -> str:
    key = "::".join(parts)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def stable_bucket(*parts: str, mod: int = 100) -> int:
    key = "::".join(parts)
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % mod


def load_single_column_excel(path: Path) -> list[str]:
    if not path.exists():
        return []
    excel = pd.ExcelFile(path)
    values: list[str] = []
    for sheet in excel.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        if df.empty:
            continue
        column = df.columns[0]
        for value in df[column].dropna().astype(str):
            text = value.strip()
            if text:
                values.append(text)
    deduped = sorted(set(values), key=lambda text: (len(text), text))
    return deduped


def rank_scenes(*texts: str) -> list[dict[str, Any]]:
    merged_text = " ".join(text for text in texts if text)
    ranked: list[dict[str, Any]] = []
    for scene in SCENE_CATALOG:
        matched = [kw for kw in scene["zh_keywords"] if kw in merged_text]
        matched += [kw for kw in scene["wz_keywords"] if kw in merged_text]
        if not matched:
            continue
        ranked.append(
            {
                "scene_id": scene["scene_id"],
                "scene_label": scene["label"],
                "score": len(set(matched)),
                "matched_keywords": sorted(set(matched)),
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["scene_id"]))
    return ranked


def build_scene_payload(*texts: str) -> dict[str, Any]:
    ranked = rank_scenes(*texts)
    if ranked:
        primary = ranked[0]
        total = sum(item["score"] for item in ranked)
        confidence = primary["score"] / max(total, 1)
        return {
            "primary_scene_id": primary["scene_id"],
            "primary_scene_label": primary["scene_label"],
            "primary_scene_confidence": round(confidence, 4),
            "scene_tags": [item["scene_id"] for item in ranked[:3]],
            "scene_scores": {item["scene_id"]: item["score"] for item in ranked[:5]},
            "scene_candidates": ranked[:5],
        }
    fallback = next(scene for scene in SCENE_CATALOG if scene["scene_id"] == "daily_chat")
    return {
        "primary_scene_id": fallback["scene_id"],
        "primary_scene_label": fallback["label"],
        "primary_scene_confidence": 0.0,
        "scene_tags": [fallback["scene_id"]],
        "scene_scores": {fallback["scene_id"]: 0},
        "scene_candidates": [],
    }


def build_speech_function_payload(*texts: str) -> dict[str, Any]:
    merged_text = " ".join(text for text in texts if text)
    ranked: list[dict[str, Any]] = []
    for item in SPEECH_FUNCTION_CATALOG:
        matched = [kw for kw in item["keywords"] if kw and kw in merged_text]
        score = len(set(matched))
        if score == 0 and item["function_id"] != "statement":
            continue
        ranked.append(
            {
                "function_id": item["function_id"],
                "function_label": item["label"],
                "score": score,
                "matched_keywords": sorted(set(matched)),
            }
        )
    ranked.sort(key=lambda row: (-row["score"], row["function_id"]))
    if ranked and ranked[0]["score"] > 0:
        primary = ranked[0]
        total = sum(item["score"] for item in ranked if item["score"] > 0)
        return {
            "primary_speech_function": primary["function_id"],
            "primary_speech_function_label": primary["function_label"],
            "speech_function_confidence": round(primary["score"] / max(total, 1), 4),
            "speech_function_tags": [item["function_id"] for item in ranked if item["score"] > 0][:3],
            "speech_function_scores": {item["function_id"]: item["score"] for item in ranked if item["score"] > 0},
        }
    return {
        "primary_speech_function": "statement",
        "primary_speech_function_label": "陈述",
        "speech_function_confidence": 0.0,
        "speech_function_tags": ["statement"],
        "speech_function_scores": {"statement": 0},
    }


def scene_compatible(scene_tags: list[str], primary_scene_id: str) -> bool:
    if not scene_tags:
        return True
    return primary_scene_id in set(scene_tags)


def guess_slot_kind(headword: str | None, definition: str | None, wz_word: str) -> str:
    headword = headword or ""
    definition = definition or ""
    seed = f"{headword} {definition}"
    if any(definition.startswith(prefix) for prefix in FUNCTION_DEFINITION_PREFIXES):
        return "function"
    if any(hint in seed for hint in TIME_HINTS):
        return "time"
    if any(hint in seed for hint in PLACE_HINTS):
        return "place"
    if any(hint in definition for hint in ADJECTIVE_HINTS):
        return "adjective"
    if any(hint in headword for hint in ADJECTIVE_HINTS):
        return "adjective"
    if any(hint in seed for hint in VERB_HINTS):
        return "verb"
    if any(hint in seed for hint in NOUN_HINTS):
        return "noun"
    if len(wz_word) >= 2 and len(headword) >= 2:
        return "noun"
    return "unknown"


def guess_semantic_class(
    headword: str | None,
    definition: str | None,
    wz_word: str | None,
    slot_kind: str,
    primary_scene_id: str | None = None,
) -> str:
    if slot_kind == "place":
        return "place"
    if slot_kind == "time":
        return "time"
    if slot_kind == "verb":
        return "action"
    if slot_kind == "adjective":
        return "adjective"
    seed = " ".join(part for part in [headword or "", definition or "", wz_word or ""] if part)
    for semantic_class, hints in SEMANTIC_CLASS_HINTS.items():
        if any(hint in seed for hint in hints):
            return semantic_class
    if primary_scene_id == "food_dining":
        return "food"
    if primary_scene_id == "digital_chat":
        return "device"
    if primary_scene_id == "transport_trip":
        return "transport"
    if primary_scene_id == "health_medical":
        return "medicine"
    return "unknown"


def semantic_compatible(term_class: str, slot_class: str) -> bool:
    if term_class == "unknown" or slot_class == "unknown":
        return True
    if term_class == slot_class:
        return True
    if {term_class, slot_class} <= {"place", "institution"}:
        return True
    if {term_class, slot_class} <= {"device", "transport"}:
        return True
    if {term_class, slot_class} <= {"document", "device"}:
        return True
    if {term_class, slot_class} <= {"service", "device"}:
        return True
    if {term_class, slot_class} <= {"service", "document"}:
        return True
    return False


def concrete_slot_classes(slots: list[dict[str, Any]]) -> list[str]:
    return [
        slot_class
        for slot_class in (slot.get("semantic_class", "unknown") for slot in slots)
        if slot_class not in {"unknown", "adjective", "action", "time"}
    ]


def slots_allow_semantic_injection(slots: list[dict[str, Any]]) -> bool:
    slot_classes = [
        slot.get("semantic_class", "unknown")
        for slot in slots
        if slot.get("slot_kind") != "function"
    ]
    if not slot_classes:
        return False
    return all(slot_class not in {"unknown", "adjective", "action", "time"} for slot_class in slot_classes)


def filter_terms_by_slot_semantics(term_rows: list[dict[str, Any]], slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not term_rows:
        return []
    slot_classes = concrete_slot_classes([slot for slot in slots if slot.get("slot_kind") != "function"])
    if not slot_classes:
        return []
    filtered = [
        row
        for row in term_rows
        if row.get("semantic_class", "unknown") != "unknown"
        if any(semantic_compatible(row.get("semantic_class", "unknown"), slot_class) for slot_class in slot_classes)
    ]
    return filtered


def filter_words_by_slot_semantics(words: list[str], scene_id: str, slots: list[dict[str, Any]]) -> list[str]:
    slot_classes = concrete_slot_classes([slot for slot in slots if slot.get("slot_kind") != "function"])
    if not slot_classes:
        return []
    filtered: list[str] = []
    for word in words:
        semantic_class = guess_semantic_class(word, word, word, "noun", primary_scene_id=scene_id)
        if any(semantic_compatible(semantic_class, slot_class) for slot_class in slot_classes):
            filtered.append(word)
    return filtered


def build_dense_semantic_index(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        if not entry.get("usable_as_slot"):
            continue
        semantic_class = entry.get("semantic_class", "unknown")
        if semantic_class == "unknown":
            continue
        index.setdefault(semantic_class, []).append(entry)
    return index


def dense_slot_priority(slot: dict[str, Any]) -> str:
    semantic_class = slot.get("semantic_class", "unknown")
    if semantic_class in DENSE_HIGH_PRIORITY_SEMANTIC_CLASSES:
        return "high"
    if semantic_class in DENSE_MEDIUM_PRIORITY_SEMANTIC_CLASSES:
        return "medium"
    return "low"


def slot_looks_like_pronoun(slot: dict[str, Any]) -> bool:
    surface_zh = str(slot.get("surface_zh") or "").strip()
    surface_wz = str(slot.get("surface_wz") or "").strip()
    if surface_zh in DENSE_LOCKED_ZH_WORDS:
        return True
    if surface_wz in {"我", "你", "渠", "伊", "阿拉", "你拉"}:
        return True
    return False


def slot_looks_like_meta_headword(slot: dict[str, Any]) -> bool:
    surface_zh = str(slot.get("surface_zh") or "").strip()
    if not surface_zh:
        return False
    meta_hints = (
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
        "形容词",
        "动词",
        "名词",
        "补语",
        "状语",
    )
    return any(hint in surface_zh for hint in meta_hints)


def rank_dense_slot_candidates(
    slot: dict[str, Any],
    scene_id: str,
    semantic_index: dict[str, list[dict[str, Any]]],
    preview_limit: int = 12,
) -> tuple[list[dict[str, Any]], int]:
    slot_class = slot.get("semantic_class", "unknown")
    slot_kind = slot.get("slot_kind", "unknown")
    source_term = slot.get("surface_wz")
    if slot_class == "unknown":
        return [], 0

    candidate_classes = [
        semantic_class
        for semantic_class in semantic_index
        if semantic_compatible(semantic_class, slot_class)
    ]
    ranked: list[tuple[tuple[int, int, int, int, float, str], dict[str, Any]]] = []
    seen_terms: set[str] = set()
    total_terms: set[str] = set()

    for semantic_class in candidate_classes:
        for row in semantic_index.get(semantic_class, []):
            term = row.get("wz_word")
            if not term or term == source_term or term in seen_terms:
                continue
            is_scene_primary = int(row.get("primary_scene_id") == scene_id)
            is_scene_compatible = int(scene_compatible(row.get("scene_tags") or [], scene_id))
            if not is_scene_compatible and not is_scene_primary:
                continue
            total_terms.add(term)
            score_key = (
                int(row.get("semantic_class") == slot_class),
                int(row.get("slot_kind") == slot_kind),
                is_scene_primary,
                is_scene_compatible,
                -abs(len(term) - len(str(source_term or ""))),
                float(row.get("confidence", 0.0)),
                term,
            )
            ranked.append((score_key, row))

    ranked.sort(key=lambda item: item[0], reverse=True)
    preview_rows: list[dict[str, Any]] = []
    for _, row in ranked:
        term = row["wz_word"]
        if term in seen_terms:
            continue
        seen_terms.add(term)
        preview_rows.append(
            {
                "term": term,
                "mandarin_headword": row.get("mandarin_headword"),
                "semantic_class": row.get("semantic_class", "unknown"),
                "slot_kind": row.get("slot_kind", "unknown"),
                "primary_scene_id": row.get("primary_scene_id"),
                "confidence": round(float(row.get("confidence", 0.0)), 4),
                "record_type": row.get("record_type"),
                "source_file": row.get("source_file"),
            }
        )
        if len(preview_rows) >= preview_limit:
            break
    return preview_rows, len(total_terms)


def annotate_dense_slot(
    slot: dict[str, Any],
    scene_id: str,
    semantic_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    reasons: list[str] = []
    slot_kind = slot.get("slot_kind", "unknown")
    semantic_class = slot.get("semantic_class", "unknown")
    replaceable = True

    if slot_kind in DENSE_LOCKED_SLOT_KINDS:
        replaceable = False
        reasons.append(f"slot_kind_{slot_kind}")
    if semantic_class in DENSE_LOCKED_SEMANTIC_CLASSES:
        replaceable = False
        reasons.append(f"semantic_class_{semantic_class}")
    if slot_looks_like_pronoun(slot):
        replaceable = False
        reasons.append("pronoun_like")
    if slot_looks_like_meta_headword(slot):
        replaceable = False
        reasons.append("meta_headword_like")

    pool_preview: list[dict[str, Any]] = []
    pool_total = 0
    if replaceable:
        pool_preview, pool_total = rank_dense_slot_candidates(slot, scene_id, semantic_index)
        if pool_total < 4:
            replaceable = False
            reasons.append("candidate_pool_lt_4")

    return {
        "dense_replaceable_rule": replaceable,
        "dense_lock_reasons": reasons,
        "dense_priority": dense_slot_priority(slot),
        "dense_candidate_pool_total": pool_total,
        "dense_candidate_pool_preview": pool_preview,
    }


def estimate_dense_combination_upper_bound(slots: list[dict[str, Any]]) -> int:
    upper_bound = 1
    replaceable_count = 0
    for slot in slots:
        if not slot.get("dense_replaceable_rule"):
            continue
        replaceable_count += 1
        pool_total = int(slot.get("dense_candidate_pool_total") or 0)
        upper_bound *= max(min(pool_total, 8), 1)
    if replaceable_count == 0:
        return 0
    return upper_bound


def dense_reuse_rule_score(row: dict[str, Any]) -> tuple[float, list[str], str]:
    slots = row.get("slots") or []
    replaceable_slots = [slot for slot in slots if slot.get("dense_replaceable_rule")]
    replaceable_count = len(replaceable_slots)
    unknown_count = sum(1 for slot in slots if slot.get("semantic_class", "unknown") == "unknown")
    pronoun_locked_count = sum(1 for slot in slots if "pronoun_like" in (slot.get("dense_lock_reasons") or []))
    all_content_slots_replaceable = replaceable_count == len(slots) and replaceable_count > 0
    combination_upper_bound = estimate_dense_combination_upper_bound(slots)
    min_pool_total = min(
        (int(slot.get("dense_candidate_pool_total") or 0) for slot in replaceable_slots),
        default=0,
    )

    score = float(row.get("skeleton_quality_score", 0.0))
    reasons: list[str] = []
    if row.get("source_type") == "dictionary_example":
        score += 0.04
    if replaceable_count >= 2:
        score += 0.1
        reasons.append("replaceable_slots_gte_2")
    if replaceable_count >= 3:
        score += 0.06
        reasons.append("replaceable_slots_gte_3")
    if all_content_slots_replaceable:
        score += 0.08
        reasons.append("all_slots_replaceable")
    if min_pool_total >= 6:
        score += 0.06
        reasons.append("pool_min_gte_6")
    elif min_pool_total >= 4:
        score += 0.03
        reasons.append("pool_min_gte_4")
    if combination_upper_bound >= 100:
        score += 0.1
        reasons.append("combination_bound_gte_100")
    elif combination_upper_bound >= 30:
        score += 0.04
        reasons.append("combination_bound_gte_30")
    if row.get("primary_scene_confidence", 0.0) >= 0.6:
        score += 0.02
    if row.get("primary_speech_function") in {"statement", "reminder", "planning", "evaluation"}:
        score += 0.02
    if replaceable_count < 2:
        score -= 0.18
        reasons.append("replaceable_slots_lt_2")
    if unknown_count > 0:
        score -= 0.08 * unknown_count
        reasons.append("unknown_slots_present")
    if pronoun_locked_count > 0:
        score -= 0.05 * pronoun_locked_count
        reasons.append("pronoun_like_slots_present")

    score = round(min(max(score, 0.0), 0.99), 4)
    if replaceable_count >= 2 and min_pool_total >= 4 and combination_upper_bound >= 100 and score >= 0.82:
        tier = "dense_reusable"
    elif replaceable_count >= 2 and min_pool_total >= 3 and combination_upper_bound >= 12 and score >= 0.72:
        tier = "medium_reusable"
    else:
        tier = "single_use"
    return score, reasons, tier


def annotate_dense_reusable_skeletons(
    skeleton_rows: list[dict[str, Any]],
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    semantic_index = build_dense_semantic_index(entries)
    annotated_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for row in skeleton_rows:
        merged = dict(row)
        annotated_slots: list[dict[str, Any]] = []
        for slot in row.get("slots") or []:
            annotated_slot = dict(slot)
            annotated_slot.update(annotate_dense_slot(slot, row["scene_id"], semantic_index))
            annotated_slots.append(annotated_slot)
        merged["slots"] = annotated_slots
        merged["dense_eligible_slot_ids"] = [
            slot["slot_id"] for slot in annotated_slots if slot.get("dense_replaceable_rule")
        ]
        merged["dense_locked_slot_ids"] = [
            slot["slot_id"] for slot in annotated_slots if not slot.get("dense_replaceable_rule")
        ]
        merged["dense_combination_upper_bound"] = estimate_dense_combination_upper_bound(annotated_slots)
        score, reasons, tier = dense_reuse_rule_score(merged)
        merged["dense_reuse_rule_score"] = score
        merged["dense_reuse_rule_reasons"] = reasons
        merged["dense_reuse_rule_tier"] = tier
        merged["dense_reuse_rule_candidate"] = tier in {"dense_reusable", "medium_reusable"}
        annotated_rows.append(merged)
        if merged["dense_reuse_rule_candidate"]:
            candidate_rows.append(merged)

    annotated_rows.sort(
        key=lambda row: (
            {"dense_reusable": 2, "medium_reusable": 1, "single_use": 0}[row["dense_reuse_rule_tier"]],
            row["dense_reuse_rule_score"],
            row["skeleton_quality_score"],
            row["skeleton_id"],
        ),
        reverse=True,
    )
    candidate_rows.sort(key=lambda row: (row["dense_reuse_rule_score"], row["skeleton_quality_score"], row["skeleton_id"]), reverse=True)

    summary = {
        "annotated_skeleton_count": len(annotated_rows),
        "dense_rule_candidate_count": len(candidate_rows),
        "dense_rule_tier_counts": dict(Counter(row["dense_reuse_rule_tier"] for row in annotated_rows)),
        "dense_rule_scene_counts": dict(Counter(row["scene_id"] for row in candidate_rows)),
        "replaceable_slot_count_distribution": dict(
            Counter(len(row["dense_eligible_slot_ids"]) for row in annotated_rows)
        ),
        "combination_upper_bound_distribution": {
            "gte_100": sum(1 for row in annotated_rows if row["dense_combination_upper_bound"] >= 100),
            "gte_30": sum(1 for row in annotated_rows if row["dense_combination_upper_bound"] >= 30),
            "gte_12": sum(1 for row in annotated_rows if row["dense_combination_upper_bound"] >= 12),
        },
    }
    return annotated_rows, candidate_rows, summary


def usable_as_slot(wz_word: str, headword: str | None, definition: str | None, slot_kind: str, confidence: float, fixed: bool) -> bool:
    if fixed:
        return False
    if not wz_word or not headword:
        return False
    if slot_kind == "function":
        return False
    if wz_word in BAD_WZ_WORDS:
        return False
    if headword in BAD_HEADWORDS:
        return False
    if slot_kind == "unknown" and len(wz_word) <= 1:
        return False
    if len(wz_word) == 1 and confidence < 0.85:
        return False
    if len(headword) == 1 and confidence < 0.9:
        return False
    if definition and any(definition.startswith(prefix) for prefix in FUNCTION_DEFINITION_PREFIXES):
        return False
    return confidence >= MIN_LEXICON_CONFIDENCE


def build_replaceable_lexicon(records: list[dict[str, Any]], extraction_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record in records:
        if not record.get("is_exact_primary"):
            continue
        if not record.get("v1_candidate"):
            continue
        if record.get("record_type") not in {"dictionary_entry", "lexicon_entry", "fixed_expression"}:
            continue
        wz_word = record.get("wz_word_train")
        if not wz_word:
            continue
        extraction_row = extraction_map.get(record["dedup_exact_key"])
        extraction = extraction_row.get("extraction", {}) if extraction_row else {}
        headword = extraction.get("mandarin_headword")
        definition = extraction.get("definition_clean") or record.get("definition_norm")
        confidence = float(extraction.get("confidence") or 0.0)
        fixed = bool(extraction.get("is_fixed_expression"))
        slot_kind = guess_slot_kind(headword, definition, wz_word)
        usable = usable_as_slot(wz_word, headword, definition, slot_kind, confidence, fixed)
        scene_payload = build_scene_payload(
            headword or "",
            definition or "",
            record.get("example_zh_train") or "",
            record.get("example_wz_train") or "",
        )
        semantic_class = guess_semantic_class(
            headword,
            definition,
            wz_word,
            slot_kind,
            primary_scene_id=scene_payload["primary_scene_id"],
        )
        speech_payload = build_speech_function_payload(
            record.get("example_zh_train") or "",
            record.get("example_wz_train") or "",
            definition or "",
        )
        if record["dedup_exact_key"] in seen_ids:
            continue
        seen_ids.add(record["dedup_exact_key"])
        entries.append(
            {
                "lexicon_id": record["dedup_exact_key"],
                "wz_word": wz_word,
                "mandarin_headword": headword,
                "definition": definition,
                "slot_kind": slot_kind,
                "semantic_class": semantic_class,
                "confidence": confidence,
                "source_file": record["source_file"],
                "source_row_id": record["source_row_id"],
                "record_type": record["record_type"],
                "usable_as_slot": usable,
                "has_example": bool(record.get("example_wz_train")),
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
        )
    entries.sort(key=lambda row: (-int(row["usable_as_slot"]), -len(row["wz_word"]), -row["confidence"], row["wz_word"]))
    return entries


def build_lexicon_index(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        if not entry["usable_as_slot"]:
            continue
        first = entry["wz_word"][0]
        index.setdefault(first, []).append(entry)
    for first, bucket in index.items():
        bucket.sort(key=lambda row: (-len(row["wz_word"]), -row["confidence"], row["wz_word"]))
    return index


def merge_example_mined_lexicon(entries: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return entries
    existing_ids = {row["lexicon_id"] for row in entries if "lexicon_id" in row}
    for row in load_jsonl(path):
        lexicon_id = f"example_mined::{row['term']}"
        if lexicon_id in existing_ids:
            continue
        entries.append(
            {
                "lexicon_id": lexicon_id,
                "wz_word": row["wz_word"],
                "mandarin_headword": row["mandarin_headword"],
                "definition": f"来自例句挖掘的高频生活词:{row['term']}",
                "slot_kind": row.get("slot_kind", "noun"),
                "semantic_class": row.get("semantic_class")
                or guess_semantic_class(
                    row.get("mandarin_headword"),
                    row.get("definition"),
                    row.get("wz_word"),
                    row.get("slot_kind", "noun"),
                    primary_scene_id=row.get("primary_topic_scene"),
                ),
                "confidence": row.get("confidence", 0.7),
                "source_file": "example_mined_lexicon",
                "source_row_id": None,
                "record_type": "example_mined",
                "usable_as_slot": row.get("usable_as_slot", False),
                "has_example": bool(row.get("evidence_examples")),
                "scene_tags": row.get("topic_scene_tags") or [],
                "scene_scores": row.get("topic_scene_scores") or {},
                "primary_scene_id": row.get("primary_topic_scene"),
                "primary_scene_confidence": row.get("topic_scene_confidence", 0.0),
                "topic_scene_tags": row.get("topic_scene_tags") or [],
                "topic_scene_scores": row.get("topic_scene_scores") or {},
                "primary_topic_scene": row.get("primary_topic_scene"),
                "topic_scene_confidence": row.get("topic_scene_confidence", 0.0),
                "speech_function_tags": row.get("speech_function_tags") or ["statement"],
                "speech_function_scores": row.get("speech_function_scores") or {"statement": 0},
                "primary_speech_function": row.get("primary_speech_function", "statement"),
                "speech_function_confidence": row.get("speech_function_confidence", 0.0),
            }
        )
    entries.sort(key=lambda row: (-int(row["usable_as_slot"]), -len(row["wz_word"]), -row["confidence"], row["wz_word"]))
    return entries


def collect_sentence_sources(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if not record.get("v1_candidate"):
            continue
        if record["record_type"] == "sentence_pair":
            wz = record.get("wz_text_train")
            zh = record.get("zh_text_train")
            if not wz or not zh:
                continue
            rows.append(
                {
                    "source_record_id": record["dedup_exact_key"],
                    "source_type": "sentence_pair",
                    "source_file": record["source_file"],
                    "source_row_id": record["source_row_id"],
                    "source_wz_sentence": wz,
                    "source_zh_sentence": zh,
                    "anchor_wz_word": None,
                }
            )
        elif record["record_type"] == "dictionary_entry":
            wz = record.get("example_wz_train")
            zh = record.get("example_zh_train")
            if not wz or not zh:
                continue
            rows.append(
                {
                    "source_record_id": record["dedup_exact_key"],
                    "source_type": "dictionary_example",
                    "source_file": record["source_file"],
                    "source_row_id": record["source_row_id"],
                    "source_wz_sentence": wz,
                    "source_zh_sentence": zh,
                    "anchor_wz_word": record.get("wz_word_train"),
                }
            )
    return rows


def match_sentence_slots(
    sentence: str,
    lexicon_index: dict[str, list[dict[str, Any]]],
    anchor_wz_word: str | None,
    primary_scene_id: str,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    anchor_done = False
    pos = 0
    while pos < len(sentence):
        picked = None
        for entry in lexicon_index.get(sentence[pos], []):
            wz_word = entry["wz_word"]
            if not sentence.startswith(wz_word, pos):
                continue
            if not scene_compatible(entry.get("scene_tags") or [], primary_scene_id):
                continue
            end = pos + len(wz_word)
            if any(not (end <= s or pos >= e) for s, e in occupied):
                continue
            picked = entry
            break
        if picked is None:
            pos += 1
            continue
        end = pos + len(picked["wz_word"])
        is_anchor = bool(anchor_wz_word and not anchor_done and picked["wz_word"] == anchor_wz_word)
        if is_anchor:
            anchor_done = True
        matches.append(
            {
                "lexicon_id": picked["lexicon_id"],
                "wz_word": picked["wz_word"],
                "surface_zh": picked["mandarin_headword"],
                "slot_kind": picked["slot_kind"],
                "semantic_class": picked.get("semantic_class", "unknown"),
                "confidence": picked["confidence"],
                "span_start": pos,
                "span_end": end,
                "is_anchor": is_anchor,
                "score": len(picked["wz_word"]) * 2 + picked["confidence"] + (1.0 if is_anchor else 0.0),
            }
        )
        occupied.append((pos, end))
        pos = end
    return matches


def choose_slots(matches: list[dict[str, Any]], sentence: str, max_slots: int) -> list[dict[str, Any]]:
    if not matches:
        return []
    ranked = sorted(matches, key=lambda row: (-row["score"], row["span_start"]))
    chosen: list[dict[str, Any]] = []
    for match in ranked:
        if len(chosen) >= max_slots:
            break
        if any(not (match["span_end"] <= item["span_start"] or match["span_start"] >= item["span_end"]) for item in chosen):
            continue
        if any(match["span_start"] == item["span_end"] or match["span_end"] == item["span_start"] for item in chosen):
            continue
        chosen.append(match)
    chosen.sort(key=lambda row: row["span_start"])
    slot_chars = sum(item["span_end"] - item["span_start"] for item in chosen)
    if not chosen:
        return []
    if slot_chars / max(len(sentence), 1) > 0.55:
        return []
    return chosen


def build_skeleton_template(sentence: str, chosen_slots: list[dict[str, Any]]) -> tuple[str, list[str], list[dict[str, Any]]]:
    pieces: list[str] = []
    fixed_chunks: list[str] = []
    slots: list[dict[str, Any]] = []
    cursor = 0
    for index, match in enumerate(chosen_slots, start=1):
        slot_id = f"SLOT_{index}"
        fixed = sentence[cursor : match["span_start"]]
        pieces.append(fixed)
        fixed_chunks.append(fixed)
        pieces.append(f"[{slot_id}]")
        slots.append(
            {
                "slot_id": slot_id,
                "surface_wz": match["wz_word"],
                "surface_zh": match["surface_zh"],
                "slot_kind": match["slot_kind"],
                "semantic_class": match.get("semantic_class", "unknown"),
                "span_start": match["span_start"],
                "span_end": match["span_end"],
                "lexicon_id": match["lexicon_id"],
                "confidence": match["confidence"],
                "is_anchor": match["is_anchor"],
            }
        )
        cursor = match["span_end"]
    tail = sentence[cursor:]
    pieces.append(tail)
    fixed_chunks.append(tail)
    return "".join(pieces), fixed_chunks, slots


def skeleton_quality_score(source_type: str, sentence: str, zh_sentence: str, slots: list[dict[str, Any]], scene_score: int) -> float:
    slot_chars = sum(slot["span_end"] - slot["span_start"] for slot in slots)
    coverage = slot_chars / max(len(sentence), 1)
    anchor_bonus = 0.08 if any(slot["is_anchor"] for slot in slots) else 0.0
    source_bonus = 0.06 if source_type == "dictionary_example" else 0.0
    dialect_bonus = 0.05 if normalize_compact(sentence) != normalize_compact(zh_sentence) else 0.0
    scene_bonus = min(scene_score, 3) * 0.02
    base = 0.55 + source_bonus + anchor_bonus + dialect_bonus + scene_bonus
    if 0.1 <= coverage <= 0.45:
        base += 0.18
    elif coverage <= 0.55:
        base += 0.08
    if len(slots) == 1:
        base += 0.05
    elif len(slots) == 2:
        base += 0.03
    return round(min(base, 0.99), 4)


def build_skeleton_rows(sources: list[dict[str, Any]], lexicon_index: dict[str, list[dict[str, Any]]], max_slots: int, min_quality: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        sentence = source["source_wz_sentence"]
        if len(sentence) < 6 or len(sentence) > 40:
            continue
        scene = build_scene_payload(source["source_zh_sentence"], sentence)
        speech = build_speech_function_payload(source["source_zh_sentence"], sentence)
        matches = match_sentence_slots(sentence, lexicon_index, source["anchor_wz_word"], scene["primary_scene_id"])
        chosen = choose_slots(matches, sentence, max_slots=max_slots)
        if not chosen:
            continue
        template, fixed_chunks, slots = build_skeleton_template(sentence, chosen)
        quality = skeleton_quality_score(
            source["source_type"],
            sentence,
            source["source_zh_sentence"],
            slots,
            scene["scene_scores"].get(scene["primary_scene_id"], 0),
        )
        if quality < min_quality:
            continue
        skeleton_id = stable_id("sk", source["source_record_id"], template)
        rows.append(
            {
                "skeleton_id": skeleton_id,
                "source_record_id": source["source_record_id"],
                "source_type": source["source_type"],
                "source_file": source["source_file"],
                "source_row_id": source["source_row_id"],
                "source_wz_sentence": sentence,
                "source_zh_sentence": source["source_zh_sentence"],
                "scene_id": scene["primary_scene_id"],
                "scene_label": scene["primary_scene_label"],
                "scene_tags": scene["scene_tags"],
                "scene_scores": scene["scene_scores"],
                "primary_scene_confidence": scene["primary_scene_confidence"],
                "scene_candidates": scene["scene_candidates"],
                "topic_scene_tags": scene["scene_tags"],
                "topic_scene_scores": scene["scene_scores"],
                "primary_topic_scene": scene["primary_scene_id"],
                "topic_scene_confidence": scene["primary_scene_confidence"],
                "speech_function_tags": speech["speech_function_tags"],
                "speech_function_scores": speech["speech_function_scores"],
                "primary_speech_function": speech["primary_speech_function"],
                "speech_function_confidence": speech["speech_function_confidence"],
                "skeleton_template": template,
                "fixed_chunks": fixed_chunks,
                "slots": slots,
                "slot_count": len(slots),
                "skeleton_quality_score": quality,
            }
        )
    rows.sort(key=lambda row: (-row["skeleton_quality_score"], row["skeleton_id"]))
    return rows


def choose_external_term(
    term_rows: list[dict[str, Any]],
    task_id: str,
    tag: str,
    scene_id: str,
    strict_primary: bool = False,
) -> str | None:
    if not term_rows:
        return None
    exact_primary = [
        row
        for row in term_rows
        if row.get("primary_scene_id") == scene_id or row.get("primary_topic_scene") == scene_id
    ]
    if exact_primary:
        scoped = exact_primary
    elif strict_primary:
        return None
    else:
        scoped = [row for row in term_rows if scene_compatible(row.get("scene_tags") or [], scene_id)]
    if not scoped:
        return None
    index = stable_bucket(task_id, tag, mod=len(scoped))
    return scoped[index]["term"]


def choose_modern_words(
    scene_id: str,
    scene_map: dict[str, dict[str, Any]],
    source_sentence: str,
    task_id: str,
    slots: list[dict[str, Any]],
    example_mined_terms: list[dict[str, Any]],
    new_nouns: list[dict[str, Any]],
    place_names: list[dict[str, Any]],
    new_noun_ratio: float,
    place_ratio: float,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    scene = scene_map[scene_id]
    allow_injection = slots_allow_semantic_injection(slots)
    default_required = (
        filter_words_by_slot_semantics(list(scene["default_required_modern_words"]), scene_id, slots)
        if allow_injection
        else []
    )
    default_preferred = (
        filter_words_by_slot_semantics(list(scene["default_preferred_modern_words"]), scene_id, slots)
        if allow_injection
        else []
    )
    external_required: list[str] = []
    external_preferred: list[str] = []
    injections: list[dict[str, Any]] = []
    default_required = [word for word in default_required if word not in source_sentence][:1]
    default_preferred = [word for word in default_preferred if word not in source_sentence][:2]
    if not default_required and allow_injection:
        fallback_default = filter_words_by_slot_semantics(["手机"], scene_id, slots)
        default_required = [word for word in fallback_default if word not in source_sentence][:1]

    scoped_example_mined = filter_terms_by_slot_semantics(example_mined_terms, slots) if allow_injection else []
    if scoped_example_mined and stable_bucket(task_id, "example_mined_ratio") < 18:
        term = choose_external_term(
            scoped_example_mined,
            task_id,
            "example_mined_pick",
            scene_id,
            strict_primary=True,
        )
        if term and not external_required and term not in source_sentence and term not in default_required and term not in default_preferred:
            external_required.append(term)
            injections.append({"type": "example_mined", "term": term, "level": "required"})

    scoped_new_nouns = filter_terms_by_slot_semantics(new_nouns, slots) if allow_injection else []
    if scoped_new_nouns and stable_bucket(task_id, "new_noun_ratio") < int(new_noun_ratio * 100):
        term = choose_external_term(scoped_new_nouns, task_id, "new_noun_pick", scene_id)
        if term and term not in source_sentence and term not in default_required and term not in default_preferred:
            if not external_required:
                external_required.append(term)
                level = "required"
            else:
                external_preferred.append(term)
                level = "preferred"
            injections.append({"type": "new_noun", "term": term, "level": level})

    scoped_place_names = filter_terms_by_slot_semantics(place_names, slots) if allow_injection else []
    if scoped_place_names and stable_bucket(task_id, "place_ratio") < int(place_ratio * 100):
        term = choose_external_term(scoped_place_names, task_id, "place_pick", scene_id)
        if term and term not in source_sentence and term not in default_required and term not in default_preferred and term not in external_required:
            if scene_id == "transport_trip" and not external_required:
                external_required.append(term)
                level = "required"
            else:
                external_preferred.append(term)
                level = "preferred"
            injections.append({"type": "place_name", "term": term, "level": level})

    required = (external_required + default_required)[:2]
    preferred = [word for word in (external_preferred + default_preferred) if word not in required][:2]
    return required, preferred, injections


def build_user_prompt(task_payload: dict[str, Any]) -> str:
    return "根据下面 JSON 生成候选句，返回 JSON：\n" + json.dumps(task_payload, ensure_ascii=False, indent=2)


def build_generation_tasks(
    skeleton_rows: list[dict[str, Any]],
    scene_map: dict[str, dict[str, Any]],
    example_mined_terms: list[dict[str, Any]],
    new_nouns: list[dict[str, Any]],
    place_names: list[dict[str, Any]],
    new_noun_ratio: float,
    place_ratio: float,
    generation_count: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for skeleton in skeleton_rows:
        task_id = stable_id("gen", skeleton["skeleton_id"], skeleton["scene_id"])
        min_slot_changes = 2 if skeleton["slot_count"] >= 2 else 1
        required_words, preferred_words, injections = choose_modern_words(
            skeleton["scene_id"],
            scene_map,
            skeleton["source_wz_sentence"],
            task_id,
            skeleton["slots"],
            example_mined_terms,
            new_nouns,
            place_names,
            new_noun_ratio,
            place_ratio,
        )
        slot_payload = [
            {
                "slot_id": slot["slot_id"],
                "surface_wz": slot["surface_wz"],
                "surface_zh": slot["surface_zh"],
                "slot_kind": slot["slot_kind"],
                "semantic_class": slot.get("semantic_class", "unknown"),
            }
            for slot in skeleton["slots"]
        ]
        payload = {
            "scene_label": skeleton["scene_label"],
            "topic_scene_id": skeleton["scene_id"],
            "topic_scene_label": skeleton["scene_label"],
            "speech_function": skeleton["primary_speech_function"],
            "source_wz_sentence": skeleton["source_wz_sentence"],
            "skeleton_template": skeleton["skeleton_template"],
            "slots": slot_payload,
            "slot_role_constraints": [
                {"slot_id": slot["slot_id"], "semantic_class": slot.get("semantic_class", "unknown")}
                for slot in skeleton["slots"]
            ],
            "slot_replacement_policy": {
                "slot_count": skeleton["slot_count"],
                "target_slot_changes": min_slot_changes,
                "replace_all_slots": skeleton["slot_count"] >= 2,
            },
            "required_words": required_words,
            "preferred_words": preferred_words,
            "external_constraints": injections,
            "generation_count": generation_count,
        }
        tasks.append(
            {
                "task_id": task_id,
                "task_type": "scene_conditioned_wz_generation",
                "skeleton_id": skeleton["skeleton_id"],
                "scene_id": skeleton["scene_id"],
                "scene_label": skeleton["scene_label"],
                "topic_scene_id": skeleton["scene_id"],
                "topic_scene_label": skeleton["scene_label"],
                "topic_scene_tags": skeleton.get("topic_scene_tags") or skeleton.get("scene_tags") or [],
                "primary_speech_function": skeleton["primary_speech_function"],
                "speech_function_tags": skeleton["speech_function_tags"],
                "source_wz_sentence": skeleton["source_wz_sentence"],
                "source_zh_sentence": skeleton["source_zh_sentence"],
                "skeleton_template": skeleton["skeleton_template"],
                "fixed_chunks": skeleton["fixed_chunks"],
                "slots": skeleton["slots"],
                "required_words": required_words,
                "preferred_words": preferred_words,
                "external_constraints": injections,
                "slot_role_constraints": payload["slot_role_constraints"],
                "slot_replacement_policy": payload["slot_replacement_policy"],
                "generation_count": generation_count,
                "quality_gate": {
                    "min_required_hits": len(required_words),
                    "min_slot_changes": min_slot_changes,
                    "preserve_fixed_chunks": True,
                },
                "prompt_bundle": {
                    "system_prompt": SYSTEM_PROMPT,
                    "user_payload": payload,
                    "user_prompt": build_user_prompt(payload),
                },
                "llm_generated": False,
            }
        )
    return tasks


def build_modern_word_catalog(scene_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in scene_map.values():
        for word in scene["default_required_modern_words"]:
            rows.append({"scene_id": scene["scene_id"], "word": word, "level": "required"})
        for word in scene["default_preferred_modern_words"]:
            rows.append({"scene_id": scene["scene_id"], "word": word, "level": "preferred"})
    rows.sort(key=lambda row: (row["scene_id"], row["level"], row["word"]))
    return rows


def build_short_term_catalog(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        term = entry.get("wz_word") or ""
        if not 2 <= len(term) <= 4:
            continue
        rows.append(
            {
                "term": term,
                "lexicon_id": entry.get("lexicon_id"),
                "source_file": entry.get("source_file"),
                "record_type": entry.get("record_type"),
                "mandarin_headword": entry.get("mandarin_headword"),
                "definition": entry.get("definition"),
                "usable_as_slot": bool(entry.get("usable_as_slot")),
                "has_example": bool(entry.get("has_example")),
                "slot_kind": entry.get("slot_kind"),
                "semantic_class": entry.get("semantic_class"),
                "confidence": entry.get("confidence", 0.0),
                "primary_scene_id": entry.get("primary_scene_id"),
                "scene_tags": entry.get("scene_tags") or [],
            }
        )
    rows.sort(key=lambda row: (-int(row["usable_as_slot"]), -int(row["has_example"]), row["term"], row["lexicon_id"] or ""))
    return rows


def build_short_term_coverage_report(
    short_term_rows: list[dict[str, Any]],
    skeleton_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    catalog_terms = {row["term"] for row in short_term_rows}
    skeleton_terms = Counter()
    skeleton_anchor_terms = Counter()
    for row in skeleton_rows:
        for slot in row.get("slots") or []:
            term = slot.get("surface_wz")
            if term in catalog_terms:
                skeleton_terms[term] += 1
                if slot.get("is_anchor"):
                    skeleton_anchor_terms[term] += 1

    required_terms = Counter()
    preferred_terms = Counter()
    for row in task_rows:
        for term in row.get("required_words") or []:
            if term in catalog_terms:
                required_terms[term] += 1
        for term in row.get("preferred_words") or []:
            if term in catalog_terms:
                preferred_terms[term] += 1

    covered_terms = {term for term, count in skeleton_terms.items() if count > 0}
    return {
        "short_term_total": len(short_term_rows),
        "short_term_usable_as_slot": sum(1 for row in short_term_rows if row["usable_as_slot"]),
        "short_term_with_examples": sum(1 for row in short_term_rows if row["has_example"]),
        "covered_in_skeleton_slots": len(covered_terms),
        "unused_in_skeleton_slots": len(catalog_terms - covered_terms),
        "source_counts": dict(Counter(row["source_file"] for row in short_term_rows)),
        "record_type_counts": dict(Counter(row["record_type"] for row in short_term_rows)),
        "slot_kind_counts": dict(Counter(row["slot_kind"] for row in short_term_rows)),
        "semantic_class_counts": dict(Counter(row["semantic_class"] for row in short_term_rows)),
        "top_skeleton_terms": dict(skeleton_terms.most_common(50)),
        "top_anchor_terms": dict(skeleton_anchor_terms.most_common(50)),
        "top_required_terms": dict(required_terms.most_common(50)),
        "top_preferred_terms": dict(preferred_terms.most_common(50)),
    }


def build_external_term_rows(new_nouns: list[str], place_names: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    noun_rows: list[dict[str, Any]] = []
    for term in new_nouns:
        scene_payload = build_scene_payload(term)
        speech_payload = build_speech_function_payload(term)
        noun_rows.append(
            {
                "term": term,
                "term_type": "new_noun",
                "semantic_class": guess_semantic_class(term, term, term, "noun", primary_scene_id=scene_payload["primary_scene_id"]),
                "scene_tags": scene_payload["scene_tags"],
                "scene_scores": scene_payload["scene_scores"],
                "primary_scene_id": scene_payload["primary_scene_id"],
                "topic_scene_tags": scene_payload["scene_tags"],
                "topic_scene_scores": scene_payload["scene_scores"],
                "primary_topic_scene": scene_payload["primary_scene_id"],
                "speech_function_tags": speech_payload["speech_function_tags"],
                "primary_speech_function": speech_payload["primary_speech_function"],
            }
        )
    place_rows: list[dict[str, Any]] = []
    for term in place_names:
        place_rows.append(
            {
                "term": term,
                "term_type": "place_name",
                "semantic_class": "place",
                "scene_tags": ["transport_trip", "daily_chat", "home_life"],
                "scene_scores": {"transport_trip": 2, "daily_chat": 1, "home_life": 1},
                "primary_scene_id": "transport_trip",
                "topic_scene_tags": ["transport_trip", "daily_chat", "home_life"],
                "topic_scene_scores": {"transport_trip": 2, "daily_chat": 1, "home_life": 1},
                "primary_topic_scene": "transport_trip",
                "speech_function_tags": ["statement"],
                "primary_speech_function": "statement",
            }
        )
    return noun_rows, place_rows


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build controlled-generation assets, skeleton templates, and generation tasks.")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--heuristic", type=Path, default=DEFAULT_HEURISTIC)
    parser.add_argument("--api", type=Path, default=DEFAULT_API)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--new-nouns", type=Path, default=DEFAULT_NEW_NOUNS)
    parser.add_argument("--place-names", type=Path, default=DEFAULT_PLACE_NAMES)
    parser.add_argument("--example-mined-lexicon", type=Path, default=DEFAULT_EXAMPLE_MINED)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--max-slots", type=int, default=4)
    parser.add_argument("--generation-count", type=int, default=4)
    parser.add_argument("--min-skeleton-quality", type=float, default=0.72)
    parser.add_argument("--new-noun-ratio", type=float, default=0.18)
    parser.add_argument("--place-ratio", type=float, default=0.12)
    parser.add_argument("--example-mined-scene-threshold", type=float, default=0.5)
    args = parser.parse_args()

    records = load_jsonl(args.records)
    extraction_map = load_extraction_map(args.heuristic, args.api)
    scene_map = {scene["scene_id"]: scene for scene in SCENE_CATALOG}
    new_noun_terms = load_single_column_excel(args.new_nouns)
    place_name_terms = load_single_column_excel(args.place_names)
    new_nouns, place_names = build_external_term_rows(new_noun_terms, place_name_terms)

    lexicon_rows = build_replaceable_lexicon(records, extraction_map)
    lexicon_rows = merge_example_mined_lexicon(lexicon_rows, args.example_mined_lexicon)
    lexicon_index = build_lexicon_index(lexicon_rows)
    example_mined_terms = [
        {
            "term": row["wz_word"],
            "scene_tags": row.get("topic_scene_tags") or [],
            "primary_topic_scene": row.get("primary_topic_scene"),
            "topic_scene_confidence": row.get("topic_scene_confidence", 0.0),
            "semantic_class": row.get("semantic_class", "unknown"),
        }
        for row in lexicon_rows
        if (
            row.get("record_type") == "example_mined"
            and row.get("usable_as_slot")
            and row.get("topic_scene_confidence", 0.0) >= args.example_mined_scene_threshold
        )
    ]
    source_rows = collect_sentence_sources(records)
    skeleton_rows = build_skeleton_rows(
        source_rows,
        lexicon_index,
        max_slots=args.max_slots,
        min_quality=args.min_skeleton_quality,
    )
    skeleton_rows, dense_rule_candidates, dense_rule_summary = annotate_dense_reusable_skeletons(
        skeleton_rows,
        lexicon_rows,
    )
    task_rows = build_generation_tasks(
        skeleton_rows,
        scene_map,
        example_mined_terms,
        new_nouns,
        place_names,
        new_noun_ratio=args.new_noun_ratio,
        place_ratio=args.place_ratio,
        generation_count=args.generation_count,
    )
    sample_tasks = task_rows[: args.sample_size]
    modern_catalog = build_modern_word_catalog(scene_map)
    short_term_catalog = build_short_term_catalog(lexicon_rows)
    short_term_coverage = build_short_term_coverage_report(short_term_catalog, skeleton_rows, task_rows)

    output_dir: Path = args.output_dir
    assets_dir = output_dir / "assets"
    skeleton_dir = output_dir / "skeletons"
    tasks_dir = output_dir / "tasks"
    reports_dir = output_dir / "reports"
    assets_dir.mkdir(parents=True, exist_ok=True)
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(assets_dir / "replaceable_lexicon.jsonl", lexicon_rows)
    write_summary(assets_dir / "scene_catalog.json", {"scenes": SCENE_CATALOG})
    write_summary(assets_dir / "modern_word_catalog.json", {"rows": modern_catalog})
    write_jsonl(assets_dir / "short_term_catalog.jsonl", short_term_catalog)
    write_summary(
        assets_dir / "external_term_catalog.json",
        {
            "example_mined_terms": example_mined_terms,
            "example_mined_scene_threshold": args.example_mined_scene_threshold,
            "new_nouns": new_nouns,
            "place_names": place_names,
            "new_noun_ratio": args.new_noun_ratio,
            "place_ratio": args.place_ratio,
        },
    )
    write_summary(assets_dir / "scene_taxonomy.json", {"scenes": SCENE_CATALOG})
    write_summary(assets_dir / "speech_function_taxonomy.json", {"functions": SPEECH_FUNCTION_CATALOG})
    write_jsonl(skeleton_dir / "skeleton_templates.jsonl", skeleton_rows)
    write_jsonl(skeleton_dir / "dense_reusable_rule_candidates.jsonl", dense_rule_candidates)
    write_jsonl(tasks_dir / "generation_tasks_full.jsonl", task_rows)
    write_jsonl(tasks_dir / "generation_tasks_sample.jsonl", sample_tasks)

    skeleton_summary = {
        "source_sentence_count": len(source_rows),
        "replaceable_lexicon_count": len(lexicon_rows),
        "replaceable_slot_lexicon_count": sum(1 for row in lexicon_rows if row["usable_as_slot"]),
        "example_mined_slot_terms": sum(
            1 for row in lexicon_rows if row.get("record_type") == "example_mined" and row["usable_as_slot"]
        ),
        "example_mined_terms_used_for_generation": len(example_mined_terms),
        "skeleton_template_count": len(skeleton_rows),
        "scene_counts": dict(Counter(row["scene_id"] for row in skeleton_rows)),
        "slot_count_distribution": dict(Counter(row["slot_count"] for row in skeleton_rows)),
        "dense_rule_candidate_count": len(dense_rule_candidates),
        "dense_rule_tier_counts": dense_rule_summary["dense_rule_tier_counts"],
    }
    task_summary = {
        "generation_task_count": len(task_rows),
        "sample_task_count": len(sample_tasks),
        "scene_counts": dict(Counter(row["scene_id"] for row in task_rows)),
        "required_word_counts": dict(Counter(len(row["required_words"]) for row in task_rows)),
        "speech_function_counts": dict(Counter(row["primary_speech_function"] for row in task_rows)),
        "tasks_with_example_mined_terms": sum(
            1 for row in task_rows if any(item["type"] == "example_mined" for item in row["external_constraints"])
        ),
        "tasks_with_new_nouns": sum(
            1 for row in task_rows if any(item["type"] == "new_noun" for item in row["external_constraints"])
        ),
        "tasks_with_place_names": sum(
            1 for row in task_rows if any(item["type"] == "place_name" for item in row["external_constraints"])
        ),
    }
    write_summary(reports_dir / "skeleton_summary.json", skeleton_summary)
    write_summary(reports_dir / "task_summary.json", task_summary)
    write_summary(reports_dir / "short_term_coverage_report.json", short_term_coverage)
    write_summary(reports_dir / "dense_reusable_rule_summary.json", dense_rule_summary)

    print(f"Loaded cleaned records: {len(records)}")
    print(f"Replaceable lexicon rows: {len(lexicon_rows)}")
    print(f"Usable slot lexicon rows: {sum(1 for row in lexicon_rows if row['usable_as_slot'])}")
    print(f"Example-mined slot terms: {sum(1 for row in lexicon_rows if row.get('record_type') == 'example_mined' and row['usable_as_slot'])}")
    print(f"Sentence sources: {len(source_rows)}")
    print(f"Skeleton templates: {len(skeleton_rows)}")
    print(f"Dense rule candidates: {len(dense_rule_candidates)}")
    print(f"Generation tasks: {len(task_rows)}")
    print(f"Sample tasks: {len(sample_tasks)}")
    print(f"External new nouns: {len(new_nouns)}")
    print(f"External place names: {len(place_names)}")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
