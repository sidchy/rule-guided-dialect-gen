#!/usr/bin/env python3
"""
轨道 A · LLM 自动质检流程：

步骤 1：词池预筛
  对每个模板槽位，把候选词+释义发给 LLM，让它判断哪些词
  能自然地填入普通话版本的句式模板。
  → 输出: filtered_pools.json

步骤 2：用筛后词池重新生成句子
  → 输出: generated_sentences_filtered.jsonl

步骤 3：成句批判
  把生成的普通话翻译批量发给 LLM，判断通顺性。
  → 输出: generated_sentences_final.jsonl

全程不需要人工审查。LLM 只需要懂普通话，不需要懂温州话。
"""

import json
import os
import re
import sys
import time
import hashlib
import random
import itertools
from collections import defaultdict
from pathlib import Path

# 强制行缓冲
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

LEXICON_PATH = ROOT / "data" / "controlled_generation" / "assets" / "replaceable_lexicon.jsonl"
SENTENCES_DIR = ROOT / "data" / "extracted_training_sentences"
OUT_DIR = ROOT / "data" / "sentence_templates"

random.seed(42)

# ---------- API ----------

def get_client():
    """用 DeepSeek（便宜、快、88.6% pass rate）。"""
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
    ), os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"


def call_llm(client, model, system_prompt, user_prompt, temperature=0.1, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=4000,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    API error: {e}, retry in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    API failed after {max_retries} retries: {e}")
                return None


# ---------- 语义过滤器（复用） ----------

SEMANTIC_FILTERS = {
    "adj_character": {
        "include": ["形容", "比喻", "态度", "脾气", "性格", "品质", "习性",
                     "心理", "懒", "笨", "聪明", "固执", "吝啬", "大方", "勤",
                     "精明", "凶", "猛", "胆小", "愚", "啰嗦", "邋遢", "狡",
                     "贪", "狠", "呆", "傻", "蠢", "骄", "谦", "急", "慢",
                     "好吃", "好赌", "嘴碎", "手脚", "不正", "坏习",
                     "小心", "粗心", "谨慎", "马虎", "爽快", "怕事"],
        "exclude": ["植物", "动物名", "鱼名", "鸟名", "工具", "器具", "地名",
                     "节气", "食物", "糕点", "药材", "疾病名", "菜名", "果名",
                     "一种", "虫", "蛇", "树"],
        "char_lens": [2, 3],
    },
    "verb_behavior": {
        "include": ["到处", "乱", "不正", "耍", "闹", "吵", "骂", "闲",
                     "多嘴", "多管", "偷", "赌", "撒", "装", "拖", "磨",
                     "浪费", "挥霍", "瞎", "胡", "犟", "争", "抢", "吹",
                     "行为", "举止"],
        "exclude": ["植物", "动物", "工具", "地名", "节气", "食物", "虫", "树",
                     "一种"],
        "char_lens": [2, 3, 4],
    },
    "verb_action": {
        "include": ["做", "干", "去", "来", "问", "找", "买", "卖",
                     "借", "还", "看", "听", "拿", "放", "送", "带", "帮",
                     "修", "洗", "扫", "煮", "烧", "写", "读", "学", "教",
                     "种", "收", "搬", "抬", "提", "挑", "捡", "接",
                     "商量", "讨论", "解决", "处理", "办", "管", "检查"],
        "exclude": ["植物", "鸟", "虫", "一种", "地名", "节气"],
        "char_lens": [2, 3],
    },
    "noun_person_role": {
        "include": ["指人", "称呼", "的人", "之人", "者", "师傅", "老师",
                     "工人", "医生", "商人", "农民", "学生", "儿童", "老人",
                     "妇女", "小孩", "邻居", "亲戚", "朋友", "同事"],
        "exclude": ["植物", "动物", "工具", "食物", "一种", "地名"],
        "char_lens": [2, 3],
    },
    "noun_concrete": {
        "include": ["东西", "物品", "器", "衣", "鞋", "帽", "碗", "盘",
                     "桌", "椅", "门", "窗", "书", "笔", "纸", "钱", "票",
                     "房", "屋", "车", "船", "菜", "饭", "米", "面", "肉",
                     "鱼", "茶", "酒", "药", "花", "树", "石", "水", "火"],
        "exclude": ["形容", "比喻", "态度"],
        "char_lens": [2, 3],
    },
    "state_change": {
        "include": ["开始", "变", "转", "好", "坏", "热", "冷",
                     "暖", "凉", "亮", "暗", "湿", "干", "满", "空",
                     "停", "完", "结束", "退", "散", "落", "涨"],
        "exclude": ["植物", "动物", "一种", "地名", "节气"],
        "char_lens": [2, 3],
    },
    "noun_place": {
        "include": ["地方", "里", "处", "场", "家", "店", "街", "路",
                     "山", "河", "海", "城", "村", "镇", "学校", "医院",
                     "市场", "码头", "桥", "弄堂", "巷"],
        "exclude": ["形容", "比喻", "性格"],
        "char_lens": [2, 3],
    },
    "verb_result": {
        "include": ["弄", "搞", "整", "修", "打", "拆", "烧", "煮", "洗",
                     "擦", "扔", "丢", "卖", "赔", "坏", "碎", "断", "烂"],
        "exclude": ["植物", "动物", "一种", "地名"],
        "char_lens": [2, 3],
    },
}


# ---------- 模板定义（短句+长句合并） ----------

TEMPLATES = [
    # ===== 短句模板 =====
    {"id": "S01", "tpl": "个人[X1]显", "zh": "这个人很[Z1]",
     "slots": [{"name": "X1", "filter": "adj_character", "zh_desc": "形容性格/状态的词"}]},
    {"id": "S02", "tpl": "该个人[X1]显", "zh": "这个人很[Z1]",
     "slots": [{"name": "X1", "filter": "adj_character", "zh_desc": "形容性格/状态的词"}]},
    {"id": "S03", "tpl": "渠[X1]显", "zh": "他/她很[Z1]",
     "slots": [{"name": "X1", "filter": "adj_character", "zh_desc": "形容性格/状态的词"}]},
    {"id": "S04", "tpl": "你真[X1]", "zh": "你真[Z1]",
     "slots": [{"name": "X1", "filter": "adj_character", "zh_desc": "形容性格/状态的词"}]},
    {"id": "S05", "tpl": "你恁[X1]", "zh": "你这么[Z1]",
     "slots": [{"name": "X1", "filter": "adj_character", "zh_desc": "形容性格/状态的词"}]},
    {"id": "S06", "tpl": "你覅[X1]", "zh": "你别[Z1]",
     "slots": [{"name": "X1", "filter": "verb_behavior", "zh_desc": "不良行为/可被劝阻的动作"}]},
    {"id": "S07", "tpl": "你不用[X1]", "zh": "你不用[Z1]",
     "slots": [{"name": "X1", "filter": "verb_behavior", "zh_desc": "不必要的行为/动作"}]},
    {"id": "S08", "tpl": "[X1]爻罢", "zh": "[Z1]了",
     "slots": [{"name": "X1", "filter": "state_change", "zh_desc": "表示状态变化或完成的动作"}]},
    {"id": "S09", "tpl": "[X1]不牢", "zh": "[Z1]不住",
     "slots": [{"name": "X1", "filter": "verb_action", "zh_desc": "可以搭配'不住'的动作"}]},
    {"id": "S10", "tpl": "走来[X1]", "zh": "来[Z1]",
     "slots": [{"name": "X1", "filter": "verb_action", "zh_desc": "来做的事情/目的"}]},
    {"id": "S11", "tpl": "走去[X1]", "zh": "去[Z1]",
     "slots": [{"name": "X1", "filter": "verb_action", "zh_desc": "去做的事情/目的"}]},
    {"id": "S12", "tpl": "你着[X1]", "zh": "你得[Z1]",
     "slots": [{"name": "X1", "filter": "verb_action", "zh_desc": "应该做的事情"}]},
    {"id": "S13", "tpl": "该个[X1]真[X2]", "zh": "这个[Z1]真[Z2]",
     "slots": [
         {"name": "X1", "filter": "noun_person_role", "zh_desc": "人物角色"},
         {"name": "X2", "filter": "adj_character", "zh_desc": "性格/状态形容词"},
     ]},

    # ===== 长句复合模板 =====
    {"id": "L01", "tpl": "该个人[X1]显，你覅伉渠[X2]",
     "zh": "这个人很[Z1]，你别跟他[Z2]",
     "slots": [
         {"name": "X1", "filter": "adj_character", "zh_desc": "性格/状态形容词"},
         {"name": "X2", "filter": "verb_behavior", "zh_desc": "不良行为/应避免的互动方式"},
     ]},
    {"id": "L02", "tpl": "个人恁[X1]，你覅[X2]，不好个",
     "zh": "这人这么[Z1]，你别[Z2]，不好的",
     "slots": [
         {"name": "X1", "filter": "adj_character", "zh_desc": "性格/状态形容词"},
         {"name": "X2", "filter": "verb_behavior", "zh_desc": "不良行为"},
     ]},
    {"id": "L03", "tpl": "你恁[X1]，人家沃不相信你个，覅[X2]",
     "zh": "你这么[Z1]，别人都不相信你的，别[Z2]",
     "slots": [
         {"name": "X1", "filter": "adj_character", "zh_desc": "导致不被信任的性格缺点"},
         {"name": "X2", "filter": "verb_behavior", "zh_desc": "应停止的行为"},
     ]},
    {"id": "L04", "tpl": "能界[X1]爻罢，个人还是[X2]显",
     "zh": "现在[Z1]了，这个人还是很[Z2]",
     "slots": [
         {"name": "X1", "filter": "state_change", "zh_desc": "某种变化/事件"},
         {"name": "X2", "filter": "adj_character", "zh_desc": "依然存在的性格特点"},
     ]},
    {"id": "L05", "tpl": "我伉你讲，你着[X1]，覅[X2]",
     "zh": "我跟你说，你得[Z1]，别[Z2]",
     "slots": [
         {"name": "X1", "filter": "verb_action", "zh_desc": "应该做的正确行为"},
         {"name": "X2", "filter": "verb_behavior", "zh_desc": "应该避免的行为"},
     ]},
    {"id": "L06", "tpl": "我伉你讲个，你覅[X1]，着[X2]牢",
     "zh": "我跟你说的，你别[Z1]，得[Z2]住",
     "slots": [
         {"name": "X1", "filter": "verb_behavior", "zh_desc": "应停止的行为"},
         {"name": "X2", "filter": "verb_action", "zh_desc": "能搭配'住'的动作"},
     ]},
    {"id": "L07", "tpl": "覅匄渠[X1]爻，不得[X2]不牢个",
     "zh": "别让他[Z1]了，否则[Z2]不住的",
     "slots": [
         {"name": "X1", "filter": "verb_action", "zh_desc": "需要阻止的动作"},
         {"name": "X2", "filter": "verb_action", "zh_desc": "可能失控的动作"},
     ]},
    {"id": "L08", "tpl": "渠[X1]显，称称能[X2]，你覅[X3]",
     "zh": "他/她很[Z1]，总是[Z2]，你别[Z3]",
     "slots": [
         {"name": "X1", "filter": "adj_character", "zh_desc": "性格特点"},
         {"name": "X2", "filter": "verb_behavior", "zh_desc": "经常做的行为"},
         {"name": "X3", "filter": "verb_behavior", "zh_desc": "应避免的回应"},
     ]},
    {"id": "L09", "tpl": "该个[X1]真[X2]，走来就[X3]",
     "zh": "这个[Z1]真[Z2]，一来就[Z3]",
     "slots": [
         {"name": "X1", "filter": "noun_person_role", "zh_desc": "人物角色"},
         {"name": "X2", "filter": "adj_character", "zh_desc": "性格评价"},
         {"name": "X3", "filter": "verb_behavior", "zh_desc": "到了就做的令人反感的行为"},
     ]},
    {"id": "L10", "tpl": "你着[X1]，覅[X2]，不得[X3]不起",
     "zh": "你得[Z1]，别[Z2]，否则[Z3]不起",
     "slots": [
         {"name": "X1", "filter": "verb_action", "zh_desc": "应做的正确动作"},
         {"name": "X2", "filter": "verb_behavior", "zh_desc": "应避免的行为"},
         {"name": "X3", "filter": "verb_action", "zh_desc": "可能承担不起的后果"},
     ]},
]


# ---------- 词典加载 ----------

def load_lexicon():
    entries = []
    with open(LEXICON_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            w = r["wz_word"]
            if r.get("slot_kind") == "function":
                continue
            if len(w) < 2 or len(w) > 4:
                continue
            if re.search(r"[?？□()（）]", w):
                continue
            entries.append({
                "wz_word": w,
                "mandarin": (r.get("mandarin_headword") or "")[:20],
                "definition": (r.get("definition") or ""),
                "kind": r.get("slot_kind", "unknown"),
                "char_len": len(w),
                "has_example": r.get("has_example", False),
            })
    return entries


def apply_rule_filter(entries, filter_name):
    filt = SEMANTIC_FILTERS[filter_name]
    inc_kw = filt["include"]
    exc_kw = filt["exclude"]
    char_lens = set(filt["char_lens"])
    pool = []
    seen = set()
    for e in entries:
        if e["char_len"] not in char_lens:
            continue
        # 不再限制 has_example — 让 LLM 决定质量
        defn = e["definition"]
        if any(k in defn for k in exc_kw):
            continue
        if inc_kw and not any(k in defn for k in inc_kw):
            continue
        if e["wz_word"] not in seen:
            seen.add(e["wz_word"])
            pool.append(e)
    return pool


def get_mandarin_short(entry):
    m = entry["mandarin"]
    if m and m != entry["wz_word"] and len(m) <= 15:
        return m
    defn = entry["definition"]
    short = defn.split("。")[0].split("，")[0].split("；")[0]
    if short and len(short) <= 15:
        return short
    return defn[:10]


# ---------- 步骤 1：LLM 词池预筛 ----------

POOL_FILTER_SYSTEM = """你是一个语言学助手。用户会给你一个中文句式模板和一批候选词（附释义）。
你需要判断每个词填入句式后，整句话是否**语义通顺、语法正确、生活中会自然使用**。

规则：
1. 只看中文/普通话层面的通顺性
2. 每个词回答"✓"（通顺）或"✗"（不通顺）
3. 严格按编号逐一回答，格式：序号.✓ 或 序号.✗
4. 不需要解释原因"""


def filter_pool_with_llm(client, model, template_zh, slot_desc, pool, batch_size=60):
    """对一个槽位的词池做 LLM 筛选，返回通过的词条列表。"""
    passed = []

    for batch_start in range(0, len(pool), batch_size):
        batch = pool[batch_start:batch_start + batch_size]

        # 构建候选词列表
        lines = []
        for i, entry in enumerate(batch, 1):
            zh_word = get_mandarin_short(entry)
            lines.append(f"{i}. {zh_word}（{entry['definition'][:30]}）")

        user_prompt = f"""句式模板：{template_zh}
槽位要求：{slot_desc}

候选词：
{chr(10).join(lines)}

请逐一判断每个词填入模板后是否通顺："""

        response = call_llm(client, model, POOL_FILTER_SYSTEM, user_prompt)
        if not response:
            # API 失败，保守策略：全部保留
            passed.extend(batch)
            continue

        # 解析结果
        for i, entry in enumerate(batch, 1):
            # 查找 "序号.✓" 或 "序号. ✓"
            pattern = rf"(?:^|\n)\s*{i}\s*[.．、]\s*✓"
            if re.search(pattern, response):
                passed.append(entry)

    return passed


# ---------- 步骤 2：生成句子 ----------

def fill_and_generate(tpl_def, filtered_pools, existing, max_combos=200):
    tpl_str = tpl_def["tpl"]
    zh_str = tpl_def["zh"]
    slots = tpl_def["slots"]
    results = []

    word_lists = [[e["wz_word"] for e in p] for p in filtered_pools]
    total = 1
    for wl in word_lists:
        total *= len(wl)

    if total <= max_combos:
        combos = list(itertools.product(*word_lists))
    else:
        combos = set()
        while len(combos) < max_combos:
            combo = tuple(random.choice(wl) for wl in word_lists)
            combos.add(combo)
        combos = list(combos)

    for combo in combos:
        wz = tpl_str
        zh = zh_str
        for i, slot in enumerate(slots):
            wz = wz.replace(f"[{slot['name']}]", combo[i], 1)
            mandarin = get_mandarin_short(
                next((e for e in filtered_pools[i] if e["wz_word"] == combo[i]),
                     {"mandarin": combo[i], "definition": combo[i], "wz_word": combo[i]})
            )
            zh = zh.replace(f"[Z{i+1}]", mandarin, 1)

        if wz in existing:
            continue
        char_len = len(wz)
        if char_len < 4 or char_len > 35:
            continue

        results.append({
            "sentence_id": f"af_{hashlib.md5(wz.encode()).hexdigest()[:10]}",
            "wz_sentence": wz,
            "zh_sentence": zh,
            "char_len": char_len,
            "length_bucket": "short" if char_len <= 20 else "long",
            "source": "template_llm_filtered",
            "template_id": tpl_def["id"],
            "template": tpl_str,
            "filled_words": list(combo),
        })
        existing.add(wz)

    return results


# ---------- 步骤 3：成句批判 ----------

CRITIC_SYSTEM = """你是中文语法审查员。用户会给你一批中文句子，请判断每句是否语法正确且语义通顺。

规则：
1. 判断标准：语法正确、意思说得通、日常生活中可能说出的话
2. 每句回答"✓"（通过）或"✗"（不通过）
3. 格式：序号.✓ 或 序号.✗
4. 不需要解释"""


def batch_critic(client, model, sentences, batch_size=40):
    """对生成的句子做普通话通顺性批判。"""
    results = []

    for batch_start in range(0, len(sentences), batch_size):
        batch = sentences[batch_start:batch_start + batch_size]

        lines = []
        for i, s in enumerate(batch, 1):
            lines.append(f"{i}. {s['zh_sentence']}")

        user_prompt = f"请判断以下句子是否通顺：\n\n{chr(10).join(lines)}"

        response = call_llm(client, model, CRITIC_SYSTEM, user_prompt)
        if not response:
            results.extend(batch)  # 保守：全保留
            continue

        for i, s in enumerate(batch, 1):
            pattern = rf"(?:^|\n)\s*{i}\s*[.．、]\s*✓"
            if re.search(pattern, response):
                s["critic_pass"] = True
            else:
                s["critic_pass"] = False
            results.append(s)

    return results


# ---------- 主流程 ----------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("加载词典...")
    all_entries = load_lexicon()
    print(f"  条目: {len(all_entries)}")

    # 加载已有句子
    existing = set()
    for fname in ["short_8_20.jsonl", "long_20_30.jsonl"]:
        path = SENTENCES_DIR / fname
        if path.exists():
            with open(path) as f:
                for line in f:
                    existing.add(json.loads(line)["wz_sentence"])

    client, model = get_client()
    print(f"  LLM: {model}")

    # ===== 步骤 1：LLM 词池预筛 =====
    print(f"\n===== 步骤 1：LLM 词池预筛 ({len(TEMPLATES)} 个模板) =====")
    template_pools = {}  # template_id -> list of pools

    # 先做规则过滤
    rule_pools = {}
    for fname in SEMANTIC_FILTERS:
        rule_pools[fname] = apply_rule_filter(all_entries, fname)

    # 检查是否有上次的 checkpoint
    pools_path = OUT_DIR / "filtered_pools.json"
    cached_pools = {}
    if pools_path.exists():
        with open(pools_path) as f:
            cached_pools = json.load(f)
        print(f"  发现 checkpoint: {len(cached_pools)} 个模板已有筛后词池")

    # 对每个模板每个槽位做 LLM 筛选
    for tpl_def in TEMPLATES:
        tid = tpl_def["id"]
        zh_tpl = tpl_def["zh"]
        slots = tpl_def["slots"]
        pools = []

        # 检查 checkpoint
        if tid in cached_pools and len(cached_pools[tid]) == len(slots):
            # 从 checkpoint 恢复
            for si, slot in enumerate(slots):
                cached = cached_pools[tid][si]
                # 重建 entry 对象
                cached_entries = []
                for ce in cached:
                    for e in rule_pools[slot["filter"]]:
                        if e["wz_word"] == ce["wz"]:
                            cached_entries.append(e)
                            break
                pools.append(cached_entries)
                print(f"  {tid}.slot[{si}] ({slot['name']}): 从checkpoint恢复 {len(cached_entries)} 词")
        else:
            for si, slot in enumerate(slots):
                rule_pool = rule_pools[slot["filter"]]
                print(f"  {tid}.slot[{si}] ({slot['name']}): 规则池 {len(rule_pool)} →", end=" ", flush=True)

                # LLM 筛选
                filtered = filter_pool_with_llm(
                    client, model,
                    template_zh=zh_tpl,
                    slot_desc=slot["zh_desc"],
                    pool=rule_pool,
                )
                print(f"LLM筛后 {len(filtered)}", flush=True)
                pools.append(filtered)

        template_pools[tid] = pools

        # 增量保存 checkpoint
        pools_export_inc = {}
        for t, ps in template_pools.items():
            pools_export_inc[t] = [
                [{"wz": e["wz_word"], "zh": get_mandarin_short(e)} for e in p]
                for p in ps
            ]
        with open(pools_path, "w", encoding="utf-8") as f:
            json.dump(pools_export_inc, f, ensure_ascii=False, indent=2)

    # 保存筛后词池
    pools_path = OUT_DIR / "filtered_pools.json"
    pools_export = {}
    for tid, pools in template_pools.items():
        pools_export[tid] = [
            [{"wz": e["wz_word"], "zh": get_mandarin_short(e)} for e in p]
            for p in pools
        ]
    with open(pools_path, "w", encoding="utf-8") as f:
        json.dump(pools_export, f, ensure_ascii=False, indent=2)
    print(f"\n  词池保存: {pools_path}")

    # ===== 步骤 2：生成句子 =====
    print(f"\n===== 步骤 2：用筛后词池生成 =====")
    all_generated = []

    for tpl_def in TEMPLATES:
        tid = tpl_def["id"]
        pools = template_pools[tid]
        pool_sizes = [len(p) for p in pools]

        if any(s < 3 for s in pool_sizes):
            print(f"  {tid} SKIP (pool太小: {pool_sizes})")
            continue

        generated = fill_and_generate(tpl_def, pools, existing)
        all_generated.extend(generated)
        short_n = sum(1 for g in generated if g["length_bucket"] == "short")
        long_n = len(generated) - short_n
        print(f"  {tid:4s} {tpl_def['tpl']:45s} pools={pool_sizes} → {len(generated):4d} (短{short_n}/长{long_n})")

    print(f"\n  步骤2 总计: {len(all_generated)} 句")

    # ===== 步骤 3：成句批判 =====
    print(f"\n===== 步骤 3：普通话通顺性批判 =====")
    all_judged = batch_critic(client, model, all_generated)

    passed = [s for s in all_judged if s.get("critic_pass", True)]
    failed = [s for s in all_judged if not s.get("critic_pass", True)]

    print(f"  通过: {len(passed)}, 淘汰: {len(failed)}")
    print(f"  通过率: {len(passed)*100/max(len(all_judged),1):.1f}%")

    # 写最终输出
    final_path = OUT_DIR / "generated_sentences_final.jsonl"
    with open(final_path, "w", encoding="utf-8") as f:
        for s in passed:
            s.pop("critic_pass", None)
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n  最终输出: {final_path} ({len(passed)} 句)")

    # 也写被淘汰的（调试用）
    rejected_path = OUT_DIR / "generated_sentences_rejected.jsonl"
    with open(rejected_path, "w", encoding="utf-8") as f:
        for s in failed:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 统计
    from collections import Counter
    bucket_dist = Counter(s["length_bucket"] for s in passed)
    tpl_dist = Counter(s["template_id"] for s in passed)

    summary = {
        "total_templates": len(TEMPLATES),
        "step1_rule_pools": {k: len(v) for k, v in rule_pools.items()},
        "step2_generated": len(all_generated),
        "step3_passed": len(passed),
        "step3_rejected": len(failed),
        "pass_rate": f"{len(passed)*100/max(len(all_judged),1):.1f}%",
        "length_buckets": dict(bucket_dist),
        "per_template_passed": dict(tpl_dist),
    }
    with open(OUT_DIR / "llm_filter_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n===== 最终汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 抽样
    print(f"\n===== 通过样例 =====")
    samples = random.sample(passed, min(15, len(passed)))
    for s in samples:
        print(f"  温: {s['wz_sentence']}  ({s['char_len']}字)")
        print(f"  普: {s['zh_sentence']}")
        print()


if __name__ == "__main__":
    main()
