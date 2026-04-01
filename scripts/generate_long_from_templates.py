#!/usr/bin/env python3
"""
轨道 A · 第 4 步：复合句式模板批量生成长句（20-30 字）。

策略：
  - 定义 2-3 子句复合模板，每个模板 2-4 个实词槽位
  - 子句用温州话常用连接词衔接（，、罢、就、还、不得）
  - 复用第 1-3 步的语义过滤词池
  - 目标：每模板 200 句 × 50+ 模板 = 10,000+ 长句

输出:
  data/sentence_templates/long_sentences.jsonl          — 生成的长句
  data/sentence_templates/long_generation_summary.json  — 统计
  data/sentence_templates/long_sample_review.tsv        — 抽样审查
"""

import hashlib
import itertools
import json
import os
import random
import re
from collections import Counter, defaultdict

random.seed(42)

LEXICON_PATH = "data/controlled_generation/assets/replaceable_lexicon.jsonl"
SENTENCES_DIR = "data/extracted_training_sentences"
OUT_DIR = "data/sentence_templates"

MAX_COMBOS = 200
MIN_POOL = 5

# ---------- 语义过滤器（同 curate_and_fill_templates.py） ----------
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
    "noun_time": {
        "include": ["时候", "天", "日", "月", "年", "早", "晚", "夜",
                     "春", "夏", "秋", "冬", "节", "季", "前", "后"],
        "exclude": ["植物", "动物", "一种", "工具"],
        "char_lens": [2, 3],
    },
}


# ---------- 复合长句模板 ----------
# 每条模板：
#   tpl: 模板字符串，用逗号分隔子句
#   zh:  普通话翻译模板
#   slots: 每个[X]的语义过滤器
#   min_len/max_len: 目标长度约束

LONG_TEMPLATES = [
    # ===== 评价 + 劝阻 (comment → advice) =====
    {
        "id": "L01", "category": "评价劝阻",
        "tpl": "该个人[X1]显，你覅伉渠[X2]",
        "zh": "这个人很[Z1]，你别跟他[Z2]",
        "slots": [
            {"name": "X1", "filter": "adj_character"},
            {"name": "X2", "filter": "verb_behavior"},
        ],
    },
    {
        "id": "L02", "category": "评价劝阻",
        "tpl": "渠[X1]显，你不用伉渠[X2]",
        "zh": "他/她很[Z1]，你不用跟他[Z2]",
        "slots": [
            {"name": "X1", "filter": "adj_character"},
            {"name": "X2", "filter": "verb_behavior"},
        ],
    },
    {
        "id": "L03", "category": "评价劝阻",
        "tpl": "个人恁[X1]，你覅[X2]，不好个",
        "zh": "这人这么[Z1]，你别[Z2]，不好的",
        "slots": [
            {"name": "X1", "filter": "adj_character"},
            {"name": "X2", "filter": "verb_behavior"},
        ],
    },
    {
        "id": "L04", "category": "评价劝阻",
        "tpl": "你恁[X1]，人家沃不相信你个，覅[X2]",
        "zh": "你这么[Z1]，别人都不相信你的，别[Z2]",
        "slots": [
            {"name": "X1", "filter": "adj_character"},
            {"name": "X2", "filter": "verb_behavior"},
        ],
    },

    # ===== 事件 + 评价 (event → comment) =====
    {
        "id": "L05", "category": "叙事评价",
        "tpl": "[X1]爻罢，该个人[X2]显",
        "zh": "[Z1]了，这个人很[Z2]",
        "slots": [
            {"name": "X1", "filter": "state_change"},
            {"name": "X2", "filter": "adj_character"},
        ],
    },
    {
        "id": "L06", "category": "叙事评价",
        "tpl": "能界[X1]爻罢，个人还是[X2]显",
        "zh": "现在[Z1]了，这个人还是很[Z2]",
        "slots": [
            {"name": "X1", "filter": "state_change"},
            {"name": "X2", "filter": "adj_character"},
        ],
    },

    # ===== 事件 + 结果 (event → result) =====
    {
        "id": "L07", "category": "叙事结果",
        "tpl": "[X1]起罢，[X2]阿[X3]不牢",
        "zh": "[Z1]起来了，[Z2]也[Z3]不住",
        "slots": [
            {"name": "X1", "filter": "state_change"},
            {"name": "X2", "filter": "noun_concrete"},
            {"name": "X3", "filter": "verb_action"},
        ],
    },
    {
        "id": "L08", "category": "叙事结果",
        "tpl": "逮[X1][X2]爻，能界[X3]不着",
        "zh": "把[Z1][Z2]了，现在[Z3]不了",
        "slots": [
            {"name": "X1", "filter": "noun_concrete"},
            {"name": "X2", "filter": "verb_action"},
            {"name": "X3", "filter": "verb_action"},
        ],
    },

    # ===== 趋向 + 目的 (direction → purpose) =====
    {
        "id": "L09", "category": "趋向目的",
        "tpl": "渠走来[X1]，你着[X2]准备",
        "zh": "他来[Z1]了，你得[Z2]准备",
        "slots": [
            {"name": "X1", "filter": "verb_action"},
            {"name": "X2", "filter": "verb_action"},
        ],
    },
    {
        "id": "L10", "category": "趋向目的",
        "tpl": "我走去[X1]，你宿屋里[X2]牢",
        "zh": "我去[Z1]，你在家里[Z2]着",
        "slots": [
            {"name": "X1", "filter": "verb_action"},
            {"name": "X2", "filter": "verb_action"},
        ],
    },

    # ===== 疑问 + 评价 (question → comment) =====
    {
        "id": "L11", "category": "疑问评价",
        "tpl": "訾那[X1]显？该个人真[X2]",
        "zh": "怎么这么[Z1]？这个人真[Z2]",
        "slots": [
            {"name": "X1", "filter": "adj_character"},
            {"name": "X2", "filter": "adj_character"},
        ],
    },
    {
        "id": "L12", "category": "疑问评价",
        "tpl": "你訾那恁[X1]？人家沃[X2]显",
        "zh": "你怎么这么[Z1]？别人都很[Z2]",
        "slots": [
            {"name": "X1", "filter": "adj_character"},
            {"name": "X2", "filter": "adj_character"},
        ],
    },

    # ===== 被动 + 后果 (passive → consequence) =====
    {
        "id": "L13", "category": "被动后果",
        "tpl": "匄渠[X1]爻，能界[X2]不着",
        "zh": "被他[Z1]了，现在[Z2]不了",
        "slots": [
            {"name": "X1", "filter": "verb_action"},
            {"name": "X2", "filter": "verb_action"},
        ],
    },
    {
        "id": "L14", "category": "被动后果",
        "tpl": "覅匄渠[X1]爻，不得[X2]不牢个",
        "zh": "别让他[Z1]了，否则[Z2]不住的",
        "slots": [
            {"name": "X1", "filter": "verb_action"},
            {"name": "X2", "filter": "verb_action"},
        ],
    },

    # ===== 条件 + 结果 (condition → result) =====
    {
        "id": "L15", "category": "条件结果",
        "tpl": "你着[X1]，覅[X2]，不得[X3]不起",
        "zh": "你得[Z1]，别[Z2]，否则[Z3]不起",
        "slots": [
            {"name": "X1", "filter": "verb_action"},
            {"name": "X2", "filter": "verb_behavior"},
            {"name": "X3", "filter": "verb_action"},
        ],
    },
    {
        "id": "L16", "category": "条件结果",
        "tpl": "[X1]爻罢就[X2]，覅[X3]显",
        "zh": "[Z1]了就[Z2]，别[Z3]得很",
        "slots": [
            {"name": "X1", "filter": "state_change"},
            {"name": "X2", "filter": "verb_action"},
            {"name": "X3", "filter": "verb_behavior"},
        ],
    },

    # ===== 处置 + 评价 (disposal → evaluation) =====
    {
        "id": "L17", "category": "处置评价",
        "tpl": "逮[X1][X2]爻，该个人真[X3]",
        "zh": "把[Z1][Z2]了，这个人真[Z3]",
        "slots": [
            {"name": "X1", "filter": "noun_concrete"},
            {"name": "X2", "filter": "verb_action"},
            {"name": "X3", "filter": "adj_character"},
        ],
    },

    # ===== 时间 + 变化 (time → change) =====
    {
        "id": "L18", "category": "时间变化",
        "tpl": "能界[X1]爻罢，[X2]沃[X3]起",
        "zh": "现在[Z1]了，[Z2]都[Z3]起来",
        "slots": [
            {"name": "X1", "filter": "state_change"},
            {"name": "X2", "filter": "noun_concrete"},
            {"name": "X3", "filter": "state_change"},
        ],
    },

    # ===== 对话 + 要求 (dialog → demand) =====
    {
        "id": "L19", "category": "对话要求",
        "tpl": "我伉你讲，你着[X1]，覅[X2]",
        "zh": "我跟你说，你得[Z1]，别[Z2]",
        "slots": [
            {"name": "X1", "filter": "verb_action"},
            {"name": "X2", "filter": "verb_behavior"},
        ],
    },
    {
        "id": "L20", "category": "对话要求",
        "tpl": "我伉你讲个，你覅[X1]，着[X2]牢",
        "zh": "我跟你说的，你别[Z1]，得[Z2]住",
        "slots": [
            {"name": "X1", "filter": "verb_behavior"},
            {"name": "X2", "filter": "verb_action"},
        ],
    },

    # ===== 判断 + 解释 =====
    {
        "id": "L21", "category": "判断解释",
        "tpl": "该个[X1]沃是[X2]个，冇[X3]",
        "zh": "这个[Z1]都是[Z2]的，没[Z3]",
        "slots": [
            {"name": "X1", "filter": "noun_person_role"},
            {"name": "X2", "filter": "adj_character"},
            {"name": "X3", "filter": "verb_action"},
        ],
    },

    # ===== 多子句评价叙事 =====
    {
        "id": "L22", "category": "评价叙事",
        "tpl": "该个[X1]真[X2]，走来就[X3]",
        "zh": "这个[Z1]真[Z2]，一来就[Z3]",
        "slots": [
            {"name": "X1", "filter": "noun_person_role"},
            {"name": "X2", "filter": "adj_character"},
            {"name": "X3", "filter": "verb_behavior"},
        ],
    },
    {
        "id": "L23", "category": "评价叙事",
        "tpl": "你真[X1]，逮[X2]沃[X3]爻",
        "zh": "你真[Z1]，把[Z2]都[Z3]了",
        "slots": [
            {"name": "X1", "filter": "adj_character"},
            {"name": "X2", "filter": "noun_concrete"},
            {"name": "X3", "filter": "verb_action"},
        ],
    },
    {
        "id": "L24", "category": "评价叙事",
        "tpl": "渠[X1]显，称称能[X2]，你覅[X3]",
        "zh": "他/她很[Z1]，总是[Z2]，你别[Z3]",
        "slots": [
            {"name": "X1", "filter": "adj_character"},
            {"name": "X2", "filter": "verb_behavior"},
            {"name": "X3", "filter": "verb_behavior"},
        ],
    },

    # ===== 存在 + 建议 =====
    {
        "id": "L25", "category": "描述建议",
        "tpl": "该个[X1]有[X2]个，你着[X3]",
        "zh": "这个[Z1]有[Z2]的，你得[Z3]",
        "slots": [
            {"name": "X1", "filter": "noun_place"},
            {"name": "X2", "filter": "noun_concrete"},
            {"name": "X3", "filter": "verb_action"},
        ],
    },
]


# ---------- 词典 + 过滤 ----------

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


def apply_filter(entries, filter_name):
    filt = SEMANTIC_FILTERS[filter_name]
    inc_kw = filt["include"]
    exc_kw = filt["exclude"]
    char_lens = set(filt["char_lens"])

    pool = []
    seen = set()
    for e in entries:
        if e["char_len"] not in char_lens:
            continue
        if not e["has_example"]:
            continue
        defn = e["definition"]
        if any(k in defn for k in exc_kw):
            continue
        if inc_kw and not any(k in defn for k in inc_kw):
            continue
        if e["wz_word"] not in seen:
            seen.add(e["wz_word"])
            pool.append(e)

    # Fallback if too small
    if len(pool) < MIN_POOL:
        seen2 = set()
        pool2 = []
        for e in entries:
            if e["char_len"] not in char_lens:
                continue
            defn = e["definition"]
            if any(k in defn for k in exc_kw):
                continue
            if inc_kw and not any(k in defn for k in inc_kw):
                continue
            if e["wz_word"] not in seen2:
                seen2.add(e["wz_word"])
                pool2.append(e)
        return pool2

    return pool


def get_mandarin(word, pool):
    for e in pool:
        if e["wz_word"] == word:
            m = e["mandarin"]
            if m and m != word and len(m) <= 15:
                return m
            defn = e["definition"]
            short_def = defn.split("。")[0].split("，")[0].split("；")[0]
            if short_def and len(short_def) <= 15:
                return short_def
            return defn[:10]
    return word


def load_existing():
    existing = set()
    for fname in ["short_8_20.jsonl", "long_20_30.jsonl"]:
        path = os.path.join(SENTENCES_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    existing.add(json.loads(line)["wz_sentence"])
    # 也加入短句生成的
    gen_path = os.path.join(OUT_DIR, "generated_sentences.jsonl")
    if os.path.exists(gen_path):
        with open(gen_path) as f:
            for line in f:
                existing.add(json.loads(line)["wz_sentence"])
    return existing


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("加载词典...")
    all_entries = load_lexicon()
    print(f"  条目: {len(all_entries)}")

    existing = load_existing()
    print(f"  已有句子: {len(existing)}")

    # 预构建所有过滤器的词池
    print("\n构建词池...")
    filter_pools = {}
    for fname in SEMANTIC_FILTERS:
        pool = apply_filter(all_entries, fname)
        filter_pools[fname] = pool
        print(f"  {fname:20s}: {len(pool)} 词")

    # 生成
    print(f"\n===== 批量生成长句 ({len(LONG_TEMPLATES)} 个模板) =====")
    all_generated = []
    per_tpl_stats = []

    for tpl_def in LONG_TEMPLATES:
        tid = tpl_def["id"]
        tpl_str = tpl_def["tpl"]
        zh_str = tpl_def["zh"]
        slots = tpl_def["slots"]

        # 获取词池
        pools = [filter_pools[s["filter"]] for s in slots]
        pool_sizes = [len(p) for p in pools]

        if any(s < MIN_POOL for s in pool_sizes):
            per_tpl_stats.append({"id": tid, "tpl": tpl_str, "generated": 0,
                                  "reason": "pool_too_small", "pools": pool_sizes})
            print(f"  {tid} SKIP pools={pool_sizes}")
            continue

        # 笛卡尔积 + 采样
        word_lists = [[e["wz_word"] for e in p] for p in pools]
        total_combos = 1
        for wl in word_lists:
            total_combos *= len(wl)

        if total_combos <= MAX_COMBOS:
            combos = list(itertools.product(*word_lists))
        else:
            # 采样
            combos = set()
            while len(combos) < MAX_COMBOS:
                combo = tuple(random.choice(wl) for wl in word_lists)
                combos.add(combo)
            combos = list(combos)

        generated = []
        for combo in combos:
            # 填充温州话
            wz = tpl_str
            for i, slot in enumerate(slots):
                wz = wz.replace(f"[{slot['name']}]", combo[i], 1)

            if wz in existing:
                continue

            char_len = len(wz)
            # 长句目标 20-30
            if char_len < 15 or char_len > 35:
                continue

            # 填充普通话
            zh = zh_str
            for i, slot in enumerate(slots):
                mandarin = get_mandarin(combo[i], pools[i])
                zh = zh.replace(f"[Z{i+1}]", mandarin, 1)

            generated.append({
                "sentence_id": f"ltpl_{hashlib.md5(wz.encode()).hexdigest()[:10]}",
                "wz_sentence": wz,
                "zh_sentence": zh,
                "char_len": char_len,
                "length_bucket": "short" if char_len <= 20 else "long",
                "source": "long_template_fill",
                "template_id": tid,
                "template": tpl_str,
                "category": tpl_def["category"],
                "filled_words": [
                    {"slot": slots[i]["name"], "wz": combo[i],
                     "mandarin": get_mandarin(combo[i], pools[i])}
                    for i in range(len(combo))
                ],
            })
            existing.add(wz)

        all_generated.extend(generated)
        short_n = sum(1 for g in generated if g["length_bucket"] == "short")
        long_n = len(generated) - short_n
        per_tpl_stats.append({
            "id": tid, "tpl": tpl_str,
            "pools": pool_sizes, "generated": len(generated),
            "short": short_n, "long": long_n,
        })
        print(f"  {tid:4s} {tpl_str:45s} pools={pool_sizes} → {len(generated):4d} (短{short_n}/长{long_n})")

    # 写 JSONL
    out_path = os.path.join(OUT_DIR, "long_sentences.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for g in all_generated:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"\n  总生成: {len(all_generated)} 句 → {out_path}")

    # 写抽样 TSV
    tsv_path = os.path.join(OUT_DIR, "long_sample_review.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("模板ID\t模板\t类别\t温州话\t普通话\t字数\t填充词\t通顺?\t备注\n")
        by_tpl = defaultdict(list)
        for g in all_generated:
            by_tpl[g["template_id"]].append(g)
        for tid in sorted(by_tpl):
            sents = by_tpl[tid]
            samples = random.sample(sents, min(5, len(sents)))
            for s in samples:
                fills = " | ".join(f"{fw['slot']}={fw['wz']}({fw['mandarin']})" for fw in s["filled_words"])
                f.write(f"{s['template_id']}\t{s['template']}\t{s['category']}\t"
                        f"{s['wz_sentence']}\t{s['zh_sentence']}\t{s['char_len']}\t{fills}\t\t\n")
    print(f"  抽样 TSV: {tsv_path}")

    # 统计
    bucket_dist = Counter(g["length_bucket"] for g in all_generated)
    cat_dist = Counter(g["category"] for g in all_generated)
    len_dist = Counter(g["char_len"] for g in all_generated)

    summary = {
        "total_templates": len(LONG_TEMPLATES),
        "total_generated": len(all_generated),
        "length_buckets": dict(bucket_dist),
        "category_distribution": dict(cat_dist),
        "char_len_distribution": {str(k): v for k, v in sorted(len_dist.items())},
        "per_template": per_tpl_stats,
    }
    with open(os.path.join(OUT_DIR, "long_generation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n===== 汇总 =====")
    print(f"  模板: {len(LONG_TEMPLATES)}")
    print(f"  总生成: {len(all_generated)} (长句 {bucket_dist.get('long',0)}, 短句 {bucket_dist.get('short',0)})")
    print(f"  类别: {dict(cat_dist)}")

    # 抽样
    print(f"\n===== 抽样 =====")
    for cat in sorted(cat_dist):
        sents = [g for g in all_generated if g["category"] == cat]
        samples = random.sample(sents, min(3, len(sents)))
        print(f"\n  [{cat}]")
        for s in samples:
            fills = ", ".join(f"{fw['wz']}({fw['mandarin']})" for fw in s["filled_words"])
            print(f"    温: {s['wz_sentence']}  ({s['char_len']}字)")
            print(f"    普: {s['zh_sentence']}")
            print(f"    填: {fills}")


if __name__ == "__main__":
    main()
