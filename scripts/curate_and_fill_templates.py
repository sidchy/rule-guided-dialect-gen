#!/usr/bin/env python3
"""
轨道 A · 第 1-3 步合并脚本：
  1. 定义句式模板 + 槽位语义过滤器
  2. 从词典中按语义筛选兼容词池（每槽 ~50-200 词）
  3. 笛卡尔积批量组合生成句子

设计原则：
  - 每个 [X] 对应 1 个实词（2-4 字）
  - 语义过滤靠释义关键词匹配（不靠 slot_kind，太粗）
  - 输出标注清晰：template_id, category, slot fills + mandarin
  - 每条句子带 concat_tags 方便后续第 4 步拼长句

输出:
  data/sentence_templates/curated_templates.jsonl     — 带词池统计的模板
  data/sentence_templates/generated_sentences.jsonl   — 生成句
  data/sentence_templates/generation_summary.json     — 统计
  data/sentence_templates/sample_review.tsv           — 抽样人工检查用
"""

import hashlib
import itertools
import json
import os
import random
import re
from collections import Counter, defaultdict

random.seed(42)

# ---------- 路径 ----------
LEXICON_PATH = "data/controlled_generation/assets/replaceable_lexicon.jsonl"
SENTENCES_DIR = "data/extracted_training_sentences"
OUT_DIR = "data/sentence_templates"

# ---------- 配置 ----------
MAX_COMBOS_PER_TEMPLATE = 200
MIN_POOL_SIZE = 5

# ---------- 语义过滤器定义 ----------
# 每种过滤器: include_keywords (释义中含任一即候选)
#             exclude_keywords (释义中含任一则排除)
#             prefer_has_example: 优先有例句的词

SEMANTIC_FILTERS = {
    "adj_character": {
        # 性格/状态/品质形容词 — 用于 "X显" "你真X" "你恁X"
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
        # 可禁止/劝阻的行为 — 用于 "你覅X" "你不用X" "𧟰X"
        "include": ["到处", "乱", "不正", "耍", "闹", "吵", "骂", "闲",
                     "多嘴", "多管", "偷", "赌", "撒", "装", "拖", "磨",
                     "浪费", "挥霍", "嚼", "说", "讲", "瞎", "胡", "犟",
                     "争", "抢", "吹", "行为", "举止", "习惯不好"],
        "exclude": ["植物", "动物", "工具", "地名", "节气", "食物", "虫", "树",
                     "一种"],
        "char_lens": [2, 3, 4],
    },
    "verb_action": {
        # 一般动作 — 用于 "走来X" "走去X" "你着X" "匄渠X"
        "include": ["动作", "做", "干", "去", "来", "问", "找", "买", "卖",
                     "借", "还", "看", "听", "拿", "放", "送", "带", "帮",
                     "修", "洗", "扫", "煮", "烧", "写", "读", "学", "教",
                     "种", "收", "搬", "抬", "提", "挑", "捡", "接", "打电话",
                     "商量", "讨论", "解决", "处理", "办", "管", "检查"],
        "exclude": ["植物", "鸟", "虫", "一种", "地名", "节气"],
        "char_lens": [2, 3],
    },
    "noun_person_role": {
        # 人物角色 — 用于 "该个[X]真..."
        "include": ["指人", "称呼", "的人", "之人", "者", "师傅", "老师",
                     "工人", "医生", "商人", "农民", "学生", "儿童", "老人",
                     "妇女", "小孩", "邻居", "亲戚", "朋友", "同事"],
        "exclude": ["植物", "动物", "工具", "食物", "一种", "地名"],
        "char_lens": [2, 3],
    },
    "noun_concrete": {
        # 具体事物 — 用于 "逮[X]..." "该[X]..."
        "include": ["东西", "物品", "器", "具", "衣", "鞋", "帽", "碗", "盘",
                     "桌", "椅", "门", "窗", "书", "笔", "纸", "钱", "票",
                     "房", "屋", "车", "船", "菜", "饭", "米", "面", "肉",
                     "鱼", "茶", "酒", "药", "花", "树", "石", "水", "火"],
        "exclude": ["形容", "比喻", "态度"],
        "char_lens": [2, 3],
    },
    "verb_result": {
        # 动作+结果 — 用于 "逮X [X]爻"
        "include": ["弄", "搞", "整", "修", "打", "拆", "烧", "煮", "洗",
                     "擦", "扔", "丢", "卖", "赔", "坏", "碎", "断", "烂"],
        "exclude": ["植物", "动物", "一种", "地名"],
        "char_lens": [2, 3],
    },
    "state_change": {
        # 状态变化 — 用于 "[X]起罢" "[X]爻罢"
        "include": ["开始", "起来", "变", "转", "好", "坏", "热", "冷",
                     "暖", "凉", "亮", "暗", "湿", "干", "满", "空",
                     "停", "完", "毕", "结束", "退", "散", "落", "涨"],
        "exclude": ["植物", "动物", "一种", "地名", "节气"],
        "char_lens": [2, 3],
    },
    "any_content": {
        # 宽泛内容词（兜底）— 只过滤掉明显不搭的
        "include": [],  # 空 = 不做include过滤
        "exclude": ["偏旁", "笔画", "部首", "数词", "量词",
                     "拼音", "字母", "符号"],
        "char_lens": [2, 3],
    },
}


# ---------- 模板定义（用语义过滤器名称引用槽位类型） ----------
TEMPLATES = [
    # ===== 评价类（适合拼长句的前半句：主题+评价） =====
    {
        "id": "eval_01", "template": "个人[X]显",
        "zh_tpl": "这个人很[X1]",
        "category": "评价", "concat_role": "comment",
        "slots": [{"filter": "adj_character", "zh_slot": "[X1]"}],
    },
    {
        "id": "eval_02", "template": "该个人[X]显",
        "zh_tpl": "这个人很[X1]",
        "category": "评价", "concat_role": "comment",
        "slots": [{"filter": "adj_character", "zh_slot": "[X1]"}],
    },
    {
        "id": "eval_03", "template": "渠[X]显",
        "zh_tpl": "他/她很[X1]",
        "category": "评价", "concat_role": "comment",
        "slots": [{"filter": "adj_character", "zh_slot": "[X1]"}],
    },
    {
        "id": "eval_04", "template": "你真[X]",
        "zh_tpl": "你真[X1]",
        "category": "评价", "concat_role": "comment",
        "slots": [{"filter": "adj_character", "zh_slot": "[X1]"}],
    },
    {
        "id": "eval_05", "template": "你恁[X]",
        "zh_tpl": "你这么[X1]",
        "category": "评价", "concat_role": "comment",
        "slots": [{"filter": "adj_character", "zh_slot": "[X1]"}],
    },
    {
        "id": "eval_06", "template": "该个[X]真[X]",
        "zh_tpl": "这个[X1]真[X2]",
        "category": "评价", "concat_role": "comment",
        "slots": [
            {"filter": "noun_person_role", "zh_slot": "[X1]"},
            {"filter": "adj_character", "zh_slot": "[X2]"},
        ],
    },

    # ===== 禁止/劝阻类（适合拼长句的后半句：劝告） =====
    {
        "id": "prohib_01", "template": "你覅[X]",
        "zh_tpl": "你别[X1]",
        "category": "禁止", "concat_role": "advice",
        "slots": [{"filter": "verb_behavior", "zh_slot": "[X1]"}],
    },
    {
        "id": "prohib_02", "template": "你不用[X]",
        "zh_tpl": "你不用[X1]",
        "category": "禁止", "concat_role": "advice",
        "slots": [{"filter": "verb_behavior", "zh_slot": "[X1]"}],
    },
    {
        "id": "prohib_03", "template": "覅匄渠[X]爻",
        "zh_tpl": "别让他[X1]了",
        "category": "禁止", "concat_role": "advice",
        "slots": [{"filter": "verb_action", "zh_slot": "[X1]"}],
    },

    # ===== 完成体（适合叙事前半句） =====
    {
        "id": "compl_01", "template": "[X]爻罢",
        "zh_tpl": "[X1]了",
        "category": "完成", "concat_role": "event",
        "slots": [{"filter": "state_change", "zh_slot": "[X1]"}],
    },
    {
        "id": "compl_02", "template": "[X]起罢",
        "zh_tpl": "[X1]起来了",
        "category": "完成", "concat_role": "event",
        "slots": [{"filter": "state_change", "zh_slot": "[X1]"}],
    },

    # ===== 否定结果补语（适合拼长句后半句：能力/结果） =====
    {
        "id": "neg_01", "template": "[X]不牢",
        "zh_tpl": "[X1]不住",
        "category": "否定结果", "concat_role": "result",
        "slots": [{"filter": "verb_action", "zh_slot": "[X1]"}],
    },
    {
        "id": "neg_02", "template": "[X]不着",
        "zh_tpl": "[X1]不了",
        "category": "否定结果", "concat_role": "result",
        "slots": [{"filter": "verb_action", "zh_slot": "[X1]"}],
    },
    {
        "id": "neg_03", "template": "[X]不落",
        "zh_tpl": "[X1]不下",
        "category": "否定结果", "concat_role": "result",
        "slots": [{"filter": "verb_action", "zh_slot": "[X1]"}],
    },
    {
        "id": "neg_04", "template": "[X]不起",
        "zh_tpl": "[X1]不起",
        "category": "否定结果", "concat_role": "result",
        "slots": [{"filter": "verb_action", "zh_slot": "[X1]"}],
    },

    # ===== 趋向类（适合叙事：来/去做某事） =====
    {
        "id": "dir_01", "template": "走来[X]",
        "zh_tpl": "来[X1]",
        "category": "趋向", "concat_role": "action",
        "slots": [{"filter": "verb_action", "zh_slot": "[X1]"}],
    },
    {
        "id": "dir_02", "template": "走去[X]",
        "zh_tpl": "去[X1]",
        "category": "趋向", "concat_role": "action",
        "slots": [{"filter": "verb_action", "zh_slot": "[X1]"}],
    },

    # ===== 疑问类（适合对话上下文） =====
    {
        "id": "q_01", "template": "訾那[X]",
        "zh_tpl": "怎么[X1]",
        "category": "疑问", "concat_role": "question",
        "slots": [{"filter": "adj_character", "zh_slot": "[X1]"}],
    },

    # ===== 祈使类 =====
    {
        "id": "imp_01", "template": "你着[X]",
        "zh_tpl": "你得[X1]",
        "category": "祈使", "concat_role": "advice",
        "slots": [{"filter": "verb_action", "zh_slot": "[X1]"}],
    },

    # ===== 时间/状态 =====
    {
        "id": "time_01", "template": "能界[X]",
        "zh_tpl": "现在[X1]",
        "category": "时间", "concat_role": "event",
        "slots": [{"filter": "state_change", "zh_slot": "[X1]"}],
    },

    # ===== 被动/处置 =====
    {
        "id": "pass_01", "template": "匄渠[X]",
        "zh_tpl": "让他[X1]",
        "category": "被动", "concat_role": "event",
        "slots": [{"filter": "verb_action", "zh_slot": "[X1]"}],
    },
    {
        "id": "disp_01", "template": "逮[X][X]爻",
        "zh_tpl": "把[X1][X2]了",
        "category": "处置", "concat_role": "event",
        "slots": [
            {"filter": "noun_concrete", "zh_slot": "[X1]"},
            {"filter": "verb_result", "zh_slot": "[X2]"},
        ],
    },

    # ===== 判断/强调 =====
    {
        "id": "judge_01", "template": "沃是[X]个",
        "zh_tpl": "都是[X1]的",
        "category": "判断", "concat_role": "comment",
        "slots": [{"filter": "adj_character", "zh_slot": "[X1]"}],
    },

    # ===== 对话类 =====
    {
        "id": "dialog_01", "template": "我伉你[X]",
        "zh_tpl": "我跟你[X1]",
        "category": "对话", "concat_role": "action",
        "slots": [{"filter": "verb_action", "zh_slot": "[X1]"}],
    },
]


# ---------- 词典加载 ----------

def load_lexicon():
    """加载词典，返回全量列表 + 按(kind,len)分桶。"""
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
                "scene": r.get("primary_scene_id", ""),
            })
    return entries


def apply_semantic_filter(entries, filter_name, require_example=True):
    """用语义过滤器筛选词池。require_example=True 只选有例句的词（质量更高）。"""
    filt = SEMANTIC_FILTERS[filter_name]
    inc_kw = filt["include"]
    exc_kw = filt["exclude"]
    char_lens = set(filt["char_lens"])

    pool = []
    for e in entries:
        if e["char_len"] not in char_lens:
            continue
        if require_example and not e["has_example"]:
            continue
        defn = e["definition"]
        # exclude check
        if any(k in defn for k in exc_kw):
            continue
        # include check (empty = no filter)
        if inc_kw and not any(k in defn for k in inc_kw):
            continue
        pool.append(e)

    # 去重
    seen = set()
    deduped = []
    for e in pool:
        if e["wz_word"] not in seen:
            seen.add(e["wz_word"])
            deduped.append(e)

    # 如果 require_example 导致池子太小，放宽
    if len(deduped) < MIN_POOL_SIZE and require_example:
        return apply_semantic_filter(entries, filter_name, require_example=False)

    deduped.sort(key=lambda x: (not x["has_example"], x["wz_word"]))
    return deduped


# ---------- 句子生成 ----------

def fill_template(tpl_str, combo):
    result = tpl_str
    for word in combo:
        result = result.replace("[X]", word, 1)
    return result


def get_mandarin(word, pool):
    """查找词的最佳普通话翻译：mandarin_headword > definition前15字 > 原词。"""
    for e in pool:
        if e["wz_word"] == word:
            m = e["mandarin"]
            if m and m != word and len(m) <= 15:
                return m
            # fallback: 取 definition 第一个句号前的内容
            defn = e["definition"]
            short_def = defn.split("。")[0].split("，")[0].split("；")[0]
            if short_def and len(short_def) <= 15:
                return short_def
            return defn[:10]
    return word


def fill_zh(zh_tpl, combo, pools):
    result = zh_tpl
    for i, word in enumerate(combo):
        mandarin = get_mandarin(word, pools[i])
        result = result.replace(f"[X{i+1}]", mandarin, 1)
    return result


def load_existing_wz():
    """加载已有句子用于去重。"""
    existing = set()
    for fname in ["short_8_20.jsonl", "long_20_30.jsonl"]:
        path = os.path.join(SENTENCES_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                existing.add(json.loads(line)["wz_sentence"])
    return existing


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("加载词典...")
    all_entries = load_lexicon()
    print(f"  实词条目: {len(all_entries)}")

    existing_wz = load_existing_wz()
    print(f"  已有句子: {len(existing_wz)}")

    # ===== 构建词池 + 输出 curated 模板 =====
    print("\n===== 构建词池 =====")
    curated_records = []
    all_pools = {}  # template_id -> list of pools

    for tpl_def in TEMPLATES:
        tid = tpl_def["id"]
        pools = []
        for si, slot in enumerate(tpl_def["slots"]):
            pool = apply_semantic_filter(all_entries, slot["filter"])
            pools.append(pool)
        all_pools[tid] = pools

        pool_sizes = [len(p) for p in pools]
        viable = all(s >= MIN_POOL_SIZE for s in pool_sizes)

        curated_records.append({
            "template_id": tid,
            "template": tpl_def["template"],
            "zh_template": tpl_def["zh_tpl"],
            "category": tpl_def["category"],
            "concat_role": tpl_def["concat_role"],
            "slot_count": len(tpl_def["slots"]),
            "slots": [
                {
                    "index": si,
                    "filter": slot["filter"],
                    "char_lens": SEMANTIC_FILTERS[slot["filter"]]["char_lens"],
                    "pool_size": len(pools[si]),
                    "sample_words": [e["wz_word"] for e in pools[si][:10]],
                }
                for si, slot in enumerate(tpl_def["slots"])
            ],
            "viable": viable,
        })

        status = "OK" if viable else "SKIP"
        print(f"  {tid:12s} [{status}] {tpl_def['template']:20s} pools={pool_sizes}")

    curated_path = os.path.join(OUT_DIR, "curated_templates.jsonl")
    with open(curated_path, "w", encoding="utf-8") as f:
        for c in curated_records:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\n  curated 模板: {curated_path} ({len(curated_records)} 条)")

    # ===== 批量生成 =====
    print("\n===== 批量生成 =====")
    all_generated = []
    per_tpl_stats = []

    for tpl_def in TEMPLATES:
        tid = tpl_def["id"]
        pools = all_pools[tid]

        if not all(len(p) >= MIN_POOL_SIZE for p in pools):
            per_tpl_stats.append({"id": tid, "template": tpl_def["template"],
                                  "generated": 0, "reason": "pool_too_small"})
            continue

        word_lists = [[e["wz_word"] for e in p] for p in pools]
        combos = list(itertools.product(*word_lists))
        if len(combos) > MAX_COMBOS_PER_TEMPLATE:
            combos = random.sample(combos, MAX_COMBOS_PER_TEMPLATE)

        generated = []
        for combo in combos:
            wz = fill_template(tpl_def["template"], combo)
            if wz in existing_wz:
                continue
            char_len = len(wz)
            if char_len < 4 or char_len > 30:
                continue

            zh = fill_zh(tpl_def["zh_tpl"], combo, pools)

            generated.append({
                "sentence_id": f"tpl_{hashlib.md5(wz.encode()).hexdigest()[:10]}",
                "wz_sentence": wz,
                "zh_sentence": zh,
                "char_len": char_len,
                "length_bucket": "short" if char_len <= 20 else "long",
                # ----- 标注信息（后续拼长句用） -----
                "source": "template_fill",
                "template_id": tid,
                "template": tpl_def["template"],
                "category": tpl_def["category"],
                "concat_role": tpl_def["concat_role"],  # comment/advice/event/action/question/result
                "filled_words": [
                    {
                        "slot_index": i,
                        "wz_word": combo[i],
                        "mandarin": next(
                            (e["mandarin"] for e in pools[i] if e["wz_word"] == combo[i]),
                            combo[i]
                        ),
                    }
                    for i in range(len(combo))
                ],
            })
            existing_wz.add(wz)

        all_generated.extend(generated)
        short_n = sum(1 for g in generated if g["length_bucket"] == "short")
        long_n = len(generated) - short_n
        per_tpl_stats.append({
            "id": tid, "template": tpl_def["template"],
            "generated": len(generated), "short": short_n, "long": long_n,
        })
        print(f"  {tid:12s} {tpl_def['template']:20s} → {len(generated):4d} 句")

    # 写 JSONL
    out_path = os.path.join(OUT_DIR, "generated_sentences.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for g in all_generated:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"\n  总生成: {len(all_generated)} 句 → {out_path}")

    # 写抽样 TSV
    tsv_path = os.path.join(OUT_DIR, "sample_review.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("模板ID\t模板\t类别\t拼接角色\t温州话\t普通话\t字数\t填充词\t通顺?\t备注\n")
        by_tpl = defaultdict(list)
        for g in all_generated:
            by_tpl[g["template_id"]].append(g)
        for tid in sorted(by_tpl):
            sents = by_tpl[tid]
            samples = random.sample(sents, min(5, len(sents)))
            for s in samples:
                fills = "|".join(fw["wz_word"] for fw in s["filled_words"])
                f.write(f"{s['template_id']}\t{s['template']}\t{s['category']}\t"
                        f"{s['concat_role']}\t{s['wz_sentence']}\t{s['zh_sentence']}\t"
                        f"{s['char_len']}\t{fills}\t\t\n")
    print(f"  抽样 TSV: {tsv_path}")

    # 统计
    cat_dist = Counter(g["category"] for g in all_generated)
    role_dist = Counter(g["concat_role"] for g in all_generated)
    bucket_dist = Counter(g["length_bucket"] for g in all_generated)

    summary = {
        "curated_templates": len(curated_records),
        "viable_templates": sum(1 for c in curated_records if c["viable"]),
        "total_generated": len(all_generated),
        "length_buckets": dict(bucket_dist),
        "category_distribution": dict(cat_dist),
        "concat_role_distribution": dict(role_dist),
        "per_template": per_tpl_stats,
    }

    with open(os.path.join(OUT_DIR, "generation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n===== 汇总 =====")
    print(f"  可用模板: {summary['viable_templates']}/{summary['curated_templates']}")
    print(f"  总生成:   {summary['total_generated']}")
    print(f"  长度:     {dict(bucket_dist)}")
    print(f"  类别:     {dict(cat_dist)}")
    print(f"  拼接角色: {dict(role_dist)}")

    # 抽样展示
    print(f"\n===== 随机抽样 =====")
    for cat in sorted(cat_dist):
        sents = [g for g in all_generated if g["category"] == cat]
        samples = random.sample(sents, min(2, len(sents)))
        print(f"\n  [{cat}]")
        for s in samples:
            fills = ", ".join(f"{fw['wz_word']}({fw['mandarin']})" for fw in s["filled_words"])
            print(f"    温: {s['wz_sentence']}")
            print(f"    普: {s['zh_sentence']}")
            print(f"    填: {fills}")


if __name__ == "__main__":
    main()
