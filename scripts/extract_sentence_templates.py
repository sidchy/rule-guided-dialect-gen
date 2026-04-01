#!/usr/bin/env python3
"""
轨道 A · 第 1 步：从词典例句中自动提取高频句式模板。

核心策略（反转法）：
  不是"找出实词替换掉" → 而是"找出虚词保留，其余替换为 [X]"。
  温州话虚词/语法词是有限集合，这样骨架更稳定。

三层提取：
  层 1：虚词骨架模板 — 识别句中虚词位置，其余合并为 [X]
  层 2：语法锚点子模式 — 围绕高频语法结构的正则匹配
  层 3：双子句组合模式 — 两个子句模板用连接词拼成完整句

输出:
  data/sentence_templates/extracted_templates.jsonl
  data/sentence_templates/extraction_stats.json
  data/sentence_templates/top_templates.tsv
"""

import hashlib
import json
import os
import re
from collections import Counter, defaultdict

# ---------- 路径 ----------
LEXICON_PATH = "data/controlled_generation/assets/replaceable_lexicon.jsonl"
SENTENCES_DIR = "data/extracted_training_sentences"
OUT_DIR = "data/sentence_templates"

# ---------- 温州话虚词/语法词表 ----------
# 按长度降序排列以支持最长匹配
# 这些词在模板中保留，其余替换为 [X]
GRAMMAR_WORDS = [
    # 4字
    "痴痴疑疑", "阿乜名堂",
    # 3字
    "起码货", "不得不", "不晓得", "晓不得", "走来走去",
    # 2字 — 代词
    "我俫", "你俫", "渠俫", "人家", "自家", "大家",
    "该个", "恁个", "该俫", "许俫", "阿乜", "訾那",
    # 2字 — 副词/连词
    "能界", "称称", "早早", "总归", "其实", "不过",
    "要是", "着是", "虽然", "所以", "到底", "从来",
    "不得", "不用", "不会", "不好", "不是",
    # 2字 — 助词/补语
    "走来", "走去", "起来", "下去", "出来", "回来",
    "沃是", "只当", "好比", "到底",
    # 1字 — 代词
    "我", "你", "渠",
    # 1字 — 指示/疑问
    "该", "恁",
    # 1字 — 否定
    "不", "冇", "未", "覅", "𧟰",
    # 1字 — 助词
    "个", "爻", "罢", "着", "牢", "底", "起", "过",
    "搭", "显", "嘎", "喏", "咩", "啊", "哦",
    # 1字 — 副词/介词
    "沃", "还", "就", "先", "再", "真", "阿", "亦",
    "匄", "逮", "伉", "宿", "有", "会", "是",
    # 1字 — 趋向
    "到", "出", "落",
]

# 高频语法词的 set（用于快速查询）
GRAMMAR_SET = set(GRAMMAR_WORDS)
# 按长度分桶（降序匹配用）
GRAMMAR_BY_LEN = defaultdict(set)
for w in GRAMMAR_WORDS:
    GRAMMAR_BY_LEN[len(w)].add(w)
GRAMMAR_MAX_LEN = max(len(w) for w in GRAMMAR_WORDS)

# 子句分割
CLAUSE_SPLIT_RE = re.compile(r"[，。！？；：、]")
# 连接标点（保留在模板中）
CONNECTORS = {"，", "；", "、"}


def template_id(tpl):
    h = hashlib.md5(tpl.encode()).hexdigest()[:10]
    return f"tpl_{h}"


def load_sentences():
    sentences = []
    for fname in ["short_8_20.jsonl", "long_20_30.jsonl"]:
        path = os.path.join(SENTENCES_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                sentences.append(json.loads(line))
    return sentences


# ---------- 层 1：虚词骨架提取 ----------

def extract_grammar_skeleton(text):
    """
    扫描文本，识别虚词位置 → 保留虚词，非虚词部分替换为 [X]。
    返回 (skeleton_str, slot_fills_list)。

    例: "该个人做事干恁痴下痴下" → "该个[X]恁[X]"
    """
    tokens = []  # list of (type, text): 'gram' or 'content'
    i = 0
    n = len(text)

    while i < n:
        # 跳过标点
        if text[i] in "，。！？；：、""''（）":
            tokens.append(("punct", text[i]))
            i += 1
            continue

        # 尝试最长匹配语法词
        matched = False
        for length in range(min(GRAMMAR_MAX_LEN, n - i), 0, -1):
            substr = text[i:i + length]
            if substr in GRAMMAR_BY_LEN.get(length, set()):
                tokens.append(("gram", substr))
                i += length
                matched = True
                break

        if not matched:
            # 非语法词字符 → content
            tokens.append(("content", text[i]))
            i += 1

    # 合并连续 content tokens
    merged = []
    for typ, txt in tokens:
        if typ == "content" and merged and merged[-1][0] == "content":
            merged[-1] = ("content", merged[-1][1] + txt)
        else:
            merged.append((typ, txt))

    # 构建骨架
    skeleton_parts = []
    fills = []
    for typ, txt in merged:
        if typ == "gram":
            skeleton_parts.append(txt)
        elif typ == "punct":
            skeleton_parts.append(txt)
        else:  # content
            skeleton_parts.append("[X]")
            fills.append(txt)

    skeleton = "".join(skeleton_parts)

    # 合并相邻 [X]
    while "[X][X]" in skeleton:
        skeleton = skeleton.replace("[X][X]", "[X]")

    return skeleton, fills


def score_template(tpl, freq):
    """
    评分函数：综合考虑语法丰富度、槽位数、频率。
    用于排序和筛选。
    """
    slot_count = tpl.count("[X]")
    # 骨架中的语法词数量
    skel_no_slots = tpl.replace("[X]", " ")
    gram_chars = len(re.sub(r"[\s，。！？；：、\[\]X]", "", skel_no_slots))

    # 理想模板：1-2 个槽位，骨架 ≥ 3 字，频率高
    if slot_count < 1 or slot_count > 3:
        return 0
    if gram_chars < 2:
        return 0

    score = freq * (gram_chars / max(slot_count, 1))
    return score


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("加载训练句...")
    sentences = load_sentences()
    print(f"  句子数: {len(sentences)}")

    # ===== 层 1：全句虚词骨架 =====
    print("\n===== 层 1：全句虚词骨架 =====")
    full_skel_examples = defaultdict(list)

    for sent in sentences:
        wz = sent["wz_sentence"].rstrip("。！？")
        skel, fills = extract_grammar_skeleton(wz)
        if "[X]" not in skel:
            continue
        full_skel_examples[skel].append({
            "wz": sent["wz_sentence"],
            "zh": sent.get("zh_sentence", ""),
            "fills": fills,
        })

    full_freq_ge2 = sum(1 for exs in full_skel_examples.values() if len(exs) >= 2)
    print(f"  唯一骨架: {len(full_skel_examples)}, freq≥2: {full_freq_ge2}")

    # ===== 层 2：子句级骨架 =====
    print("\n===== 层 2：子句级骨架 =====")
    clause_skel_examples = defaultdict(list)

    for sent in sentences:
        wz = sent["wz_sentence"].rstrip("。！？")
        clauses = CLAUSE_SPLIT_RE.split(wz)
        for clause in clauses:
            clause = clause.strip()
            if len(clause) < 4:
                continue
            skel, fills = extract_grammar_skeleton(clause)
            if "[X]" not in skel:
                continue
            clause_skel_examples[skel].append({
                "wz_clause": clause,
                "wz": sent["wz_sentence"],
                "zh": sent.get("zh_sentence", ""),
                "fills": fills,
            })

    clause_freq_ge3 = sum(1 for exs in clause_skel_examples.values() if len(exs) >= 3)
    print(f"  唯一子句骨架: {len(clause_skel_examples)}, freq≥3: {clause_freq_ge3}")

    # ===== 合并 + 打分 =====
    print("\n===== 合并打分 =====")

    all_templates = {}  # tpl_str -> {source, freq, examples}

    # 先加子句模板（频率 ≥ 3）
    for tpl, exs in clause_skel_examples.items():
        freq = len(exs)
        if freq < 3:
            continue
        sc = score_template(tpl, freq)
        if sc <= 0:
            continue
        all_templates[tpl] = {
            "source": "clause",
            "frequency": freq,
            "score": sc,
            "examples": exs[:5],
        }

    # 再加全句模板（频率 ≥ 2，可能覆盖新模式）
    for tpl, exs in full_skel_examples.items():
        freq = len(exs)
        if freq < 2:
            continue
        sc = score_template(tpl, freq)
        if sc <= 0:
            continue
        if tpl in all_templates:
            # 已有子句版本且频率更高 → 保留子句版本
            if all_templates[tpl]["frequency"] >= freq:
                all_templates[tpl]["source"] = "clause+full"
                continue
        all_templates[tpl] = {
            "source": "full",
            "frequency": freq,
            "score": sc,
            "examples": [{"wz": e["wz"], "zh": e["zh"], "fills": e["fills"]} for e in exs[:5]],
        }

    # 排序
    results = []
    for tpl, info in all_templates.items():
        slot_count = tpl.count("[X]")
        skel_chars = len(re.sub(r"[\s\[\]X，。！？；：、]", "", tpl))
        results.append({
            "template_id": template_id(tpl),
            "template": tpl,
            "source": info["source"],
            "slot_count": slot_count,
            "skeleton_chars": skel_chars,
            "frequency": info["frequency"],
            "score": round(info["score"], 1),
            "example_sentences": info["examples"],
        })

    results.sort(key=lambda x: (-x["score"], -x["frequency"]))

    # 写 JSONL
    out_path = os.path.join(OUT_DIR, "extracted_templates.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  写入: {len(results)} 条模板")

    # 写 TSV（人工筛选用）
    tsv_path = os.path.join(OUT_DIR, "top_templates.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("排名\t来源\t模板\t槽位\t骨架字数\t频率\t评分\t示例1\t示例2\t示例3\n")
        for i, r in enumerate(results[:200], 1):
            exs = r["example_sentences"]
            e1 = exs[0].get("wz_clause", exs[0]["wz"]) if len(exs) > 0 else ""
            e2 = exs[1].get("wz_clause", exs[1]["wz"]) if len(exs) > 1 else ""
            e3 = exs[2].get("wz_clause", exs[2]["wz"]) if len(exs) > 2 else ""
            f.write(f"{i}\t{r['source']}\t{r['template']}\t{r['slot_count']}\t{r['skeleton_chars']}\t{r['frequency']}\t{r['score']}\t{e1}\t{e2}\t{e3}\n")
    print(f"  写入 TSV: {tsv_path}")

    # 统计
    slot_dist = Counter(r["slot_count"] for r in results)
    source_dist = Counter(r["source"] for r in results)
    skel_len_dist = Counter()
    for r in results:
        bucket = "2-3" if r["skeleton_chars"] <= 3 else "4-6" if r["skeleton_chars"] <= 6 else "7+"
        skel_len_dist[bucket] += 1

    stats = {
        "total_sentences": len(sentences),
        "full_unique_skeletons": len(full_skel_examples),
        "clause_unique_skeletons": len(clause_skel_examples),
        "output_templates": len(results),
        "slot_distribution": {str(k): v for k, v in sorted(slot_dist.items())},
        "source_distribution": dict(source_dist),
        "skeleton_length_buckets": dict(skel_len_dist),
        "top_30": [
            {
                "tpl": r["template"],
                "freq": r["frequency"],
                "slots": r["slot_count"],
                "skel": r["skeleton_chars"],
                "score": r["score"],
            }
            for r in results[:30]
        ],
    }

    stats_path = os.path.join(OUT_DIR, "extraction_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 打印
    print(f"\n===== Top 40 模板（按评分） =====")
    for i, r in enumerate(results[:40], 1):
        exs = r["example_sentences"]
        ex1 = exs[0].get("wz_clause", exs[0]["wz"])
        print(f"  {i:2d}. [freq={r['frequency']:3d} skel={r['skeleton_chars']} slots={r['slot_count']}] {r['template']}")
        print(f"      例: {ex1}")

    print(f"\n  总计: {len(results)} 条模板")
    print(f"  槽位分布: {dict(slot_dist)}")
    print(f"  骨架长度: {dict(skel_len_dist)}")


if __name__ == "__main__":
    main()
