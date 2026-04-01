#!/usr/bin/env python3
"""
Few-shot dictionary-grounded long sentence generation pilot.

Strategy:
  1. Cluster short WZ example sentences by scene/headword proximity
  2. Select 3-5 examples + 2-3 target WZ words per task
  3. Ask LLM to generate 20-30 char sentences using those words
  4. Validate with rules: known WZ words present, length, no pure-Mandarin markers

Usage:
  python scripts/generate_long_sentences_fewshot_pilot.py --tasks 20 --provider deepseek
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---- project imports ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from build_controlled_generation_assets import ROOT
from generate_controlled_sentences import build_client
from wz_pipeline.grammar_guardrails import GRAMMAR_PROMPT_RULES, GRAMMAR_USER_RULES, grammar_validation_reasons
from wz_pipeline.grammar_spec import GRAMMAR_SPEC_PATH, relevant_spec_labels
from wz_pipeline.source_surface_guardrails import detect_unsupported_surface_terms

# ---- paths ----
CLEANED_RECORDS = ROOT / "data" / "cleaned" / "cleaned_records_primary.jsonl"
EXTRACTED_SHORT = ROOT / "data" / "extracted_training_sentences" / "short_8_20.jsonl"
EXTRACTED_LONG = ROOT / "data" / "extracted_training_sentences" / "long_20_30.jsonl"
OUT_DIR = ROOT / "data" / "generated_long_sentences" / "fewshot_pilot"
FEWSHOT_CORE_POLICY = ROOT / "data" / "controlled_generation" / "assets" / "fewshot_core_policy.json"

# ---- cleaning ----
PAREN_RE = re.compile(r"[（(][^）)]{1,4}[）)]")


def clean_wz(text: str) -> str:
    text = PAREN_RE.sub("", text)
    return re.sub(r"[\s\u3000]+", "", text).strip()


def load_core_policy() -> dict[str, Any]:
    if not FEWSHOT_CORE_POLICY.exists():
        return {"global": {}, "scene_policies": {}}
    return json.loads(FEWSHOT_CORE_POLICY.read_text(encoding="utf-8"))


CORE_POLICY = load_core_policy()


def global_deny_terms() -> set[str]:
    return set(CORE_POLICY.get("global", {}).get("deny_terms", []))


def scene_deny_terms(scene_id: str) -> set[str]:
    return set(CORE_POLICY.get("scene_policies", {}).get(scene_id, {}).get("deny_terms", []))


# ---- data loading ----

def load_wz_word_set() -> set[str]:
    """Load all known WZ words (2-4 chars) for validation."""
    words = set()
    with open(CLEANED_RECORDS, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            w = row.get("wz_word_train", "")
            if w and 2 <= len(w) <= 4:
                words.add(w)
    return words


def load_examples_by_scene() -> dict[str, list[dict]]:
    """Load short example sentences grouped by scene."""
    # First build a scene map from skeleton templates (they have scene tags)
    scene_map: dict[str, str] = {}  # wz_sentence -> scene_id
    sk_path = ROOT / "data" / "controlled_generation" / "skeletons" / "skeleton_templates.jsonl"
    if sk_path.exists():
        with open(sk_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                wz = row.get("source_wz_sentence", "")
                scene = row.get("scene_id", "")
                if wz and scene:
                    scene_map[clean_wz(wz)] = scene

    by_scene: dict[str, list[dict]] = defaultdict(list)
    with open(EXTRACTED_SHORT, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            wz = row["wz_sentence"]
            scene = scene_map.get(wz, "daily_chat")
            row["scene_id"] = scene
            by_scene[scene].append(row)

    return dict(by_scene)


def load_words_by_scene() -> dict[str, list[dict]]:
    """Load WZ words with definitions, grouped by scene (from skeleton data)."""
    # Use replaceable_lexicon which has scene tags
    lex_path = ROOT / "data" / "controlled_generation" / "assets" / "replaceable_lexicon.jsonl"
    by_scene: dict[str, list[dict]] = defaultdict(list)

    if lex_path.exists():
        with open(lex_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                w = row.get("wz_word", "")
                d = row.get("mandarin_headword", "") or row.get("definition", "")
                scene = row.get("primary_scene_id", "daily_chat")
                if w and 2 <= len(w) <= 4 and d:
                    by_scene[scene].append({"wz_word": w, "definition": d[:40], "scene_id": scene})

    # Fallback: load from cleaned records (no scene, use "daily_chat")
    if not by_scene:
        with open(CLEANED_RECORDS, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                w = row.get("wz_word_train", "")
                d = row.get("definition_norm", "")
                if w and 2 <= len(w) <= 4 and d:
                    by_scene["daily_chat"].append({"wz_word": w, "definition": d[:40], "scene_id": "daily_chat"})

    return dict(by_scene)


# ---- task building ----

SYSTEM_PROMPT = """你是温州话句子生成器。你的任务是根据提供的温州话词典例句和词汇，生成自然的温州话长句。

规则：
1. 每句必须 20-30 个字（含标点）
2. 必须使用给定的"必用词汇"中至少 2 个
3. 句子要像温州人日常说话的口语，不是书面语
4. 保持例句中展示的方言特征，但不要为了像方言而乱拼功能词
5. 不要写成普通话
6. 每句要有完整的语义，适合语音训练朗读
7. 生成 5 句，每句独立
8. 不要自己发明新的两字到四字词；除给定词和参考例句能支持的说法外，拿不准就换成来源里已有的稳妥表达
""" + "\n\n" + GRAMMAR_PROMPT_RULES + "\n\n只输出 JSON：\n" + '{"sentences": [{"wz": "温州话句子", "zh": "普通话翻译"}]}'


def build_task(
    examples: list[dict],
    target_words: list[dict],
    scene_id: str,
    task_id: str,
) -> dict:
    """Build one generation task."""
    banned_terms = sorted(global_deny_terms() | scene_deny_terms(scene_id))
    example_block = "\n".join(
        f"  {i+1}. 温州话：{ex['wz_sentence']}\n     普通话：{ex['zh_sentence']}"
        for i, ex in enumerate(examples)
    )
    word_block = "\n".join(
        f"  - {w['wz_word']}（{w['definition']}）"
        for w in target_words
    )
    banned_block = "\n".join(f"  - {term}" for term in banned_terms) or "  - 无"

    user_msg = f"""场景：{scene_id}

参考温州话例句（注意学习其中的方言风格和用词习惯）：
{example_block}

必用词汇（每句至少使用其中 2 个）：
{word_block}

禁止词汇（即使参考例句里出现，也绝对不要复用）：
{banned_block}

请额外遵守这些温州话语法约束：
{GRAMMAR_USER_RULES}

如果一句话里需要额外内容词，优先复用参考例句和来源里已出现过的表达，不要自己造新词。

请生成 5 个 20-30 字的温州话口语长句。"""

    return {
        "task_id": task_id,
        "scene_id": scene_id,
        "example_sentences": [ex["wz_sentence"] for ex in examples],
        "target_words": [w["wz_word"] for w in target_words],
        "target_word_defs": {w["wz_word"]: w["definition"] for w in target_words},
        "banned_terms": banned_terms,
        "prompt_system": SYSTEM_PROMPT,
        "prompt_user": user_msg,
    }


def create_tasks(
    examples_by_scene: dict[str, list[dict]],
    words_by_scene: dict[str, list[dict]],
    num_tasks: int,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    tasks = []
    scenes = [s for s in examples_by_scene if len(examples_by_scene[s]) >= 3]

    for i in range(num_tasks):
        scene = rng.choice(scenes)
        exs = rng.sample(examples_by_scene[scene], min(4, len(examples_by_scene[scene])))

        # Target words: prefer same scene, fallback to any
        scene_words = words_by_scene.get(scene, [])
        if len(scene_words) >= 3:
            tgt_words = rng.sample(scene_words, 3)
        else:
            all_words = [w for ws in words_by_scene.values() for w in ws]
            tgt_words = rng.sample(all_words, 3)

        tid = f"fewshot_long_{hashlib.md5(f'{i}_{seed}'.encode()).hexdigest()[:8]}"
        tasks.append(build_task(exs, tgt_words, scene, tid))

    return tasks


# ---- generation ----

def request_generation(client: Any, model: str, task: dict) -> list[dict]:
    """Call LLM and parse response."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": task["prompt_system"]},
                {"role": "user", "content": task["prompt_user"]},
            ],
            temperature=0.6,
            response_format={"type": "json_object"},
            timeout=120.0,
        )
        text = resp.choices[0].message.content or ""
        data = json.loads(text)
        sentences = data.get("sentences", [])
        if isinstance(sentences, list):
            return sentences
        return []
    except Exception as e:
        print(f"  [ERROR] {e}")
        return []


# ---- validation ----

# Common Mandarin-only patterns that shouldn't appear in Wenzhou dialect
MANDARIN_MARKERS = [
    "的话", "然后", "但是", "因为", "所以", "而且", "虽然", "如果",
    "可是", "或者", "不过", "已经", "正在", "刚才",
    "什么", "怎么", "为什么", "哪里", "这里", "那里",
    "他们", "她们", "我们", "你们",
    "非常", "特别", "真的是",
]


def validate_sentence(
    wz: str,
    zh: str,
    known_words: set[str],
    target_words: list[str],
    existing_sentences: set[str],
    banned_terms: list[str],
) -> dict:
    """Rule-based validation. Returns {pass, reasons}."""
    reasons = []
    wz_clean = clean_wz(wz)
    char_len = len(wz_clean)
    grammar_reasons = grammar_validation_reasons(wz_clean)
    unsupported_surface_terms = detect_unsupported_surface_terms(
        wz_clean,
        protected_terms=target_words,
    )

    # Length check
    if char_len < 20:
        reasons.append(f"too_short:{char_len}")
    elif char_len > 30:
        reasons.append(f"too_long:{char_len}")

    # Known WZ word presence
    found_known = [w for w in known_words if w in wz_clean]
    found_target = [w for w in target_words if w in wz_clean]
    banned_term_hits = sorted({w for w in banned_terms if w in wz_clean})
    if len(found_target) < 2:
        reasons.append(f"target_words_missing:{len(found_target)}/2")
    if len(found_known) < 2:
        reasons.append(f"low_wz_word_coverage:{len(found_known)}")
    if banned_term_hits:
        reasons.append(f"banned_terms:{','.join(banned_term_hits)}")
    for finding in unsupported_surface_terms:
        reasons.append(f"source_surface_missing:{finding['surface']}")

    # Mandarin marker check
    mandarin_hits = [m for m in MANDARIN_MARKERS if m in wz_clean]
    if len(mandarin_hits) >= 2:
        reasons.append(f"mandarin_markers:{','.join(mandarin_hits[:3])}")

    # Duplicate check
    if wz_clean in existing_sentences:
        reasons.append("duplicate")

    # Has translation
    if not zh or not zh.strip():
        reasons.append("no_translation")
    for reason in grammar_reasons:
        reasons.append(f"grammar:{reason}")

    return {
        "pass": len(reasons) == 0,
        "reasons": reasons,
        "char_len": char_len,
        "found_known_words": len(found_known),
        "found_target_words": len(found_target),
        "banned_term_hits": banned_term_hits,
        "mandarin_markers": mandarin_hits,
        "grammar_reasons": grammar_reasons,
        "grammar_spec_path": str(GRAMMAR_SPEC_PATH),
        "grammar_spec_sections": relevant_spec_labels(wz_clean, grammar_reasons),
        "unsupported_surface_terms": unsupported_surface_terms,
    }


# ---- main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--provider", choices=["primary", "secondary", "qwen", "deepseek"], default="deepseek")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    known_words = load_wz_word_set()
    examples_by_scene = load_examples_by_scene()
    words_by_scene = load_words_by_scene()

    print(f"Known WZ words: {len(known_words)}")
    print(f"Scenes with examples: {list(examples_by_scene.keys())}")
    print(f"Scenes with words: {list(words_by_scene.keys())}")

    # Load existing sentences for dedup
    existing = set()
    for path in [EXTRACTED_SHORT, EXTRACTED_LONG]:
        if path.exists():
            with open(path) as f:
                for line in f:
                    existing.add(json.loads(line)["wz_sentence"])

    print(f"Existing sentences for dedup: {len(existing)}")
    print(f"\nCreating {args.tasks} tasks...")
    tasks = create_tasks(examples_by_scene, words_by_scene, args.tasks, args.seed)

    client, model, _provider_name = build_client(args.provider)
    print(f"Provider: {args.provider}, Model: {model}")

    all_results = []
    pass_count = 0
    fail_count = 0

    for i, task in enumerate(tasks):
        print(f"\n--- Task {i+1}/{len(tasks)}: {task['task_id']} ({task['scene_id']}) ---")
        print(f"  Target words: {task['target_words']}")

        sentences = request_generation(client, model, task)
        print(f"  Raw output: {len(sentences)} sentences")

        for sent in sentences:
            wz = sent.get("wz", "")
            zh = sent.get("zh", "")
            if not wz:
                continue

            val = validate_sentence(
                wz,
                zh,
                known_words,
                task["target_words"],
                existing,
                task.get("banned_terms", []),
            )
            result = {
                "task_id": task["task_id"],
                "scene_id": task["scene_id"],
                "wz_sentence": clean_wz(wz),
                "zh_sentence": zh,
                "target_words": task["target_words"],
                "validation": val,
            }
            all_results.append(result)
            existing.add(clean_wz(wz))  # prevent dups across tasks

            if val["pass"]:
                pass_count += 1
                print(f"  PASS [{val['char_len']}字]: {clean_wz(wz)}")
            else:
                fail_count += 1
                print(f"  FAIL {val['reasons']}: {clean_wz(wz)[:40]}...")

        if args.sleep > 0 and i < len(tasks) - 1:
            time.sleep(args.sleep)

    # Write results
    results_path = OUT_DIR / "pilot_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    passed = [r for r in all_results if r["validation"]["pass"]]
    passed_path = OUT_DIR / "pilot_passed.jsonl"
    with open(passed_path, "w", encoding="utf-8") as f:
        for r in passed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "tasks": len(tasks),
        "raw_sentences": len(all_results),
        "passed": pass_count,
        "failed": fail_count,
        "pass_rate": round(pass_count / max(1, len(all_results)), 4),
        "provider": args.provider,
    }
    summary_path = OUT_DIR / "pilot_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"PILOT RESULTS")
    print(f"Tasks: {len(tasks)}")
    print(f"Raw sentences: {len(all_results)}")
    print(f"Passed: {pass_count} ({summary['pass_rate']*100:.1f}%)")
    print(f"Failed: {fail_count}")
    print(f"Output: {results_path}")


if __name__ == "__main__":
    main()
