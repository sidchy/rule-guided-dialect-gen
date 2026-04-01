#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from generate_controlled_sentences import build_client
from openai import OpenAI

from build_controlled_generation_assets import ROOT
from build_dense_replacement_pilot import (
    enumerate_plans,
    pool_jaccard,
    rebuild_task_prompt_bundle,
)


DEFAULT_INPUT = (
    ROOT
    / "data"
    / "controlled_generation"
    / "tasks"
    / "dense_replacement_pilot_tasks.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "controlled_generation"
    / "tasks"
    / "dense_replacement_pilot_tasks_pruned.jsonl"
)
DEFAULT_SUMMARY = (
    ROOT
    / "data"
    / "controlled_generation"
    / "reports"
    / "dense_slot_pool_review_summary.json"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def slot_rank(task: dict[str, Any], slot_id: str) -> tuple[int, int, float, str]:
    slot_lookup = {slot.get("slot_id"): slot for slot in task.get("slots") or []}
    slot = slot_lookup.get(slot_id) or {}
    priority_order = {"high": 2, "medium": 1, "low": 0}
    return (
        priority_order.get(slot.get("dense_priority") or "low", 0),
        len((task.get("slot_candidate_pools") or {}).get(slot_id, [])),
        float(slot.get("confidence", 0.0)),
        slot_id,
    )


def build_prompt(task: dict[str, Any], slot_id: str) -> tuple[str, str]:
    system = (
        "你是温州话高密度替换任务里的槽位兼容性审校员。"
        "你不改写句子，不新增词。"
        "你只判断当前槽位池里的每个候选，是否适合放进这个槽位。"
        "必须遵守："
        "1. 只能在输入候选里做 keep 或 drop；"
        "2. 优先保留和原槽位普通话释义最贴近、且和左右固定块拼接自然的词；"
        "3. 丢掉固定表达核心词、词类不对、只是 scene 对但 slot 不对、放进去明显别扭的词；"
        "4. 每个槽位最多保留 4 个词；"
        "5. 如果靠谱候选不足 2 个，要明确说明。"
        "只输出 JSON object。"
    )
    signature = (task.get("slot_signatures") or {}).get(slot_id) or {}
    payload = {
        "scene_id": task.get("scene_id"),
        "scene_label": task.get("scene_label"),
        "source_wz_sentence": task.get("source_wz_sentence"),
        "source_zh_sentence": task.get("source_zh_sentence"),
        "skeleton_template": task.get("skeleton_template"),
        "slot_signature": signature,
        "slot_candidate_pool": (task.get("slot_candidate_pools") or {}).get(slot_id, []),
        "output_schema": {
            "slot_id": slot_id,
            "decisions": [
                {
                    "term": "input term only",
                    "decision": "keep or drop",
                    "reason": "short Chinese reason under 16 chars",
                }
            ],
            "slot_note": "short Chinese note under 30 chars",
        },
    }
    user = "根据下面 JSON 判断该槽位的候选兼容性，只返回 JSON：\n" + json.dumps(
        payload, ensure_ascii=False, indent=2
    )
    return system, user


def request_review(
    client: OpenAI,
    model: str,
    task: dict[str, Any],
    slot_id: str,
    timeout: float | None,
) -> dict[str, Any]:
    system, user = build_prompt(task, slot_id)
    request_client = client.with_options(timeout=timeout) if timeout and timeout > 0 else client
    response = request_client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def normalize_decisions(value: Any, allowed_terms: set[str]) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term") or "").strip()
        decision = str(item.get("decision") or "").strip().lower()
        reason = str(item.get("reason") or "").strip()[:16]
        if term not in allowed_terms or term in seen or decision not in {"keep", "drop"}:
            continue
        normalized.append({"term": term, "decision": decision, "reason": reason})
        seen.add(term)
    return normalized


def fallback_review_result(task: dict[str, Any], slot_id: str, error_message: str) -> dict[str, Any]:
    pool = (task.get("slot_candidate_pools") or {}).get(slot_id, [])
    decisions = []
    for item in pool[:4]:
        term = str(item.get("term") or "").strip()
        if term:
            decisions.append({"term": term, "decision": "keep", "reason": "review_failed"})
    return {
        "slot_id": slot_id,
        "decisions": decisions,
        "slot_note": error_message[:30],
        "request_failed": True,
        "request_error": error_message,
    }


def prune_task(
    task: dict[str, Any],
    slot_review_results: dict[str, dict[str, Any]],
    max_plans_per_skeleton: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    slot_candidate_pools = task.get("slot_candidate_pools") or {}
    pruned_pools: dict[str, list[dict[str, Any]]] = {}
    slot_pool_sizes_before: dict[str, int] = {}
    slot_pool_sizes_after: dict[str, int] = {}
    slot_pool_review_notes: dict[str, Any] = {}
    dropped_by_reason = Counter()
    locked_slot_reasons = dict(task.get("locked_slot_reasons") or {})
    original_eligible_slot_ids = list(task.get("eligible_slot_ids") or [])

    for slot_id in original_eligible_slot_ids:
        pool = slot_candidate_pools.get(slot_id, [])
        allowed_terms = {item.get("term") for item in pool if item.get("term")}
        decisions = normalize_decisions(
            (slot_review_results.get(slot_id) or {}).get("decisions"),
            allowed_terms,
        )
        keep_terms = [item["term"] for item in decisions if item["decision"] == "keep"][:4]
        drop_notes = {
            item["term"]: item["reason"]
            for item in decisions
            if item["decision"] == "drop" and item["reason"]
        }
        if len(keep_terms) < 2:
            locked_slot_reasons[slot_id] = "insufficient_candidates"
            dropped_by_reason["insufficient_candidates"] += 1
            continue
        pruned_pool = [item for item in pool if item.get("term") in set(keep_terms)]
        if len(pruned_pool) > 4:
            pruned_pool = pruned_pool[:4]
        pruned_pools[slot_id] = pruned_pool
        slot_pool_sizes_before[slot_id] = len(pool)
        slot_pool_sizes_after[slot_id] = len(pruned_pool)
        slot_pool_review_notes[slot_id] = {
            "slot_note": str((slot_review_results.get(slot_id) or {}).get("slot_note") or "").strip()[:30],
            "drop_notes": drop_notes,
        }
        for reason in drop_notes.values():
            dropped_by_reason[reason] += 1

    eligible_slot_ids = [slot_id for slot_id in original_eligible_slot_ids if slot_id in pruned_pools]
    tasks_downgraded_after_review = int(len(eligible_slot_ids) < len(original_eligible_slot_ids))

    collision_pairs: list[dict[str, Any]] = []
    while True:
        pair_to_lock: tuple[str, str] | None = None
        pair_score = 0.0
        for left_index, left_slot_id in enumerate(eligible_slot_ids):
            for right_slot_id in eligible_slot_ids[left_index + 1 :]:
                score = pool_jaccard(
                    pruned_pools.get(left_slot_id, []),
                    pruned_pools.get(right_slot_id, []),
                )
                if score > 0.34 and score >= pair_score:
                    pair_score = score
                    pair_to_lock = (left_slot_id, right_slot_id)
        if not pair_to_lock:
            break
        weaker = min(pair_to_lock, key=lambda slot_id: slot_rank(task, slot_id))
        collision_pairs.append({"slot_ids": list(pair_to_lock), "jaccard": round(pair_score, 4)})
        locked_slot_reasons[weaker] = "pool_collision"
        pruned_pools.pop(weaker, None)
        eligible_slot_ids = [slot_id for slot_id in eligible_slot_ids if slot_id != weaker]
        tasks_downgraded_after_review = 1

    if len(eligible_slot_ids) < 2:
        return None, {
            "dropped_by_reason": dict(dropped_by_reason),
            "tasks_downgraded_after_review": tasks_downgraded_after_review,
            "insufficient_dense_support": True,
            "collision_pairs": collision_pairs,
            "locked_slot_reasons": locked_slot_reasons,
        }

    replacement_plans = enumerate_plans(
        eligible_slot_ids,
        pruned_pools,
        max_plans=max_plans_per_skeleton,
    )
    if len(replacement_plans) < 6:
        return None, {
            "dropped_by_reason": dict(dropped_by_reason),
            "tasks_downgraded_after_review": 1,
            "insufficient_dense_support": True,
            "collision_pairs": collision_pairs,
            "locked_slot_reasons": locked_slot_reasons,
        }

    merged = dict(task)
    merged["slot_candidate_pools_original"] = slot_candidate_pools
    merged["slot_candidate_pools"] = pruned_pools
    merged["slot_pool_sizes_before"] = slot_pool_sizes_before
    merged["slot_pool_sizes_after"] = slot_pool_sizes_after
    merged["slot_pool_review_notes"] = slot_pool_review_notes
    merged["eligible_slot_ids"] = eligible_slot_ids
    merged["required_slot_replacements"] = eligible_slot_ids
    merged["slot_signatures"] = {
        slot_id: signature
        for slot_id, signature in (task.get("slot_signatures") or {}).items()
        if slot_id in set(eligible_slot_ids)
    }
    merged["locked_slot_reasons"] = locked_slot_reasons
    merged["replacement_plans"] = replacement_plans
    merged["slot_target_pool_sizes"] = {
        slot_id: len(pruned_pools.get(slot_id, []))
        for slot_id in eligible_slot_ids
    }
    merged["post_review_collision_pairs"] = collision_pairs
    merged["task_build_decision"] = "ready_for_generation"
    merged["task_build_issues"] = sorted(
        set((merged.get("task_build_issues") or []) + list(locked_slot_reasons.values()))
    )
    rebuild_task_prompt_bundle(merged)
    return merged, {
        "dropped_by_reason": dict(dropped_by_reason),
        "tasks_downgraded_after_review": tasks_downgraded_after_review,
        "insufficient_dense_support": False,
        "collision_pairs": collision_pairs,
        "locked_slot_reasons": locked_slot_reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use an LLM to prune dense replacement slot candidate pools."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--provider", choices=["primary", "secondary", "qwen", "deepseek"], default="deepseek")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-plans-per-skeleton", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    tasks = load_jsonl(args.input)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    client, model, provider = build_client(args.provider)
    reviewed_tasks: list[dict[str, Any]] = []
    dropped_by_reason = Counter()
    tasks_downgraded_after_review = 0
    insufficient_dense_support_count = 0
    failed_slot_reviews = 0

    for task_index, task in enumerate(tasks, start=1):
        print(f"[{task_index}/{len(tasks)}] reviewing {task.get('task_id')}", flush=True)
        slot_review_results: dict[str, dict[str, Any]] = {}
        for slot_id in task.get("eligible_slot_ids") or []:
            try:
                result = request_review(
                    client,
                    model,
                    task,
                    slot_id,
                    timeout=args.timeout,
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                failed_slot_reviews += 1
                error_message = f"{type(exc).__name__}: {exc}"
                print(f"slot review failed {task.get('task_id')}::{slot_id}: {error_message}", flush=True)
                result = fallback_review_result(task, slot_id, error_message)
            slot_review_results[slot_id] = result

        pruned_task, meta = prune_task(
            task,
            slot_review_results=slot_review_results,
            max_plans_per_skeleton=args.max_plans_per_skeleton,
        )
        dropped_by_reason.update(meta.get("dropped_by_reason") or {})
        tasks_downgraded_after_review += int(meta.get("tasks_downgraded_after_review") or 0)
        if meta.get("insufficient_dense_support"):
            insufficient_dense_support_count += 1
            continue
        if pruned_task is not None:
            reviewed_tasks.append(pruned_task)

    write_jsonl(args.output, reviewed_tasks)
    write_json(
        args.summary,
        {
            "input_tasks": len(tasks),
            "reviewed_tasks": len(reviewed_tasks),
            "scene_counts": dict(Counter(task.get("scene_id") for task in reviewed_tasks)),
            "slot_pool_sizes_before": dict(
                Counter(
                    size
                    for task in reviewed_tasks
                    for size in (task.get("slot_pool_sizes_before") or {}).values()
                )
            ),
            "slot_pool_sizes_after": dict(
                Counter(
                    size
                    for task in reviewed_tasks
                    for size in (task.get("slot_pool_sizes_after") or {}).values()
                )
            ),
            "replacement_plan_count_distribution": dict(
                Counter(len(task.get("replacement_plans") or []) for task in reviewed_tasks)
            ),
            "dropped_by_reason": dict(dropped_by_reason),
            "tasks_downgraded_after_review": tasks_downgraded_after_review,
            "insufficient_dense_support_count": insufficient_dense_support_count,
            "failed_slot_reviews": failed_slot_reviews,
            "provider": provider,
            "model": model,
            "output_path": str(args.output),
        },
    )

    print(f"Input tasks: {len(tasks)}")
    print(f"Reviewed tasks: {len(reviewed_tasks)}")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
