#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "controlled_generation" / "tasks" / "generation_tasks_sample.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "controlled_generation" / "candidates" / "generation_candidates_raw.jsonl"


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


def build_client(provider: str) -> tuple[OpenAI, str, str]:
    load_dotenv(ROOT / ".env")
    provider_map = {
        "qwen": {
            "api_key": os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": os.getenv("QWEN_MODEL") or os.getenv("OPENAI_MODEL") or "qwen-plus",
        },
        "deepseek": {
            "api_key": os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY_2"),
            "base_url": os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
            "model": os.getenv("DEEPSEEK_MODEL") or "deepseek-chat",
        },
    }
    if provider == "primary":
        provider = os.getenv("PRIMARY_PROVIDER") or "qwen"
    elif provider == "secondary":
        provider = os.getenv("SECONDARY_PROVIDER") or "deepseek"
    if provider not in provider_map:
        raise RuntimeError(f"Unsupported provider: {provider}")
    config = provider_map[provider]
    api_key = config["api_key"]
    base_url = config["base_url"]
    model = config["model"]
    if not api_key:
        raise RuntimeError(f"No API key found for provider={provider} in .env")
    if not model:
        raise RuntimeError(f"MODEL is required for provider={provider} in .env")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model, provider


def request_generation(
    client: OpenAI,
    model: str,
    task: dict[str, Any],
    temperature: float,
    timeout: float | None,
) -> dict[str, Any]:
    prompt_bundle = task["prompt_bundle"]
    request_client = client.with_options(timeout=timeout) if timeout and timeout > 0 else client
    response = request_client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt_bundle["system_prompt"]},
            {"role": "user", "content": prompt_bundle["user_prompt"]},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def normalize_candidates(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        if isinstance(result.get("sentence"), str):
            return [result]
        if isinstance(result.get("candidate_sentences"), list):
            normalized: list[dict[str, Any]] = []
            for item in result["candidate_sentences"]:
                if isinstance(item, dict) and isinstance(item.get("wz_sentence"), str):
                    normalized.append(
                        {
                            "sentence": item["wz_sentence"],
                            "zh_translation": item.get("zh_translation"),
                            "slot_values": item.get("slot_values") or {},
                            "naturalness_note": item.get("naturalness_note"),
                            "plan_id": item.get("plan_id"),
                        }
                    )
            if normalized:
                return normalized
        candidates = (
            result.get("candidates")
            or result.get("outputs")
            or result.get("sentences")
            or result.get("output")
            or []
        )
    elif isinstance(result, list):
        candidates = result
    else:
        candidates = []

    if isinstance(candidates, dict):
        candidates = [candidates]
    elif isinstance(candidates, str):
        lines = [line.strip(" -•\t") for line in candidates.splitlines() if line.strip()]
        if len(lines) > 1:
            candidates = lines
        elif candidates.strip():
            candidates = [candidates.strip()]
        else:
            candidates = []

    normalized: list[dict[str, Any]] = []
    for item in candidates:
        if isinstance(item, str):
            normalized.append({"sentence": item})
        elif isinstance(item, dict):
            normalized.append(item)
    return normalized


def flatten_candidates(task: dict[str, Any], result: dict[str, Any], model: str, provider: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = normalize_candidates(result)
    slot_source_values = {slot["slot_id"]: slot["surface_wz"] for slot in task["slots"]}
    if not candidates:
        return [
            {
                "task_id": task["task_id"],
                "skeleton_id": task["skeleton_id"],
                "scene_id": task["scene_id"],
                "scene_label": task["scene_label"],
                "candidate_rank": 0,
                "source_wz_sentence": task["source_wz_sentence"],
                "source_zh_sentence": task["source_zh_sentence"],
                "skeleton_template": task["skeleton_template"],
                "fixed_chunks": task["fixed_chunks"],
                "slots": task["slots"],
                "slot_source_values": slot_source_values,
                "required_words": task.get("required_words") or [],
                "preferred_words": task.get("preferred_words") or [],
                "generation_count": task.get("generation_count") or 0,
                "external_constraints": task.get("external_constraints") or [],
                "provider": provider,
                "model": model,
                "sentence": None,
                "slot_values_model": {},
                "naturalness_note": None,
                "raw_result": result,
                "parse_failed": True,
                "llm_generated": True,
                "task_type": task.get("task_type"),
                "eligible_slot_ids": task.get("eligible_slot_ids") or [],
                "required_slot_replacements": task.get("required_slot_replacements") or [],
                "replace_all_eligible_slots": task.get("replace_all_eligible_slots", False),
                "slot_candidate_pools": task.get("slot_candidate_pools") or {},
                "replacement_plans": task.get("replacement_plans") or [],
            }
        ]
    for index, candidate in enumerate(candidates, start=1):
        sentence = candidate.get("sentence")
        if not sentence:
            continue
        rows.append(
            {
                "task_id": task["task_id"],
                "skeleton_id": task["skeleton_id"],
                "scene_id": task["scene_id"],
                "scene_label": task["scene_label"],
                "candidate_rank": index,
                "source_wz_sentence": task["source_wz_sentence"],
                "source_zh_sentence": task["source_zh_sentence"],
                "skeleton_template": task["skeleton_template"],
                "fixed_chunks": task["fixed_chunks"],
                "slots": task["slots"],
                "slot_source_values": slot_source_values,
                "required_words": task.get("required_words") or [],
                "preferred_words": task.get("preferred_words") or [],
                "generation_count": task.get("generation_count") or 0,
                "external_constraints": task.get("external_constraints") or [],
                "provider": provider,
                "model": model,
                "sentence": sentence,
                "slot_values_model": candidate.get("slot_values") or {},
                "plan_id": candidate.get("plan_id"),
                "naturalness_note": candidate.get("naturalness_note"),
                "llm_generated": True,
                "task_type": task.get("task_type"),
                "eligible_slot_ids": task.get("eligible_slot_ids") or [],
                "required_slot_replacements": task.get("required_slot_replacements") or [],
                "replace_all_eligible_slots": task.get("replace_all_eligible_slots", False),
                "slot_candidate_pools": task.get("slot_candidate_pools") or {},
                "replacement_plans": task.get("replacement_plans") or [],
            }
        )
    return rows


def build_error_row(task: dict[str, Any], model: str, provider: str, error_message: str) -> dict[str, Any]:
    slot_source_values = {slot["slot_id"]: slot["surface_wz"] for slot in task["slots"]}
    return {
        "task_id": task["task_id"],
        "skeleton_id": task["skeleton_id"],
        "scene_id": task["scene_id"],
        "scene_label": task["scene_label"],
        "candidate_rank": 0,
        "source_wz_sentence": task["source_wz_sentence"],
        "source_zh_sentence": task["source_zh_sentence"],
        "skeleton_template": task["skeleton_template"],
        "fixed_chunks": task["fixed_chunks"],
        "slots": task["slots"],
        "slot_source_values": slot_source_values,
        "required_words": task.get("required_words") or [],
        "preferred_words": task.get("preferred_words") or [],
        "generation_count": task.get("generation_count") or 0,
        "external_constraints": task.get("external_constraints") or [],
        "provider": provider,
        "model": model,
        "sentence": None,
        "slot_values_model": {},
        "naturalness_note": None,
        "llm_generated": False,
        "parse_failed": False,
        "request_failed": True,
        "request_error": error_message,
        "task_type": task.get("task_type"),
        "eligible_slot_ids": task.get("eligible_slot_ids") or [],
        "required_slot_replacements": task.get("required_slot_replacements") or [],
        "replace_all_eligible_slots": task.get("replace_all_eligible_slots", False),
        "slot_candidate_pools": task.get("slot_candidate_pools") or {},
        "replacement_plans": task.get("replacement_plans") or [],
    }


def build_preview_rows(tasks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks[:limit]:
        rows.append(
            {
                "task_id": task["task_id"],
                "skeleton_id": task["skeleton_id"],
                "scene_id": task["scene_id"],
                "scene_label": task["scene_label"],
                "source_wz_sentence": task["source_wz_sentence"],
                "skeleton_template": task["skeleton_template"],
                "required_words": task.get("required_words") or [],
                "preferred_words": task.get("preferred_words") or [],
                "prompt_bundle": task.get("prompt_bundle") or {},
                "llm_generated": False,
                "preview_only": True,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled Wenzhou sentences from task JSONL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--provider", choices=["primary", "secondary", "qwen", "deepseek"], default="primary")
    args = parser.parse_args()

    tasks = load_jsonl(args.input)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    if args.dry_run:
        preview_rows = build_preview_rows(tasks, limit=len(tasks))
        write_jsonl(args.output, preview_rows)
        print(f"Dry-run preview rows: {len(preview_rows)}")
        print(f"Wrote preview to {args.output}")
        return

    existing_rows: list[dict[str, Any]] = []
    completed_task_ids: set[str] = set()
    if args.resume and args.output.exists():
        existing_rows = load_jsonl(args.output)
        completed_task_ids = {row["task_id"] for row in existing_rows}

    client, model, provider = build_client(args.provider)
    output_rows = list(existing_rows)
    processed = 0
    failed = 0
    for index, task in enumerate(tasks, start=1):
        if task["task_id"] in completed_task_ids:
            continue
        print(f"[{index}/{len(tasks)}] generating {task['task_id']} ({task.get('scene_id')})", flush=True)
        try:
            result = request_generation(
                client,
                model,
                task,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            rows = flatten_candidates(task, result, model, provider)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            error_message = f"{type(exc).__name__}: {exc}"
            print(f"[{index}/{len(tasks)}] failed {task['task_id']}: {error_message}", flush=True)
            rows = [build_error_row(task, model, provider, error_message)]
            failed += 1
        output_rows.extend(rows)
        completed_task_ids.add(task["task_id"])
        processed += 1
        write_jsonl(args.output, output_rows)
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Input tasks: {len(tasks)}")
    print(f"Processed tasks this run: {processed}")
    print(f"Failed tasks this run: {failed}")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Output candidate rows: {len(output_rows)}")
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
