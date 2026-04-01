#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from generate_controlled_sentences import build_client
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wz_pipeline.dialect import ACTIVE_DIALECT_CONFIG
from wz_pipeline.grammar_guardrails import grammar_validation_reasons
from wz_pipeline.grammar_spec import relevant_spec_excerpt, relevant_spec_labels
from wz_pipeline.paths import DATA_DIR


DEFAULT_INPUT = DATA_DIR / "controlled_generation" / "filtered" / "pilot_v3_qwen" / "generation_candidates_accepted.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "controlled_generation" / "filtered" / "pilot_v3_qwen" / "generation_candidates_critic.jsonl"
DEFAULT_SUMMARY = DATA_DIR / "controlled_generation" / "filtered" / "pilot_v3_qwen" / "critic_summary.json"


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_grammar_payload(row: dict[str, Any]) -> dict[str, Any]:
    sentence = str(row.get("sentence") or row.get("wz_sentence") or "").strip()
    validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
    reasons = validation.get("grammar_reasons")
    if not isinstance(reasons, list):
        reasons = grammar_validation_reasons(sentence)
    sections = validation.get("grammar_spec_sections")
    if not isinstance(sections, list):
        sections = relevant_spec_labels(sentence, reasons)
    return {
        "candidate_grammar_reasons": reasons,
        "candidate_grammar_sections": sections,
        "grammar_spec_excerpt": relevant_spec_excerpt(sentence, reasons, max_lines=3),
    }


def build_domain_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain_ids": row.get("domain_ids") or [],
        "domain_labels": row.get("domain_labels") or [],
        "domain_required_terms": row.get("domain_required_terms") or [],
        "domain_preferred_terms": row.get("domain_preferred_terms") or [],
        "domain_blocked_terms": row.get("domain_blocked_terms") or [],
        "domain_prompt_notes": row.get("domain_prompt_notes") or [],
        "domain_review_notes": row.get("domain_review_notes") or [],
        "domain_required_hits": ((row.get("validation") or {}).get("domain_required_hits") or []),
        "domain_preferred_hits": ((row.get("validation") or {}).get("domain_preferred_hits") or []),
        "domain_blocked_hits": ((row.get("validation") or {}).get("domain_blocked_hits") or []),
    }


def build_prompt(row: dict[str, Any]) -> tuple[str, str]:
    grammar_payload = build_grammar_payload(row)
    domain_payload = build_domain_payload(row)
    system = (
        f"你是{ACTIVE_DIALECT_CONFIG.dialect_name}受控生成质检员。"
        "你只做质量判定，不改写。"
        "重点判断："
        f"1. 句子像不像自然{ACTIVE_DIALECT_CONFIG.dialect_name}口语；"
        "2. required_words 是否被硬塞；"
        "3. 是否和 scene_id 大体匹配；"
        "4. 是否适合后续语音训练；"
        "5. 功能词、体貌、否定、语序是否符合提供的语法规范。"
        "6. 如果指定了领域词和领域禁用词，要同时判断领域覆盖和领域误用。"
        "如果语义明显别扭，即使格式过关，也必须判 fail。"
        "只输出 JSON object。"
    )
    payload = {
        "task_type": row.get("task_type"),
        "scene_id": row.get("scene_id"),
        "source_wz_sentence": row.get("source_wz_sentence"),
        "skeleton_template": row.get("skeleton_template"),
        "required_words": row.get("required_words"),
        "preferred_words": row.get("preferred_words"),
        "candidate_sentence": row.get("sentence"),
        "eligible_slot_ids": row.get("eligible_slot_ids") or [],
        "required_slot_replacements": row.get("required_slot_replacements") or [],
        "slot_values_model": row.get("slot_values_model") or {},
        "eligible_slot_replacement_ratio": row.get("eligible_slot_replacement_ratio"),
        "pool_compliance_ratio": row.get("pool_compliance_ratio"),
        **grammar_payload,
        **domain_payload,
        "slots": [
            {
                "slot_id": slot.get("slot_id"),
                "surface_wz": slot.get("surface_wz"),
                "surface_zh": slot.get("surface_zh"),
                "slot_kind": slot.get("slot_kind"),
                "semantic_class": slot.get("semantic_class"),
            }
            for slot in row.get("slots") or []
        ],
        "output_schema": {
            "pass": "boolean",
            "score": "0.0-1.0 float",
            "issue_types": ["semantic_mismatch | forced_insertion | scene_mismatch | unnatural_wz | speech_training_risk"],
            "grammar_pass": "boolean",
            "grammar_issue_types": ["particle_misuse | aspect_misuse | negation_misuse | object_order | mandarin_leakage | unsupported_form"],
            "domain_pass": "boolean",
            "domain_issue_types": ["domain_required_missing | domain_blocked_term | domain_term_misuse | domain_scene_mismatch"],
            "cited_sections": ["grammar section titles"],
            "note": "short Chinese note under 30 chars",
        },
    }
    user = "根据下面 JSON 做质量判定，只返回 JSON：\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    return system, user


def request_review(client: OpenAI, model: str, row: dict[str, Any], timeout: float | None) -> dict[str, Any]:
    system, user = build_prompt(row)
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


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    raw_score = result.get("score", 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    score = min(max(score, 0.0), 1.0)
    issue_types = result.get("issue_types") or []
    if not isinstance(issue_types, list):
        issue_types = []
    issue_types = [str(item) for item in issue_types][:5]
    grammar_issue_types = result.get("grammar_issue_types") or []
    if not isinstance(grammar_issue_types, list):
        grammar_issue_types = []
    grammar_issue_types = [str(item) for item in grammar_issue_types][:5]
    domain_issue_types = result.get("domain_issue_types") or []
    if not isinstance(domain_issue_types, list):
        domain_issue_types = []
    domain_issue_types = [str(item) for item in domain_issue_types][:5]
    cited_sections = result.get("cited_sections") or []
    if not isinstance(cited_sections, list):
        cited_sections = []
    cited_sections = [str(item) for item in cited_sections if str(item).strip()][:6]
    return {
        "critic_pass": bool(result.get("pass", False)),
        "critic_score": round(score, 4),
        "critic_issue_types": issue_types,
        "critic_grammar_pass": bool(result.get("grammar_pass", False)),
        "critic_grammar_issue_types": grammar_issue_types,
        "critic_domain_pass": bool(result.get("domain_pass", False)),
        "critic_domain_issue_types": domain_issue_types,
        "critic_cited_sections": cited_sections,
        "critic_note": str(result.get("note") or "").strip()[:30],
    }


def build_error_result(error_message: str) -> dict[str, Any]:
    return {
        "critic_pass": False,
        "critic_score": 0.0,
        "critic_issue_types": ["critic_request_failed"],
        "critic_grammar_pass": False,
        "critic_grammar_issue_types": ["critic_request_failed"],
        "critic_domain_pass": False,
        "critic_domain_issue_types": ["critic_request_failed"],
        "critic_cited_sections": [],
        "critic_note": error_message[:30],
        "critic_request_failed": True,
        "critic_request_error": error_message,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Use an LLM critic to review accepted controlled-generation candidates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--provider", choices=["primary", "secondary", "qwen", "deepseek"], default="qwen")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    client, model, provider = build_client(args.provider)

    reviewed_rows: list[dict[str, Any]] = []
    failed = 0
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] reviewing {row.get('task_id')}#{row.get('candidate_rank')}", flush=True)
        try:
            result = request_review(client, model, row, timeout=args.timeout)
            normalized = normalize_result(result)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            error_message = f"{type(exc).__name__}: {exc}"
            print(f"[{index}/{len(rows)}] failed {row.get('task_id')}#{row.get('candidate_rank')}: {error_message}", flush=True)
            normalized = build_error_result(error_message)
            failed += 1
        merged = dict(row)
        merged.update(normalized)
        reviewed_rows.append(merged)
        write_jsonl(args.output, reviewed_rows)
        write_json(
            args.summary,
            {
                "input_rows": len(rows),
                "reviewed_rows": len(reviewed_rows),
                "critic_pass_rows": sum(1 for item in reviewed_rows if item["critic_pass"]),
                "critic_fail_rows": sum(1 for item in reviewed_rows if not item["critic_pass"]),
                "failed_request_rows": failed,
                "issue_type_counts": dict(Counter(issue for item in reviewed_rows for issue in item["critic_issue_types"])),
                "grammar_issue_type_counts": dict(
                    Counter(issue for item in reviewed_rows for issue in item["critic_grammar_issue_types"])
                ),
                "domain_issue_type_counts": dict(
                    Counter(issue for item in reviewed_rows for issue in item["critic_domain_issue_types"])
                ),
                "provider": provider,
                "model": model,
                "output_path": str(args.output),
            },
        )

    print(f"Input rows: {len(rows)}")
    print(f"Critic pass rows: {sum(1 for row in reviewed_rows if row['critic_pass'])}")
    print(f"Critic fail rows: {sum(1 for row in reviewed_rows if not row['critic_pass'])}")
    print(f"Failed request rows: {failed}")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
