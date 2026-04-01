#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from wz_pipeline.contracts import apply_contract
from wz_pipeline.jsonl import read_jsonl, write_jsonl
from wz_pipeline.registry import register_run
from wz_pipeline.review import export_review_tsv
from wz_pipeline.runs import build_summary, make_run_id, prepare_run_layout, write_json
from wz_pipeline.paths import ROOT
from wz_pipeline.scene_policy import MODERN_SIDECAR_SCENES
from wz_pipeline.pipelines.fewshot_batch import (
    CORE_POLICY,
    DENSE_SCENE_SEEDS,
    EXTRACTED_SHORT,
    choose_scene_for_example,
    clean_wz,
    core_is_blocked,
    definition_looks_archaic,
    example_looks_usable,
    load_primary_scene_by_word,
    scene_deny_terms,
    scene_match_score,
)


PIPELINE_NAME = "modern_anchor_candidates"
POLICY_VERSION = "sidecar_anchor_v1_2026-04-01"
REPORT_DIR = ROOT / "data" / "generated_long_sentences" / "reports"
OUTPUT_JSON = REPORT_DIR / "modern_anchor_candidates.json"
OUTPUT_MD = REPORT_DIR / "modern_anchor_candidates.md"
OUTPUT_JSONL = REPORT_DIR / "modern_anchor_examples.jsonl"
CURATED_GENERATED = ROOT / "curated" / "generated_reviewed.jsonl"
MIN_ANCHOR_THRESHOLD = 8
STRONG_CURATED_CUES = {
    "digital_chat": {"手机", "电话", "微信", "视频", "链接", "群里", "信号", "充电", "截图", "网络", "照片"},
    "work_study": {"公司", "学校", "考试", "老师", "同学", "作业", "电脑", "会议", "文件"},
}


def load_seed_terms() -> dict[str, set[str]]:
    if not DENSE_SCENE_SEEDS.exists():
        return {}
    data = json.loads(DENSE_SCENE_SEEDS.read_text(encoding="utf-8"))
    return {
        scene: {
            str(row.get("term") or "").strip()
            for row in rows
            if str(row.get("term") or "").strip()
        }
        for scene, rows in data.items()
    }


def dirty_source(source_file: str) -> bool:
    deny_patterns = tuple(CORE_POLICY.get("global", {}).get("deny_source_files", []))
    return bool(source_file and any(pattern in source_file for pattern in deny_patterns))


def clean_zh(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def modern_hits_for_row(scene_id: str, row: dict[str, Any], seed_terms: dict[str, set[str]]) -> list[str]:
    merged = " ".join(
        [
            str(row.get("wz_word") or ""),
            str(row.get("definition") or ""),
            str(row.get("wz_sentence") or ""),
            str(row.get("zh_sentence") or ""),
        ]
    )
    return sorted(term for term in seed_terms.get(scene_id, set()) if term and term in merged)


def strong_curated_hits(scene_id: str, row: dict[str, Any]) -> list[str]:
    cues = STRONG_CURATED_CUES.get(scene_id, set())
    merged = " ".join(
        [
            str(row.get("wz_sentence") or ""),
            str(row.get("core_word") or ""),
            " ".join(row.get("support_words", []) if isinstance(row.get("support_words"), list) else []),
            str(row.get("zh_sentence") or ""),
        ]
    )
    return sorted(cue for cue in cues if cue in merged)


def load_curated_generated_rows(scene_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not CURATED_GENERATED.exists():
        return rows
    for row in read_jsonl(CURATED_GENERATED):
        if not row.get("upstream_eligible"):
            continue
        if row.get("scene_id") != scene_id:
            continue
        hits = strong_curated_hits(scene_id, row)
        if not hits:
            continue
        rows.append(
            {
                "scene_id": scene_id,
                "wz_word": str(row.get("core_word") or row.get("wz_word") or "").strip(),
                "definition": str(row.get("core_word_def") or row.get("definition") or "").strip(),
                "wz_sentence": clean_wz(str(row.get("wz_sentence") or "")),
                "zh_sentence": clean_zh(str(row.get("zh_sentence") or "")),
                "source_file": str(CURATED_GENERATED),
                "modern_hits": hits,
                "anchor_origin": "curated_generated",
            }
        )
    return rows


def evaluate_anchor(
    scene_id: str,
    row: dict[str, Any],
    seed_terms: dict[str, set[str]],
) -> tuple[bool, list[str], list[str]]:
    reasons: list[str] = []
    wz_word = str(row.get("wz_word") or "").strip()
    definition = str(row.get("definition") or "").strip()
    source_file = str(row.get("source_file") or "")
    anchor_origin = str(row.get("anchor_origin") or "extracted_short")

    if anchor_origin == "extracted_short" and not example_looks_usable(row):
        reasons.append("example_not_usable")
    if definition_looks_archaic(definition):
        reasons.append("archaic_definition")
    if dirty_source(source_file):
        reasons.append("dirty_source")
    if core_is_blocked(scene_id, wz_word, definition, source_file):
        reasons.append("blocked_core")
    if wz_word in scene_deny_terms(scene_id):
        reasons.append("scene_deny_term")
    if anchor_origin == "extracted_short" and scene_match_score(
        scene_id,
        row.get("wz_word", ""),
        row.get("definition", ""),
        row.get("zh_sentence", ""),
        row.get("wz_sentence", ""),
    ) <= 0:
        reasons.append("scene_mismatch")

    hits = row.get("modern_hits") or modern_hits_for_row(scene_id, row, seed_terms)
    if not hits:
        reasons.append("no_modern_cues")
    return len(reasons) == 0, reasons, hits


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(
        pipeline_name=PIPELINE_NAME,
        provider="offline",
        model="inventory",
        policy_version=POLICY_VERSION,
    )
    layout = prepare_run_layout(pipeline_name=PIPELINE_NAME, run_id=run_id)
    config = {
        "extracted_short": str(EXTRACTED_SHORT),
        "curated_generated": str(CURATED_GENERATED),
        "threshold": MIN_ANCHOR_THRESHOLD,
        "target_scenes": MODERN_SIDECAR_SCENES,
        "policy_version": POLICY_VERSION,
    }
    write_json(layout.config_path, config)

    seed_terms = load_seed_terms()
    word_scene_map = load_primary_scene_by_word()
    all_rows: list[dict[str, Any]] = []
    for row in read_jsonl(EXTRACTED_SHORT):
        row["auto_scene_id"] = choose_scene_for_example(row, word_scene_map)
        row["anchor_origin"] = "extracted_short"
        all_rows.append(row)

    summary_payload: dict[str, Any] = {
        "threshold": MIN_ANCHOR_THRESHOLD,
        "target_scenes": MODERN_SIDECAR_SCENES,
        "policy_version": POLICY_VERSION,
        "scenes": {},
    }
    md_lines = [
        "# Modern Anchor Candidates",
        "",
        f"- threshold: `{MIN_ANCHOR_THRESHOLD}`",
        f"- scenes: `{', '.join(MODERN_SIDECAR_SCENES)}`",
        f"- source policy: `source + curated only`",
        "",
    ]
    exported_examples: list[dict[str, Any]] = []

    for scene_id in MODERN_SIDECAR_SCENES:
        extracted_rows = [
            row
            for row in all_rows
            if scene_match_score(
                scene_id,
                row.get("wz_word", ""),
                row.get("definition", ""),
                row.get("zh_sentence", ""),
                row.get("wz_sentence", ""),
            ) > 0
            or modern_hits_for_row(scene_id, row, seed_terms)
        ]
        curated_rows = load_curated_generated_rows(scene_id)
        rows = extracted_rows + curated_rows

        rejection_counts: Counter[str] = Counter()
        candidates: list[dict[str, Any]] = []
        for row in rows:
            ok, reasons, hits = evaluate_anchor(scene_id, row, seed_terms)
            if ok:
                candidate = {
                    "scene_id": scene_id,
                    "wz_word": str(row.get("wz_word") or "").strip(),
                    "definition": str(row.get("definition") or "").strip(),
                    "wz_sentence": clean_wz(str(row.get("wz_sentence") or "")),
                    "zh_sentence": clean_zh(str(row.get("zh_sentence") or "")),
                    "source_file": str(row.get("source_file") or ""),
                    "modern_hits": hits,
                    "anchor_origin": str(row.get("anchor_origin") or "extracted_short"),
                }
                candidates.append(candidate)
            else:
                for reason in reasons:
                    rejection_counts[reason] += 1

        candidates.sort(
            key=lambda row: (
                len(row.get("modern_hits", [])),
                row.get("anchor_origin", ""),
                row.get("wz_word", ""),
                row.get("wz_sentence", ""),
            ),
            reverse=True,
        )
        scene_summary = {
            "input_examples": len(rows),
            "input_extracted_examples": len(extracted_rows),
            "input_curated_generated_examples": len(curated_rows),
            "eligible_anchor_count": len(candidates),
            "threshold_met": len(candidates) >= MIN_ANCHOR_THRESHOLD,
            "rejection_reason_counts": dict(rejection_counts),
            "candidates": candidates[:20],
        }
        summary_payload["scenes"][scene_id] = scene_summary
        if scene_summary["threshold_met"]:
            exported_examples.extend(candidates[:20])

        md_lines.extend(
            [
                f"## {scene_id}",
                f"- input_examples: `{len(rows)}`",
                f"- input_extracted_examples: `{len(extracted_rows)}`",
                f"- input_curated_generated_examples: `{len(curated_rows)}`",
                f"- eligible_anchor_count: `{len(candidates)}`",
                f"- threshold_met: `{scene_summary['threshold_met']}`",
                "",
            ]
        )

    contracted_results = []
    for row in exported_examples:
        source_type = "reviewed_curated" if row["anchor_origin"] == "curated_generated" else "extracted_seed"
        contracted_results.append(
            apply_contract(
                row,
                source_type=source_type,
                origin_run_id=run_id,
                pipeline_name=PIPELINE_NAME,
                policy_version=POLICY_VERSION,
                scene_id=row.get("scene_id", ""),
                lane="sidecar_candidate",
                core_word=row.get("wz_word", ""),
                support_words=[],
                sample_prefix="anchor",
            )
        )

    write_jsonl(layout.results_path, contracted_results)
    write_jsonl(layout.rule_gate_path, contracted_results)
    export_review_tsv(layout.review_tsv_path, [])
    write_jsonl(layout.promotion_candidates_path, [])

    OUTPUT_JSON.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    write_jsonl(OUTPUT_JSONL, exported_examples)

    summary = build_summary(
        pipeline_name=PIPELINE_NAME,
        run_id=run_id,
        config=config,
        raw_count=sum(scene["input_examples"] for scene in summary_payload["scenes"].values()),
        rule_pass_count=len(contracted_results),
        extra={
            "compatibility_outputs": {
                "json": str(OUTPUT_JSON),
                "markdown": str(OUTPUT_MD),
                "jsonl": str(OUTPUT_JSONL),
            }
        },
    )
    write_json(layout.summary_path, summary)
    register_run(
        {
            "pipeline_name": PIPELINE_NAME,
            "run_id": run_id,
            "config_path": str(layout.config_path),
            "results_path": str(layout.results_path),
            "rule_gate_path": str(layout.rule_gate_path),
            "review_tsv_path": str(layout.review_tsv_path),
            "summary_path": str(layout.summary_path),
            "policy_version": POLICY_VERSION,
        }
    )
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
