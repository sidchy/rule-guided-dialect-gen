from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .domain_config import build_domain_context, normalize_domain_ids
from .paths import RUNS_DIR
from .scene_policy import FOCUS_SCENES, MODERN_SIDECAR_SCENES, PRIORITY_SCENES


def resolve_feedback_summary_path(
    *,
    feedback_summary_path: Path | None = None,
    feedback_run_id: str = "",
    pipeline_name: str = "fewshot_batch",
) -> Path | None:
    if feedback_summary_path is not None:
        return Path(feedback_summary_path)
    run_id = str(feedback_run_id).strip()
    if not run_id:
        return None
    return RUNS_DIR / pipeline_name / run_id / "summary.json"


def load_feedback_summary(
    *,
    feedback_summary_path: Path | None = None,
    feedback_run_id: str = "",
    pipeline_name: str = "fewshot_batch",
) -> dict[str, Any]:
    path = resolve_feedback_summary_path(
        feedback_summary_path=feedback_summary_path,
        feedback_run_id=feedback_run_id,
        pipeline_name=pipeline_name,
    )
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Feedback summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Feedback summary must be a JSON object: {path}")
    payload["_feedback_summary_path"] = str(path)
    return payload


def _string_int_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, raw_value in value.items():
        name = str(key).strip()
        if not name:
            continue
        try:
            count = int(raw_value)
        except (TypeError, ValueError):
            continue
        result[name] = count
    return result


def _primary_bucket(summary: dict[str, Any]) -> str:
    machine_metrics = summary.get("machine_metrics") if isinstance(summary.get("machine_metrics"), dict) else {}
    failure_analysis = machine_metrics.get("failure_analysis") if isinstance(machine_metrics.get("failure_analysis"), dict) else {}
    primary_bucket = str(failure_analysis.get("primary_bucket") or "").strip()
    if primary_bucket:
        return primary_bucket
    extra = summary.get("extra") if isinstance(summary.get("extra"), dict) else {}
    bucket_counts = _string_int_map(extra.get("rule_fail_bucket_counts"))
    ranked = sorted(bucket_counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[0][0] if ranked else ""


def build_feedback_plan(
    summary: dict[str, Any] | None,
    *,
    scenes: list[str] | None,
    domain_ids: list[str] | None,
    default_food_modern_trial_ratio: float,
) -> dict[str, Any]:
    selected_scenes = [scene for scene in (scenes or PRIORITY_SCENES) if str(scene).strip()]
    selected_domain_ids = normalize_domain_ids(domain_ids)
    if not summary:
        return {
            "enabled": False,
            "source_run_id": "",
            "source_summary_path": "",
            "primary_bucket": "",
            "bucket_counts": {},
            "scene_weights": {scene: 1.0 for scene in selected_scenes},
            "anchor_pool_top_k": 0,
            "domain_anchor_pool_top_k": 0,
            "food_modern_trial_ratio": default_food_modern_trial_ratio,
            "notes": [],
        }

    extra = summary.get("extra") if isinstance(summary.get("extra"), dict) else {}
    machine_metrics = summary.get("machine_metrics") if isinstance(summary.get("machine_metrics"), dict) else {}
    failure_analysis = machine_metrics.get("failure_analysis") if isinstance(machine_metrics.get("failure_analysis"), dict) else {}
    primary_bucket = _primary_bucket(summary)
    bucket_counts = _string_int_map(failure_analysis.get("bucket_counts") or extra.get("rule_fail_bucket_counts"))
    scene_pass_counts = _string_int_map(extra.get("scene_rule_pass_counts") or extra.get("scene_pass_counts"))

    scene_weights: dict[str, float] = {}
    notes: list[str] = []
    for scene in selected_scenes:
        pass_count = max(scene_pass_counts.get(scene, 0), 0)
        weight = 1.0 + min(pass_count, 4) * 0.5
        domain_active = bool(build_domain_context(selected_domain_ids, scene_id=scene).get("domain_ids"))
        if selected_domain_ids and domain_active:
            weight += 0.5

        if primary_bucket == "grammar":
            if scene in FOCUS_SCENES:
                weight += 1.0
            if scene in MODERN_SIDECAR_SCENES:
                weight = max(0.5, weight - 0.5)
        elif primary_bucket == "domain":
            if domain_active:
                weight += 2.0
            elif selected_domain_ids:
                weight = max(0.5, weight - 0.5)
        elif primary_bucket == "naturalness":
            if scene in MODERN_SIDECAR_SCENES:
                weight = max(0.5, weight - 0.25)
            if pass_count == 0:
                weight = max(0.75, weight - 0.25)

        scene_weights[scene] = round(weight, 3)

    anchor_pool_top_k = 0
    domain_anchor_pool_top_k = 0
    food_modern_trial_ratio = default_food_modern_trial_ratio
    if primary_bucket == "grammar":
        anchor_pool_top_k = 2
        domain_anchor_pool_top_k = 2
        food_modern_trial_ratio = round(max(0.05, default_food_modern_trial_ratio * 0.5), 4)
        notes.append("grammar-heavy previous run: tighter anchor sampling and lower modern food trial ratio")
    elif primary_bucket == "domain":
        anchor_pool_top_k = 3
        domain_anchor_pool_top_k = 2
        notes.append("domain-heavy previous run: bias quotas toward scenes compatible with selected domains")
    elif primary_bucket == "naturalness":
        anchor_pool_top_k = 3
        domain_anchor_pool_top_k = 3
        food_modern_trial_ratio = round(max(0.05, default_food_modern_trial_ratio * 0.75), 4)
        notes.append("naturalness-heavy previous run: favor higher-signal scenes and reduce random spread")

    return {
        "enabled": True,
        "source_run_id": str(summary.get("run_id") or ""),
        "source_summary_path": str(summary.get("_feedback_summary_path") or ""),
        "primary_bucket": primary_bucket,
        "bucket_counts": bucket_counts,
        "scene_weights": scene_weights,
        "anchor_pool_top_k": anchor_pool_top_k,
        "domain_anchor_pool_top_k": domain_anchor_pool_top_k,
        "food_modern_trial_ratio": food_modern_trial_ratio,
        "notes": notes,
    }
