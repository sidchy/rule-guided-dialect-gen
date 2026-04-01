from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import RUNS_DIR, ensure_dir


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-").lower()


def make_run_id(
    *,
    pipeline_name: str,
    provider: str = "",
    model: str = "",
    policy_version: str = "",
    seed: int | None = None,
    now: datetime | None = None,
) -> str:
    moment = now or datetime.now()
    pieces = [
        _slug(pipeline_name),
        moment.strftime("%Y%m%d_%H%M%S"),
    ]
    if provider:
        pieces.append(_slug(provider))
    if model:
        pieces.append(_slug(model))
    if policy_version:
        pieces.append(_slug(policy_version))
    if seed is not None:
        pieces.append(f"seed{seed}")
    return "__".join(piece for piece in pieces if piece)


@dataclass(frozen=True)
class RunLayout:
    pipeline_name: str
    run_id: str
    root: Path
    config_path: Path
    results_path: Path
    rule_gate_path: Path
    review_tsv_path: Path
    review_results_path: Path
    summary_path: Path
    promotion_candidates_path: Path


def prepare_run_layout(
    *,
    pipeline_name: str,
    run_id: str,
) -> RunLayout:
    root = ensure_dir(RUNS_DIR / pipeline_name / run_id)
    return RunLayout(
        pipeline_name=pipeline_name,
        run_id=run_id,
        root=root,
        config_path=root / "config.json",
        results_path=root / "results.jsonl",
        rule_gate_path=root / "rule_gate.jsonl",
        review_tsv_path=root / "review.tsv",
        review_results_path=root / "review_results.jsonl",
        summary_path=root / "summary.json",
        promotion_candidates_path=root / "promotion_candidates.jsonl",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_summary(
    *,
    pipeline_name: str,
    run_id: str,
    config: dict[str, Any],
    raw_count: int,
    rule_pass_count: int,
    reviewed_count: int = 0,
    human_pass_count: int = 0,
    scene_human_pass_rate: dict[str, float] | None = None,
    lane_human_pass_rate: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "pipeline_name": pipeline_name,
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "config": config,
        "machine_metrics": {
            "raw_count": raw_count,
            "rule_pass_count": rule_pass_count,
            "rule_pass_rate": round(rule_pass_count / max(1, raw_count), 4),
        },
        "human_metrics": {
            "reviewed_count": reviewed_count,
            "human_pass_count": human_pass_count,
            "human_pass_rate": round(human_pass_count / max(1, reviewed_count), 4),
            "scene_human_pass_rate": scene_human_pass_rate or {},
            "lane_human_pass_rate": lane_human_pass_rate or {},
        },
        "extra": extra or {},
    }

