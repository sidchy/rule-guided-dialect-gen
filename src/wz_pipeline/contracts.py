from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .paths import CONFIGS_DIR


TRUST_POLICY_PATH = CONFIGS_DIR / "trust_policy.json"

SOURCE_TYPES = {"raw_source", "extracted_seed", "generated_candidate", "reviewed_curated"}
RULE_GATE_STATUS = {"pass", "fail", "not_run"}
HUMAN_REVIEW_STATUS = {"pending", "pass", "fail", "not_required"}
TRUST_TIERS = {"source", "candidate", "reviewed", "curated"}


def load_trust_policy() -> dict[str, Any]:
    return json.loads(TRUST_POLICY_PATH.read_text(encoding="utf-8"))


def stable_sample_id(payload: dict[str, Any], prefix: str = "sample") -> str:
    digest_payload = {
        "origin_run_id": payload.get("origin_run_id", ""),
        "pipeline_name": payload.get("pipeline_name", ""),
        "scene_id": payload.get("scene_id", ""),
        "lane": payload.get("lane", ""),
        "core_word": payload.get("core_word", ""),
        "support_words": payload.get("support_words", []),
        "wz_sentence": payload.get("wz_sentence", ""),
        "zh_sentence": payload.get("zh_sentence", ""),
        "source_row_id": payload.get("source_row_id", ""),
    }
    digest = hashlib.md5(
        json.dumps(digest_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}_{digest}"


def normalize_support_words(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split("|") if item.strip()]
    return []


def apply_contract(
    row: dict[str, Any],
    *,
    source_type: str,
    origin_run_id: str,
    pipeline_name: str,
    policy_version: str,
    scene_id: str = "",
    lane: str = "",
    core_word: str = "",
    support_words: list[str] | None = None,
    rule_gate_status: str | None = None,
    human_review_status: str | None = None,
    trust_tier: str | None = None,
    upstream_eligible: bool | None = None,
    sample_prefix: str = "sample",
) -> dict[str, Any]:
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"Unsupported source_type: {source_type}")

    trust_policy = load_trust_policy()["source_types"][source_type]
    merged = dict(row)
    merged["source_type"] = source_type
    merged["origin_run_id"] = origin_run_id
    merged["pipeline_name"] = pipeline_name
    merged["policy_version"] = policy_version
    merged["scene_id"] = scene_id or str(row.get("scene_id") or "")
    merged["lane"] = lane or str(row.get("lane") or "")
    merged["core_word"] = core_word or str(row.get("core_word") or "")
    merged["support_words"] = normalize_support_words(
        support_words if support_words is not None else row.get("support_words")
    )
    merged["rule_gate_status"] = rule_gate_status or trust_policy["rule_gate_status"]
    merged["human_review_status"] = (
        human_review_status or trust_policy["human_review_status"]
    )
    merged["trust_tier"] = trust_tier or trust_policy["trust_tier"]
    merged["upstream_eligible"] = (
        upstream_eligible if upstream_eligible is not None else trust_policy["upstream_eligible"]
    )
    merged["sample_id"] = str(row.get("sample_id") or stable_sample_id(merged, prefix=sample_prefix))
    return merged


def validate_contract(row: dict[str, Any]) -> None:
    required_fields = [
        "sample_id",
        "source_type",
        "origin_run_id",
        "pipeline_name",
        "policy_version",
        "scene_id",
        "lane",
        "core_word",
        "support_words",
        "rule_gate_status",
        "human_review_status",
        "trust_tier",
        "upstream_eligible",
    ]
    missing = [field for field in required_fields if field not in row]
    if missing:
        raise ValueError(f"Missing contract fields: {missing}")
    if row["source_type"] not in SOURCE_TYPES:
        raise ValueError(f"Bad source_type: {row['source_type']}")
    if row["rule_gate_status"] not in RULE_GATE_STATUS:
        raise ValueError(f"Bad rule_gate_status: {row['rule_gate_status']}")
    if row["human_review_status"] not in HUMAN_REVIEW_STATUS:
        raise ValueError(f"Bad human_review_status: {row['human_review_status']}")
    if row["trust_tier"] not in TRUST_TIERS:
        raise ValueError(f"Bad trust_tier: {row['trust_tier']}")


def contract_fields() -> list[str]:
    return [
        "sample_id",
        "source_type",
        "origin_run_id",
        "pipeline_name",
        "policy_version",
        "scene_id",
        "lane",
        "core_word",
        "support_words",
        "rule_gate_status",
        "human_review_status",
        "trust_tier",
        "upstream_eligible",
    ]

