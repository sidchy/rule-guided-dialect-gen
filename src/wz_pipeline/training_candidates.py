from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from .contracts import load_trust_policy, validate_contract


def _accept_row(row: dict[str, Any], allowed_tiers: set[str]) -> tuple[bool, str]:
    try:
        validate_contract(row)
    except ValueError:
        return False, "invalid_contract"
    if not bool(row.get("upstream_eligible")):
        return False, "not_upstream_eligible"
    trust_tier = str(row.get("trust_tier", "")).strip()
    if trust_tier not in allowed_tiers:
        return False, "disallowed_trust_tier"
    return True, "accepted"


def build_training_candidates(
    seed_rows: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed_tiers = set(load_trust_policy().get("allowed_upstream_trust_tiers", []))
    selected: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    stats = {
        "input_seed_count": len(seed_rows),
        "input_generated_reviewed_count": len(generated_rows),
        "accepted_source_count": 0,
        "accepted_curated_count": 0,
        "skipped_counts": Counter(),
        "deduped_count": 0,
    }

    for pool_name, rows in (("source", seed_rows), ("curated", generated_rows)):
        for row in rows:
            ok, reason = _accept_row(row, allowed_tiers)
            if not ok:
                stats["skipped_counts"][f"{pool_name}:{reason}"] += 1
                continue
            sample_id = str(row.get("sample_id", "")).strip()
            if sample_id and sample_id in seen_sample_ids:
                stats["deduped_count"] += 1
                continue
            if sample_id:
                seen_sample_ids.add(sample_id)
            selected.append(dict(row))
            if pool_name == "source":
                stats["accepted_source_count"] += 1
            else:
                stats["accepted_curated_count"] += 1

    summary = {
        "pipeline_name": "build_training_candidates",
        "generated_at": datetime.now().isoformat(),
        "counts": {
            "input_seed_count": stats["input_seed_count"],
            "input_generated_reviewed_count": stats["input_generated_reviewed_count"],
            "accepted_source_count": stats["accepted_source_count"],
            "accepted_curated_count": stats["accepted_curated_count"],
            "output_total_count": len(selected),
            "deduped_count": stats["deduped_count"],
            "skipped_counts": dict(stats["skipped_counts"]),
        },
    }
    return selected, summary
