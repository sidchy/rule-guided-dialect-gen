from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from .paths import DOMAIN_CATALOG_PATH


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_domain_ids(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = list(value)
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        domain_id = str(item).strip()
        if not domain_id or domain_id in seen:
            continue
        seen.add(domain_id)
        ordered.append(domain_id)
    return ordered


@lru_cache(maxsize=1)
def load_domain_catalog() -> dict[str, dict[str, Any]]:
    if not DOMAIN_CATALOG_PATH.exists():
        return {}
    payload = json.loads(DOMAIN_CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Domain catalog must be a JSON list: {DOMAIN_CATALOG_PATH}")

    catalog: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        domain_id = str(row.get("domain_id") or "").strip()
        if not domain_id:
            continue
        catalog[domain_id] = {
            "domain_id": domain_id,
            "label": str(row.get("label") or domain_id),
            "scene_allowlist": _string_list(row.get("scene_allowlist")),
            "scene_denylist": _string_list(row.get("scene_denylist")),
            "required_terms": _string_list(row.get("required_terms")),
            "preferred_terms": _string_list(row.get("preferred_terms")),
            "blocked_terms": _string_list(row.get("blocked_terms")),
            "prompt_notes": _string_list(row.get("prompt_notes")),
            "review_notes": _string_list(row.get("review_notes")),
        }
    return catalog


def resolve_domain_entries(domain_ids: list[str] | None) -> list[dict[str, Any]]:
    catalog = load_domain_catalog()
    selected = normalize_domain_ids(domain_ids)
    missing = [domain_id for domain_id in selected if domain_id not in catalog]
    if missing:
        raise ValueError(f"Unknown domain ids: {', '.join(missing)}")
    return [catalog[domain_id] for domain_id in selected]


def _applies_to_scene(entry: dict[str, Any], scene_id: str) -> bool:
    allowlist = set(entry.get("scene_allowlist") or [])
    denylist = set(entry.get("scene_denylist") or [])
    if allowlist and scene_id not in allowlist:
        return False
    if scene_id in denylist:
        return False
    return True


def build_domain_context(domain_ids: list[str] | None, *, scene_id: str) -> dict[str, Any]:
    selected_entries = [
        entry
        for entry in resolve_domain_entries(domain_ids)
        if _applies_to_scene(entry, scene_id)
    ]
    prompt_notes: list[str] = []
    review_notes: list[str] = []
    required_terms: list[str] = []
    preferred_terms: list[str] = []
    blocked_terms: list[str] = []
    labels: list[str] = []

    for entry in selected_entries:
        labels.append(str(entry.get("label") or entry["domain_id"]))
        prompt_notes.extend(entry.get("prompt_notes") or [])
        review_notes.extend(entry.get("review_notes") or [])
        required_terms.extend(entry.get("required_terms") or [])
        preferred_terms.extend(entry.get("preferred_terms") or [])
        blocked_terms.extend(entry.get("blocked_terms") or [])

    def ordered(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            clean = str(item).strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            result.append(clean)
        return result

    return {
        "domain_ids": [entry["domain_id"] for entry in selected_entries],
        "domain_labels": ordered(labels),
        "required_terms": ordered(required_terms),
        "preferred_terms": ordered(preferred_terms),
        "blocked_terms": ordered(blocked_terms),
        "prompt_notes": ordered(prompt_notes),
        "review_notes": ordered(review_notes),
    }
