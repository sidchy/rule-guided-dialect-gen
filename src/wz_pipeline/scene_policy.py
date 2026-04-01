from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from .paths import CONFIGS_DIR


SCENE_POLICY_PATH = CONFIGS_DIR / "scene_policy.json"


@lru_cache(maxsize=1)
def load_scene_policy() -> dict[str, Any]:
    return json.loads(SCENE_POLICY_PATH.read_text(encoding="utf-8"))


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


POLICY = load_scene_policy()
FOCUS_SCENES = _ordered_unique(list(POLICY.get("mainline_focus_scenes", [])))
MODERN_SIDECAR_SCENES = _ordered_unique(list(POLICY.get("sidecar_scenes", [])))
CANDIDATE_SCENES = _ordered_unique(list(POLICY.get("candidate_scenes", [])))
DEFAULT_SCENES = _ordered_unique(list(POLICY.get("default_scenes", FOCUS_SCENES)))
PRIORITY_SCENES = _ordered_unique(list(POLICY.get("priority_scenes", DEFAULT_SCENES)))
ALL_TRACKED_SCENES = _ordered_unique(
    FOCUS_SCENES + MODERN_SIDECAR_SCENES + CANDIDATE_SCENES + PRIORITY_SCENES
)


def scene_tier(scene_id: str) -> str:
    if scene_id in FOCUS_SCENES:
        return "mainline"
    if scene_id in MODERN_SIDECAR_SCENES:
        return "sidecar"
    if scene_id in CANDIDATE_SCENES:
        return "candidate"
    return "unknown"
