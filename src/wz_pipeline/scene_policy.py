from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from .paths import SCENE_CATALOG_PATH, SCENE_POLICY_PATH

DIGITAL_PARENT_SCENE = "digital_chat"
DIGITAL_SUBSCENES = (
    "digital_ai_assistant",
    "digital_messaging_call",
    "digital_device_trouble",
    "digital_app_operation",
)
SPECIAL_PARENT_OVERRIDES: dict[str, str] = {
    scene_id: DIGITAL_PARENT_SCENE
    for scene_id in DIGITAL_SUBSCENES
}


@lru_cache(maxsize=1)
def load_scene_policy() -> dict[str, Any]:
    return json.loads(SCENE_POLICY_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_scene_catalog() -> list[dict[str, Any]]:
    if not SCENE_CATALOG_PATH.exists():
        return []
    payload = json.loads(SCENE_CATALOG_PATH.read_text(encoding="utf-8"))
    scenes = payload.get("scenes", [])
    if not isinstance(scenes, list):
        return []
    return [
        scene
        for scene in scenes
        if isinstance(scene, dict) and str(scene.get("scene_id") or "").strip()
    ]


@lru_cache(maxsize=1)
def _scene_parent_index() -> dict[str, str]:
    parents = dict(SPECIAL_PARENT_OVERRIDES)
    for scene in load_scene_catalog():
        scene_id = str(scene.get("scene_id") or "").strip()
        parent_scene_id = str(scene.get("parent_scene_id") or "").strip()
        if scene_id and parent_scene_id and scene_id not in parents:
            parents[scene_id] = parent_scene_id
    return parents


def scene_parent(scene_id: str) -> str:
    return _scene_parent_index().get(str(scene_id or "").strip(), "")


def scene_ancestors(scene_id: str) -> list[str]:
    clean_scene_id = str(scene_id or "").strip()
    seen = {clean_scene_id} if clean_scene_id else set()
    ordered: list[str] = []
    current = scene_parent(clean_scene_id)
    while current and current not in seen:
        ordered.append(current)
        seen.add(current)
        current = scene_parent(current)
    return ordered


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
