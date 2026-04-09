import wz_pipeline.scene_policy as scene_policy_module
from wz_pipeline.scene_policy import (
    ALL_TRACKED_SCENES,
    CANDIDATE_SCENES,
    DEFAULT_SCENES,
    FOCUS_SCENES,
    MODERN_SIDECAR_SCENES,
    PRIORITY_SCENES,
    load_scene_catalog,
    scene_ancestors,
    scene_parent,
    scene_tier,
)


def test_default_scenes_match_mainline_focus() -> None:
    assert DEFAULT_SCENES == FOCUS_SCENES


def test_scene_groups_do_not_overlap() -> None:
    assert not (set(FOCUS_SCENES) & set(MODERN_SIDECAR_SCENES))
    assert not (set(FOCUS_SCENES) & set(CANDIDATE_SCENES))
    assert not (set(MODERN_SIDECAR_SCENES) & set(CANDIDATE_SCENES))


def test_priority_scenes_cover_all_tracked_scenes() -> None:
    assert set(ALL_TRACKED_SCENES).issubset(set(PRIORITY_SCENES))


def test_scene_tier_matches_policy() -> None:
    assert scene_tier("home_life") == "mainline"
    assert scene_tier("digital_chat") == "sidecar"
    assert scene_tier("daily_chat") == "candidate"
    assert scene_tier("dig_phone") == "unknown"
    assert scene_tier("social_friends") == "unknown"


def test_catalog_active_scenes_exclude_candidate_descendants() -> None:
    assert "dig_phone" not in FOCUS_SCENES
    assert "social_friends" not in FOCUS_SCENES


def test_load_scene_catalog_contains_runtime_hierarchy() -> None:
    scene_by_id = {scene["scene_id"]: scene for scene in load_scene_catalog()}
    assert scene_by_id["food_grocery"]["parent_scene_id"] == "food_dining"


def test_scene_parent_resolves_catalog_and_special_overrides() -> None:
    assert scene_parent("food_grocery") == "food_dining"
    assert scene_parent("digital_ai_assistant") == "digital_chat"


def test_scene_ancestors_return_nearest_first() -> None:
    assert scene_ancestors("food_grocery") == ["food_dining"]
    assert scene_ancestors("digital_ai_assistant") == ["digital_chat"]


def test_scene_ancestors_stop_on_cycles(monkeypatch) -> None:
    parent_map = {"loop_a": "loop_b", "loop_b": "loop_c", "loop_c": "loop_b"}
    monkeypatch.setattr(
        scene_policy_module,
        "scene_parent",
        lambda scene_id: parent_map.get(scene_id, ""),
    )
    assert scene_policy_module.scene_ancestors("loop_a") == ["loop_b", "loop_c"]
