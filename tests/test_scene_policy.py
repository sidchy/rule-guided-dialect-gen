from wz_pipeline.scene_policy import (
    ALL_TRACKED_SCENES,
    CANDIDATE_SCENES,
    DEFAULT_SCENES,
    FOCUS_SCENES,
    MODERN_SIDECAR_SCENES,
    PRIORITY_SCENES,
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
