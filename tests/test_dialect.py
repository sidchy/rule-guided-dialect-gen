from wz_pipeline.dialect import ACTIVE_DIALECT_CONFIG


def test_wenzhou_runtime_config_falls_back_to_root_configs() -> None:
    assert ACTIVE_DIALECT_CONFIG.dialect_key == "wenzhou"
    assert ACTIVE_DIALECT_CONFIG.configs_dir == ACTIVE_DIALECT_CONFIG.project_root / "configs"
    assert ACTIVE_DIALECT_CONFIG.scene_policy_path == ACTIVE_DIALECT_CONFIG.project_root / "configs" / "scene_policy.json"


def test_overlay_path_resolution_can_fall_back_to_project_root() -> None:
    resolved = ACTIVE_DIALECT_CONFIG.resolve_overlay_path("configs/scene_policy.json")
    assert resolved == ACTIVE_DIALECT_CONFIG.project_root / "configs" / "scene_policy.json"


def test_wenzhou_runtime_config_uses_tracked_grammar_spec_fallback() -> None:
    assert ACTIVE_DIALECT_CONFIG.grammar_spec_path.exists()
    assert ACTIVE_DIALECT_CONFIG.grammar_spec_path.name.endswith(".md")
