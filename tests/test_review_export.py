from wz_pipeline.review import (
    build_grammar_review_metadata,
    exclude_rows_with_terms,
    filter_review_rows,
    limit_review_rows_per_scene,
    parse_selector_arg,
    prepare_review_rows,
)


def test_parse_selector_arg_handles_csv() -> None:
    assert parse_selector_arg("work_study,daily_chat") == {"work_study", "daily_chat"}
    assert parse_selector_arg("") == set()


def test_filter_review_rows_respects_scene_and_lane() -> None:
    rows = [
        {"scene_id": "work_study", "lane": "", "rule_gate_status": "pass"},
        {"scene_id": "daily_chat", "lane": "", "rule_gate_status": "pass"},
        {"scene_id": "work_study", "lane": "mainline", "rule_gate_status": "fail"},
    ]
    filtered = filter_review_rows(
        rows,
        scenes={"work_study"},
        lanes={""},
        rule_pass_only=True,
    )
    assert filtered == [{"scene_id": "work_study", "lane": "", "rule_gate_status": "pass"}]


def test_limit_review_rows_per_scene_caps_each_scene() -> None:
    rows = [
        {"scene_id": "work_study", "lane": "", "core_word": f"词{i}", "rule_gate_status": "pass"}
        for i in range(5)
    ] + [
        {"scene_id": "daily_chat", "lane": "", "core_word": f"聊{i}", "rule_gate_status": "pass"}
        for i in range(4)
    ]
    limited = limit_review_rows_per_scene(rows, limit_per_scene=2, seed=7, rule_pass_only=False)
    assert len([row for row in limited if row["scene_id"] == "work_study"]) == 2
    assert len([row for row in limited if row["scene_id"] == "daily_chat"]) == 2


def test_prepare_review_rows_fills_stable_sample_id() -> None:
    prepared = prepare_review_rows(
        [
            {
                "scene_id": "work_study",
                "lane": "",
                "core_word": "公司",
                "support_words": ["电脑"],
                "wz_sentence": "公司里开会。",
                "zh_sentence": "公司里开会。",
            }
        ],
        origin_run_id="legacy_work_study",
        pipeline_name="fewshot_batch_legacy",
        sample_prefix="gen",
        default_rule_gate_status="pass",
    )
    assert prepared[0]["sample_id"].startswith("gen_")
    assert prepared[0]["rule_gate_status"] == "pass"


def test_prepare_review_rows_derives_core_and_support_from_target_words() -> None:
    prepared = prepare_review_rows(
        [
            {
                "scene_id": "health_medical",
                "target_words": ["医生", "看病", "药水"],
                "wz_sentence": "我去寻医生看病。",
                "zh_sentence": "我去找医生看病。",
            }
        ],
        origin_run_id="legacy_health_medical",
        pipeline_name="fewshot_batch_legacy",
        sample_prefix="gen",
        default_rule_gate_status="pass",
    )
    assert prepared[0]["core_word"] == "医生"
    assert prepared[0]["support_words"] == ["看病", "药水"]


def test_exclude_rows_with_terms_filters_surface_contaminants() -> None:
    rows = [
        {
            "scene_id": "transport_trip",
            "wz_sentence": "料车高头个车钻寻不着。",
            "target_words": ["料车", "车钻"],
        },
        {
            "scene_id": "transport_trip",
            "wz_sentence": "料车里个车钻寻不着。",
            "target_words": ["料车", "车钻"],
        },
    ]
    filtered = exclude_rows_with_terms(rows, {"高头"})
    assert filtered == [rows[1]]


def test_build_grammar_review_metadata_highlights_particles() -> None:
    metadata = build_grammar_review_metadata(
        {
            "wz_sentence": "饭吃爻罢，渠还未走归。",
            "validation": {"grammar_reasons": ["sentence_final_ba_overused"]},
        }
    )
    assert metadata["grammar_markers"] == "爻|罢|未"
    assert metadata["grammar_auto_flags"] == "sentence_final_ba_overused"
    assert "3.1 完成体：`爻`" in metadata["grammar_spec_sections"]
    assert "3.4 已然体：重读 `罢` 与句末轻读 `罢`" in metadata["grammar_spec_sections"]
