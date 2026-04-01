from wz_pipeline.contracts import apply_contract


def test_generated_candidate_is_not_upstream_eligible() -> None:
    row = apply_contract(
        {"wz_sentence": "测试句", "zh_sentence": "test"},
        source_type="generated_candidate",
        origin_run_id="run_demo",
        pipeline_name="fewshot_batch",
        policy_version="policy_demo",
        scene_id="home_life",
        lane="mainline",
        core_word="测试",
        support_words=["洗衣机"],
    )
    assert row["trust_tier"] == "candidate"
    assert row["rule_gate_status"] == "not_run"
    assert row["human_review_status"] == "pending"
    assert row["upstream_eligible"] is False


def test_extracted_seed_is_upstream_eligible() -> None:
    row = apply_contract(
        {"wz_sentence": "种子句", "zh_sentence": "seed"},
        source_type="extracted_seed",
        origin_run_id="seed_demo",
        pipeline_name="extract_training_sentences",
        policy_version="policy_demo",
        lane="mainline",
        core_word="种子",
        support_words=[],
    )
    assert row["trust_tier"] == "source"
    assert row["human_review_status"] == "not_required"
    assert row["upstream_eligible"] is True

