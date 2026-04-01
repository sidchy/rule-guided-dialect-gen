from wz_pipeline.contracts import apply_contract
from wz_pipeline.training_candidates import build_training_candidates


def test_build_training_candidates_only_keeps_source_and_curated() -> None:
    source_row = apply_contract(
        {"wz_sentence": "种子句", "zh_sentence": "seed"},
        source_type="extracted_seed",
        origin_run_id="seed_run",
        pipeline_name="extract_training_sentences",
        policy_version="policy",
        scene_id="home_life",
        lane="mainline",
        core_word="种子",
        support_words=[],
        sample_prefix="seed",
    )
    curated_row = apply_contract(
        {"wz_sentence": "复核通过句", "zh_sentence": "approved"},
        source_type="reviewed_curated",
        origin_run_id="review_run",
        pipeline_name="fewshot_batch",
        policy_version="policy",
        scene_id="work_study",
        lane="mainline",
        core_word="公司",
        support_words=[],
        sample_prefix="curated",
    )
    rejected_row = apply_contract(
        {"wz_sentence": "候选句", "zh_sentence": "candidate"},
        source_type="generated_candidate",
        origin_run_id="candidate_run",
        pipeline_name="fewshot_batch",
        policy_version="policy",
        scene_id="daily_chat",
        lane="mainline",
        core_word="聊天",
        support_words=[],
        sample_prefix="gen",
    )

    candidates, summary = build_training_candidates([source_row, rejected_row], [curated_row])
    assert [row["sample_id"] for row in candidates] == [source_row["sample_id"], curated_row["sample_id"]]
    assert summary["counts"]["accepted_source_count"] == 1
    assert summary["counts"]["accepted_curated_count"] == 1
    assert summary["counts"]["output_total_count"] == 2


def test_build_training_candidates_dedupes_sample_id() -> None:
    source_row = apply_contract(
        {"wz_sentence": "种子句", "zh_sentence": "seed"},
        source_type="extracted_seed",
        origin_run_id="seed_run",
        pipeline_name="extract_training_sentences",
        policy_version="policy",
        scene_id="home_life",
        lane="mainline",
        core_word="种子",
        support_words=[],
        sample_prefix="seed",
    )
    curated_clone = dict(source_row)
    candidates, summary = build_training_candidates([source_row], [curated_clone])
    assert len(candidates) == 1
    assert summary["counts"]["deduped_count"] == 1
