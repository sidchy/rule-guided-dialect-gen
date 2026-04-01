from pathlib import Path

from wz_pipeline.promotion import build_promotion_candidates, promote_generated_rows
from wz_pipeline.review import import_review_tsv


def test_import_review_tsv_normalizes_chinese_status(tmp_path: Path) -> None:
    review_path = tmp_path / "review.tsv"
    review_path.write_text(
        "\t".join(
            [
                "sample_id",
                "scene_id",
                "lane",
                "core_word",
                "support_words",
                "wz_sentence",
                "zh_sentence",
                "rule_gate_status",
                "human_review_status",
                "review_reason",
                "reviewer",
                "reviewed_at",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "sample_1",
                "home_life",
                "mainline",
                "大蛮阵",
                "洗衣机",
                "屋里大蛮阵。",
                "家里人口多。",
                "pass",
                "通过",
                "自然",
                "tester",
                "2026-04-01T12:00:00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = import_review_tsv(
        review_path,
        origin_run_id="legacy_demo",
        pipeline_name="fewshot_batch_legacy",
        policy_version="legacy_policy",
    )
    assert rows[0]["human_review_status"] == "pass"
    assert rows[0]["trust_tier"] == "reviewed"


def test_only_rule_pass_and_human_pass_are_promotable(tmp_path: Path) -> None:
    rows = [
        {
            "sample_id": "sample_a",
            "origin_run_id": "legacy_demo",
            "pipeline_name": "fewshot_batch_legacy",
            "policy_version": "legacy_policy",
            "scene_id": "home_life",
            "lane": "mainline",
            "core_word": "大蛮阵",
            "support_words": ["洗衣机"],
            "wz_sentence": "屋里大蛮阵。",
            "zh_sentence": "家里人口多。",
            "rule_gate_status": "pass",
            "human_review_status": "pass",
            "trust_tier": "reviewed",
            "upstream_eligible": False,
        },
        {
            "sample_id": "sample_b",
            "origin_run_id": "legacy_demo",
            "pipeline_name": "fewshot_batch_legacy",
            "policy_version": "legacy_policy",
            "scene_id": "home_life",
            "lane": "mainline",
            "core_word": "大蛮阵",
            "support_words": [],
            "wz_sentence": "屋里大蛮阵，真困难。",
            "zh_sentence": "家里人口多，很困难。",
            "rule_gate_status": "pass",
            "human_review_status": "pending",
            "trust_tier": "candidate",
            "upstream_eligible": False,
        },
    ]
    candidates = build_promotion_candidates(rows)
    assert [row["sample_id"] for row in candidates] == ["sample_a"]

    output_path = tmp_path / "generated_reviewed.jsonl"
    promoted = promote_generated_rows(
        candidates,
        output_path=output_path,
        policy_version="curated_policy",
    )
    assert len(promoted) == 1
    assert promoted[0]["trust_tier"] == "curated"
    assert promoted[0]["upstream_eligible"] is True


def test_import_review_tsv_preserves_rule_gate_status(tmp_path: Path) -> None:
    review_path = tmp_path / "review.tsv"
    review_path.write_text(
        "\t".join(
            [
                "sample_id",
                "scene_id",
                "lane",
                "core_word",
                "support_words",
                "wz_sentence",
                "zh_sentence",
                "rule_gate_status",
                "human_review_status",
                "review_reason",
                "reviewer",
                "reviewed_at",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "sample_fail",
                "food_dining",
                "mainline",
                "涌烫",
                "煮饭",
                "汤还涌烫。",
                "汤还很烫。",
                "fail",
                "通过",
                "人工认可但规则未过",
                "tester",
                "2026-04-01T12:00:00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = import_review_tsv(
        review_path,
        origin_run_id="legacy_demo",
        pipeline_name="fewshot_batch_legacy",
        policy_version="legacy_policy",
    )
    assert rows[0]["rule_gate_status"] == "fail"
    assert build_promotion_candidates(rows) == []


def test_promote_generated_rows_is_idempotent(tmp_path: Path) -> None:
    row = {
        "sample_id": "sample_curated_once",
        "origin_run_id": "legacy_demo",
        "pipeline_name": "fewshot_batch_legacy",
        "policy_version": "legacy_policy",
        "scene_id": "home_life",
        "lane": "mainline",
        "core_word": "大蛮阵",
        "support_words": ["洗衣机"],
        "wz_sentence": "屋里大蛮阵。",
        "zh_sentence": "家里人口多。",
        "rule_gate_status": "pass",
        "human_review_status": "pass",
        "trust_tier": "reviewed",
        "upstream_eligible": False,
    }
    output_path = tmp_path / "generated_reviewed.jsonl"
    first = promote_generated_rows([row], output_path=output_path, policy_version="curated_policy")
    second = promote_generated_rows([row], output_path=output_path, policy_version="curated_policy")
    assert len(first) == 1
    assert second == []
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
