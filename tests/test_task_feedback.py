import json
import tempfile
from pathlib import Path

import wz_pipeline.task_feedback as feedback_module


def test_resolve_feedback_summary_path_uses_run_id_when_no_path() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        original_runs_dir = feedback_module.RUNS_DIR
        try:
            feedback_module.RUNS_DIR = Path(tmp_dir)
            path = feedback_module.resolve_feedback_summary_path(
                feedback_run_id="run_demo",
                pipeline_name="fewshot_batch",
            )
        finally:
            feedback_module.RUNS_DIR = original_runs_dir

    assert path == Path(tmp_dir) / "fewshot_batch" / "run_demo" / "summary.json"


def test_build_feedback_plan_biases_domain_scenes_for_domain_failures() -> None:
    original_build_domain_context = feedback_module.build_domain_context
    try:
        feedback_module.build_domain_context = lambda domain_ids, scene_id: (
            {"domain_ids": ["medical"]} if scene_id == "health_medical" else {"domain_ids": []}
        )
        plan = feedback_module.build_feedback_plan(
            {
                "run_id": "run_demo",
                "_feedback_summary_path": "/tmp/summary.json",
                "machine_metrics": {
                    "failure_analysis": {
                        "primary_bucket": "domain",
                        "bucket_counts": {"domain": 5, "grammar": 2},
                    }
                },
                "extra": {
                    "scene_rule_pass_counts": {"food_dining": 2, "health_medical": 1},
                },
            },
            scenes=["food_dining", "health_medical"],
            domain_ids=["medical"],
            default_food_modern_trial_ratio=0.25,
        )
    finally:
        feedback_module.build_domain_context = original_build_domain_context

    assert plan["enabled"] is True
    assert plan["primary_bucket"] == "domain"
    assert plan["domain_anchor_pool_top_k"] == 2
    assert plan["scene_weights"]["health_medical"] > plan["scene_weights"]["food_dining"]


def test_build_feedback_plan_tightens_sampling_for_grammar_failures() -> None:
    plan = feedback_module.build_feedback_plan(
        {
            "run_id": "run_demo",
            "_feedback_summary_path": "/tmp/summary.json",
            "machine_metrics": {
                "failure_analysis": {
                    "primary_bucket": "grammar",
                    "bucket_counts": {"grammar": 6},
                }
            },
            "extra": {
                "scene_rule_pass_counts": {"food_dining": 2, "work_study": 0},
            },
        },
        scenes=["food_dining", "work_study"],
        domain_ids=[],
        default_food_modern_trial_ratio=0.3,
    )

    assert plan["anchor_pool_top_k"] == 2
    assert plan["domain_anchor_pool_top_k"] == 2
    assert plan["food_modern_trial_ratio"] == 0.15


def test_load_feedback_summary_reads_json_object() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        summary_path = Path(tmp_dir) / "summary.json"
        summary_path.write_text(json.dumps({"run_id": "run_demo"}), encoding="utf-8")
        payload = feedback_module.load_feedback_summary(feedback_summary_path=summary_path)

    assert payload["run_id"] == "run_demo"
    assert payload["_feedback_summary_path"] == str(summary_path)
