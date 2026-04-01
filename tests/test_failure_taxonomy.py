import importlib.util
import sys
from pathlib import Path

from wz_pipeline.failure_taxonomy import (
    classify_critic_fail_buckets,
    classify_rule_fail_buckets,
    classify_rule_reason_bucket,
    reason_key,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_review_script_module():
    scripts_dir = ROOT / "scripts"
    scripts_path = str(scripts_dir)
    remove_after = False
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
        remove_after = True
    spec = importlib.util.spec_from_file_location(
        "review_generation_naturalness",
        scripts_dir / "review_generation_naturalness.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        if remove_after:
            sys.path.remove(scripts_path)


def test_reason_key_strips_reason_payload() -> None:
    assert reason_key("grammar:aspect_misuse") == "grammar"
    assert reason_key("domain_blocked_terms:银圆") == "domain_blocked_terms"


def test_classify_rule_reason_bucket_maps_shared_taxonomy() -> None:
    assert classify_rule_reason_bucket("grammar:aspect_misuse") == "grammar"
    assert classify_rule_reason_bucket("domain_required_terms_missing") == "domain"
    assert classify_rule_reason_bucket("low_wz_word_coverage:1") == "naturalness"


def test_classify_rule_fail_buckets_deduplicates_and_orders() -> None:
    assert classify_rule_fail_buckets(
        [
            "low_wz_word_coverage:1",
            "grammar:aspect_misuse",
            "domain_required_terms_missing",
            "grammar:particle_misuse",
        ]
    ) == ["grammar", "domain", "naturalness"]


def test_classify_critic_fail_buckets_uses_three_bucket_view() -> None:
    assert classify_critic_fail_buckets(
        critic_pass=False,
        critic_issue_types=["unnatural_wz"],
        critic_grammar_pass=False,
        critic_grammar_issue_types=["particle_misuse"],
        critic_domain_pass=True,
        critic_domain_issue_types=[],
    ) == ["grammar", "naturalness"]


def test_critic_summary_excludes_request_failures_from_bucket_counts() -> None:
    review_module = _load_review_script_module()
    summary = review_module.build_summary_payload(
        [
            {
                "critic_pass": False,
                "critic_issue_types": ["unnatural_wz"],
                "critic_grammar_issue_types": ["particle_misuse"],
                "critic_domain_issue_types": [],
                "critic_fail_buckets": ["grammar", "naturalness"],
                "critic_primary_fail_bucket": "grammar",
            },
            {
                "critic_pass": False,
                "critic_issue_types": ["critic_request_failed"],
                "critic_grammar_issue_types": ["critic_request_failed"],
                "critic_domain_issue_types": ["critic_request_failed"],
                "critic_fail_buckets": [],
                "critic_primary_fail_bucket": "",
                "critic_request_failed": True,
            },
        ],
        input_rows=2,
        failed_request_rows=1,
        provider="qwen",
        model="demo",
        output_path=ROOT / "tmp.jsonl",
    )

    assert summary["critic_fail_bucket_counts"] == {"grammar": 1, "naturalness": 1}
    assert summary["critic_primary_fail_bucket_counts"] == {"grammar": 1}
