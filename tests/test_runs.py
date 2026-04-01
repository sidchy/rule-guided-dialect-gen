from wz_pipeline.runs import build_failure_analysis, build_summary


def test_build_failure_analysis_summarizes_top_buckets_and_reasons() -> None:
    analysis = build_failure_analysis(
        {
            "rule_fail_count": 9,
            "rule_fail_bucket_counts": {"naturalness": 2, "grammar": 5, "domain": 2},
            "grammar_fail_counts": {"particle_misuse": 4, "aspect_misuse": 2},
            "domain_fail_counts": {"domain_required_terms_missing": 3},
            "naturalness_fail_counts": {"low_wz_word_coverage": 6, "duplicate": 1},
        }
    )
    assert analysis["rule_fail_count"] == 9
    assert analysis["bucket_counts"] == {"naturalness": 2, "grammar": 5, "domain": 2}
    assert analysis["primary_bucket"] == "grammar"
    assert analysis["top_grammar_reasons"][0] == {"name": "particle_misuse", "count": 4}
    assert analysis["top_domain_reasons"][0] == {"name": "domain_required_terms_missing", "count": 3}
    assert analysis["top_naturalness_reasons"][0] == {"name": "low_wz_word_coverage", "count": 6}


def test_build_summary_embeds_failure_analysis_into_machine_metrics() -> None:
    summary = build_summary(
        pipeline_name="fewshot_batch",
        run_id="run_demo",
        config={},
        raw_count=10,
        rule_pass_count=7,
        extra={
            "rule_fail_count": 3,
            "rule_fail_bucket_counts": {"grammar": 2, "domain": 1},
            "grammar_fail_counts": {"particle_misuse": 2},
            "domain_fail_counts": {"domain_required_terms_missing": 1},
        },
    )
    assert summary["machine_metrics"]["rule_pass_rate"] == 0.7
    assert summary["machine_metrics"]["failure_analysis"]["primary_bucket"] == "grammar"
    assert summary["machine_metrics"]["failure_analysis"]["bucket_counts"] == {
        "grammar": 2,
        "domain": 1,
    }


def test_build_summary_keeps_failure_analysis_when_no_failures() -> None:
    summary = build_summary(
        pipeline_name="fewshot_batch",
        run_id="run_demo",
        config={},
        raw_count=3,
        rule_pass_count=3,
        extra={},
    )
    assert summary["machine_metrics"]["failure_analysis"] == {
        "rule_fail_count": 0,
        "bucket_counts": {},
        "primary_bucket": "",
        "top_grammar_reasons": [],
        "top_domain_reasons": [],
        "top_naturalness_reasons": [],
    }
