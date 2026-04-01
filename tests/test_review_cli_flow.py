import csv
import json
import sys
import tempfile
from pathlib import Path

from wz_pipeline.cli import export_review_package as export_cli
from wz_pipeline.cli import import_review_results as import_cli
from wz_pipeline.jsonl import read_jsonl, write_jsonl


def test_export_review_package_cli_writes_extended_review_columns() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_path = root / "rule_gate.jsonl"
        output_path = root / "review.tsv"
        write_jsonl(
            input_path,
            [
                {
                    "sample_id": "gen_demo",
                    "scene_id": "health_medical",
                    "lane": "mainline",
                    "core_word": "挂号",
                    "support_words": ["门诊"],
                    "domain_ids": ["medical"],
                    "domain_required_terms": ["挂号"],
                    "domain_blocked_terms": ["偏方"],
                    "domain_review_notes": ["先看医疗词自然度"],
                    "wz_sentence": "医院门诊挂号看病。",
                    "zh_sentence": "在医院门诊挂号看病。",
                    "rule_gate_status": "pass",
                    "validation": {
                        "reasons": ["domain_required_terms_missing", "grammar:particle_misuse"],
                        "grammar_reasons": ["particle_misuse"],
                        "grammar_spec_sections": ["3.1 完成体：`爻`"],
                    },
                }
            ],
        )
        original_argv = sys.argv
        try:
            sys.argv = [
                "wz-export-review-package",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--origin-run-id",
                "run_demo",
                "--pipeline-name",
                "fewshot_batch",
            ]
            export_cli.main()
        finally:
            sys.argv = original_argv

        with output_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rows = list(reader)

    assert "failure_buckets" in reader.fieldnames
    assert "failure_primary_bucket" in reader.fieldnames
    assert "review_focus" in reader.fieldnames
    assert rows[0]["failure_buckets"] == "grammar|domain"
    assert rows[0]["failure_primary_bucket"] == "grammar"
    assert rows[0]["review_focus"] == "grammar|domain"


def test_import_review_results_cli_ignores_extended_review_columns() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_path = root / "review.tsv"
        output_path = root / "review_results.jsonl"
        input_path.write_text(
            "\t".join(
                [
                    "sample_id",
                    "scene_id",
                    "lane",
                    "core_word",
                    "support_words",
                    "domain_ids",
                    "failure_buckets",
                    "failure_primary_bucket",
                    "review_focus",
                    "wz_sentence",
                    "zh_sentence",
                    "rule_gate_status",
                    "grammar_markers",
                    "grammar_auto_flags",
                    "grammar_spec_sections",
                    "human_review_status",
                    "review_reason",
                    "reviewer",
                    "reviewed_at",
                ]
            )
            + "\n"
            + "\t".join(
                [
                    "sample_demo",
                    "health_medical",
                    "mainline",
                    "挂号",
                    "门诊",
                    "medical",
                    "grammar|domain",
                    "grammar",
                    "grammar|domain",
                    "医院门诊挂号看病。",
                    "在医院门诊挂号看病。",
                    "pass",
                    "爻",
                    "particle_misuse",
                    "3.1 完成体：`爻`",
                    "通过",
                    "自然",
                    "tester",
                    "2026-04-01T12:00:00",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        original_argv = sys.argv
        try:
            sys.argv = [
                "wz-import-review-results",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--origin-run-id",
                "run_demo",
                "--pipeline-name",
                "fewshot_batch",
                "--policy-version",
                "policy_demo",
            ]
            import_cli.main()
        finally:
            sys.argv = original_argv

        rows = read_jsonl(output_path)

    assert len(rows) == 1
    assert rows[0]["sample_id"] == "sample_demo"
    assert rows[0]["rule_gate_status"] == "pass"
    assert rows[0]["human_review_status"] == "pass"
    assert rows[0]["trust_tier"] == "reviewed"
