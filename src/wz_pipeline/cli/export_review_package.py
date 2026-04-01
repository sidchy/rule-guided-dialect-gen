from __future__ import annotations

import argparse
import json
from pathlib import Path

from wz_pipeline.paths import ROOT
from wz_pipeline.jsonl import read_jsonl, write_jsonl
from wz_pipeline.review import (
    exclude_rows_with_terms,
    export_review_tsv,
    filter_review_rows,
    limit_review_rows_per_scene,
    parse_selector_arg,
    prepare_review_rows,
    row_mentions_any_term,
    select_review_rows,
)


FEWSHOT_CORE_POLICY = ROOT / "data" / "controlled_generation" / "assets" / "fewshot_core_policy.json"


def load_core_policy() -> dict:
    if not FEWSHOT_CORE_POLICY.exists():
        return {"global": {}, "scene_policies": {}}
    return json.loads(FEWSHOT_CORE_POLICY.read_text(encoding="utf-8"))


def row_deny_terms(row: dict, core_policy: dict) -> set[str]:
    scene_id = str(row.get("scene_id", "")).strip()
    global_terms = set(core_policy.get("global", {}).get("deny_terms", []))
    scene_terms = set(core_policy.get("scene_policies", {}).get(scene_id, {}).get("deny_terms", []))
    return global_terms | scene_terms


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a standard review TSV from rule-gated rows.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-jsonl", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit-per-scene", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--origin-run-id", default="")
    parser.add_argument("--pipeline-name", default="")
    parser.add_argument("--sample-prefix", default="gen")
    parser.add_argument("--scene", default="", help="Comma-separated scene ids to include.")
    parser.add_argument("--lane", default="", help="Comma-separated lane ids to include.")
    parser.add_argument(
        "--include-rule-fail",
        action="store_true",
        help="Include rows whose rule_gate_status is not pass.",
    )
    parser.add_argument(
        "--exclude-core-policy-deny-terms",
        action="store_true",
        help="Drop rows that already contain global or scene deny terms such as northern-Wu contaminants.",
    )
    parser.add_argument(
        "--exclude-terms",
        default="",
        help="Comma-separated extra surface terms to exclude from the review package.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    filtered = filter_review_rows(
        rows,
        scenes=parse_selector_arg(args.scene),
        lanes=parse_selector_arg(args.lane),
        rule_pass_only=not args.include_rule_fail,
    )
    extra_exclude_terms = parse_selector_arg(args.exclude_terms)
    if args.exclude_core_policy_deny_terms:
        core_policy = load_core_policy()
        filtered = [
            row
            for row in filtered
            if not row_mentions_any_term(row, row_deny_terms(row, core_policy) | extra_exclude_terms)
        ]
    elif extra_exclude_terms:
        filtered = exclude_rows_with_terms(filtered, extra_exclude_terms)
    if args.limit_per_scene > 0:
        filtered = limit_review_rows_per_scene(
            filtered,
            limit_per_scene=args.limit_per_scene,
            seed=args.seed,
            rule_pass_only=False,
        )
    selected = select_review_rows(
        filtered,
        limit=args.limit if args.limit > 0 else None,
        seed=args.seed,
        rule_pass_only=False,
    )
    prepared = prepare_review_rows(
        selected,
        origin_run_id=args.origin_run_id,
        pipeline_name=args.pipeline_name,
        sample_prefix=args.sample_prefix,
        default_rule_gate_status="not_run" if args.include_rule_fail else "pass",
    )
    export_review_tsv(args.output, prepared)
    if args.selected_jsonl:
        write_jsonl(args.selected_jsonl, prepared)
    print(f"Wrote review package: {args.output}")
    print(f"Selected rows: {len(prepared)}")


if __name__ == "__main__":
    main()
