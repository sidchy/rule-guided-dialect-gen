from __future__ import annotations

import argparse
from pathlib import Path

from wz_pipeline.jsonl import read_jsonl
from wz_pipeline.promotion import build_promotion_candidates, promote_generated_rows, write_promotion_candidates
from wz_pipeline.registry import register_promotion


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote reviewed generated samples into curated pool.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--promotion-candidates-output", type=Path, required=True)
    parser.add_argument("--curated-output", type=Path, required=True)
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--origin-run-id", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    candidates = build_promotion_candidates(rows)
    write_promotion_candidates(args.promotion_candidates_output, candidates)
    promoted = promote_generated_rows(
        candidates,
        output_path=args.curated_output,
        policy_version=args.policy_version,
    )
    register_promotion(
        {
            "origin_run_id": args.origin_run_id,
            "input_review_results": str(args.input),
            "promotion_candidates_output": str(args.promotion_candidates_output),
            "curated_output": str(args.curated_output),
            "promoted_count": len(promoted),
        }
    )
    print(f"Promotion candidates: {len(candidates)}")
    print(f"Promoted rows: {len(promoted)}")


if __name__ == "__main__":
    main()

