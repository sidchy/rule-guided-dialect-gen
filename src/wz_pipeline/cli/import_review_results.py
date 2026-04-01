from __future__ import annotations

import argparse
from pathlib import Path

from wz_pipeline.review import import_review_tsv, write_review_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Import filled review TSV into structured JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin-run-id", required=True)
    parser.add_argument("--pipeline-name", required=True)
    parser.add_argument("--policy-version", required=True)
    args = parser.parse_args()

    rows = import_review_tsv(
        args.input,
        origin_run_id=args.origin_run_id,
        pipeline_name=args.pipeline_name,
        policy_version=args.policy_version,
    )
    write_review_results(args.output, rows)
    print(f"Wrote review results: {args.output}")


if __name__ == "__main__":
    main()

