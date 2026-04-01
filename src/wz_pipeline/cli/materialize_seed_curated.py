from __future__ import annotations

import argparse
from pathlib import Path

from wz_pipeline.contracts import apply_contract
from wz_pipeline.jsonl import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize extracted seeds into curated source pool.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin-run-id", default="seed_materialization")
    parser.add_argument("--policy-version", default="source_seed_v1")
    args = parser.parse_args()

    rows = []
    for row in read_jsonl(args.input):
        rows.append(
            apply_contract(
                row,
                source_type="extracted_seed",
                origin_run_id=args.origin_run_id,
                pipeline_name="extract_training_sentences",
                policy_version=args.policy_version,
                scene_id=str(row.get("scene_id", "")),
                lane="mainline",
                core_word=str(row.get("wz_word", "")),
                support_words=[],
                sample_prefix="seed",
            )
        )
    write_jsonl(args.output, rows)
    print(f"Wrote curated seed pool: {args.output}")


if __name__ == "__main__":
    main()

