from __future__ import annotations

import argparse
from pathlib import Path

from wz_pipeline.jsonl import read_jsonl, write_jsonl
from wz_pipeline.paths import CURATED_DIR
from wz_pipeline.runs import write_json
from wz_pipeline.training_candidates import build_training_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training candidates from curated source and generated pools.")
    parser.add_argument(
        "--seed-input",
        type=Path,
        default=CURATED_DIR / "seed_sentences.jsonl",
    )
    parser.add_argument(
        "--generated-input",
        type=Path,
        default=CURATED_DIR / "generated_reviewed.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CURATED_DIR / "training_candidates.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=CURATED_DIR / "training_candidates_summary.json",
    )
    args = parser.parse_args()

    candidates, summary = build_training_candidates(
        read_jsonl(args.seed_input),
        read_jsonl(args.generated_input),
    )
    summary["config"] = {
        "seed_input": str(args.seed_input),
        "generated_input": str(args.generated_input),
        "output": str(args.output),
        "summary_output": str(args.summary_output),
    }
    write_jsonl(args.output, candidates)
    write_json(args.summary_output, summary)
    print(f"Wrote training candidates: {args.output}")
    print(f"Wrote summary: {args.summary_output}")
    print(f"Output rows: {len(candidates)}")
