from __future__ import annotations

import argparse
import json
from pathlib import Path

from wz_pipeline.registry import register_legacy_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a legacy run in the registry.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--pipeline-name", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--notes", default="")
    parser.add_argument("--source-type", default="generated_candidate")
    args = parser.parse_args()

    summary_payload = {}
    if args.summary and args.summary.exists():
        summary_payload = json.loads(args.summary.read_text(encoding="utf-8"))

    register_legacy_run(
        {
            "name": args.name,
            "pipeline_name": args.pipeline_name,
            "results": str(args.results),
            "review": str(args.review) if args.review else "",
            "summary": str(args.summary) if args.summary else "",
            "notes": args.notes,
            "source_type": args.source_type,
            "summary_payload": summary_payload,
        }
    )
    print(f"Registered legacy run: {args.name}")


if __name__ == "__main__":
    main()

