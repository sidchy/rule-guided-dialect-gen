#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wz_pipeline.cli.import_review_results import main


if __name__ == "__main__":
    main()

