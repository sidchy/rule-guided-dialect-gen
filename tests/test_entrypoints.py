import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_review_generation_naturalness_help_runs_from_checkout() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/review_generation_naturalness.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_fewshot_batch_uses_empty_scene_terms_for_unconfigured_dialect() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["WZ_PIPELINE_DIALECT"] = "foo"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from wz_pipeline.pipelines import fewshot_batch as f; "
                "print(json.dumps({"
                "'shopping_prompt_blocked_terms': sorted(f.SHOPPING_PROMPT_BLOCKED_TERMS), "
                "'food_support_priority_terms': sorted(f.FOOD_SUPPORT_PRIORITY_TERMS)"
                "}, ensure_ascii=False))"
            ),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["shopping_prompt_blocked_terms"] == []
    assert payload["food_support_priority_terms"] == []
