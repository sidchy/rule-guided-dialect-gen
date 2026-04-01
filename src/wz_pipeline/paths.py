from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"
REGISTRY_DIR = ROOT / "registry"
RUNS_DIR = ROOT / "runs"
CURATED_DIR = ROOT / "curated"
DOCS_DIR = ROOT / "docs"
SCRIPTS_DIR = ROOT / "scripts"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

