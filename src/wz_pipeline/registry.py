from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .jsonl import append_jsonl
from .paths import REGISTRY_DIR, ensure_dir


RUN_REGISTRY_PATH = REGISTRY_DIR / "run_registry.jsonl"
LEGACY_REGISTRY_PATH = REGISTRY_DIR / "legacy_runs.jsonl"
PROMOTION_LOG_PATH = REGISTRY_DIR / "promotion_log.jsonl"


def register_run(entry: dict[str, Any]) -> None:
    ensure_dir(REGISTRY_DIR)
    append_jsonl(RUN_REGISTRY_PATH, [dict(entry, registered_at=datetime.now().isoformat())])


def register_legacy_run(entry: dict[str, Any]) -> None:
    ensure_dir(REGISTRY_DIR)
    append_jsonl(LEGACY_REGISTRY_PATH, [dict(entry, registered_at=datetime.now().isoformat())])


def register_promotion(entry: dict[str, Any]) -> None:
    ensure_dir(REGISTRY_DIR)
    append_jsonl(PROMOTION_LOG_PATH, [dict(entry, promoted_at=datetime.now().isoformat())])

