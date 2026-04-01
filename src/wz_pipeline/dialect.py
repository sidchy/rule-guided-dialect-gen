from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIALECT = "wenzhou"
DIALECT_ENV_VAR = "WZ_PIPELINE_DIALECT"
WENZHOU_FALLBACK_GRAMMAR_SPEC = PROJECT_ROOT / "docs" / "温州话生成语法规范_潘悟云原书67-83页.md"
WENZHOU_FALLBACK_SCENE_CATALOG = (
    PROJECT_ROOT / "data" / "controlled_generation" / "assets" / "scene_catalog.json"
)


def active_dialect() -> str:
    return str(os.environ.get(DIALECT_ENV_VAR) or DEFAULT_DIALECT).strip() or DEFAULT_DIALECT


def _pick_existing(primary: Path, fallback: Path | None = None) -> Path:
    if primary.exists():
        return primary
    if fallback is not None and fallback.exists():
        return fallback
    return primary


@dataclass(frozen=True)
class DialectRuntimeConfig:
    dialect_key: str
    dialect_dir: Path
    project_root: Path
    raw: dict[str, Any]
    dialect_name: str
    dialect_code: str
    standard_language_label: str
    configs_dir: Path
    data_dir: Path
    runs_dir: Path
    registry_dir: Path
    curated_dir: Path
    prompts_dir: Path
    scene_policy_path: Path
    trust_policy_path: Path
    pipeline_thresholds_path: Path
    grammar_spec_path: Path
    grammar_rules_path: Path
    scene_catalog_path: Path

    def resolve_overlay_path(self, value: str | Path) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        local = (self.dialect_dir / candidate).resolve()
        root_relative = (self.project_root / candidate).resolve()
        if local.exists():
            return local
        if root_relative.exists():
            return root_relative
        return local


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Dialect config must be a mapping: {path}")
    return payload


@lru_cache(maxsize=None)
def load_dialect_config(dialect: str | None = None) -> DialectRuntimeConfig:
    dialect_key = (dialect or active_dialect()).strip() or DEFAULT_DIALECT
    dialect_dir = PROJECT_ROOT / "dialects" / dialect_key
    raw = _load_yaml(dialect_dir / "dialect.yaml")

    configs_dir = _pick_existing(dialect_dir / str(raw.get("configs_dir") or "configs"), PROJECT_ROOT / "configs")
    data_dir = _pick_existing(dialect_dir / str(raw.get("data_dir") or "data"), PROJECT_ROOT / "data")
    runs_dir = _pick_existing(dialect_dir / str(raw.get("runs_dir") or "runs"), PROJECT_ROOT / "runs")
    registry_dir = _pick_existing(
        dialect_dir / str(raw.get("registry_dir") or "registry"),
        PROJECT_ROOT / "registry",
    )
    curated_dir = _pick_existing(
        dialect_dir / str(raw.get("curated_dir") or "curated"),
        PROJECT_ROOT / "curated",
    )
    prompts_dir = _pick_existing(
        dialect_dir / str(raw.get("prompts_dir") or "prompts"),
        PROJECT_ROOT / "prompts",
    )

    grammar_spec_candidate = dialect_dir / str(raw.get("grammar_spec") or "grammar/grammar_spec.md")
    grammar_spec_fallback = WENZHOU_FALLBACK_GRAMMAR_SPEC if dialect_key == "wenzhou" else None
    grammar_spec_path = _pick_existing(grammar_spec_candidate, grammar_spec_fallback)

    grammar_rules_path = dialect_dir / str(raw.get("grammar_rules") or "grammar/grammar_rules.yaml")
    scene_catalog_candidate = dialect_dir / str(raw.get("scene_catalog") or "configs/scene_catalog.json")
    scene_catalog_fallback = WENZHOU_FALLBACK_SCENE_CATALOG if dialect_key == "wenzhou" else None
    scene_catalog_path = _pick_existing(scene_catalog_candidate, scene_catalog_fallback)

    return DialectRuntimeConfig(
        dialect_key=dialect_key,
        dialect_dir=dialect_dir,
        project_root=PROJECT_ROOT,
        raw=raw,
        dialect_name=str(raw.get("dialect_name") or dialect_key),
        dialect_code=str(raw.get("dialect_code") or dialect_key),
        standard_language_label=str(raw.get("standard_language_label") or "普通话"),
        configs_dir=configs_dir,
        data_dir=data_dir,
        runs_dir=runs_dir,
        registry_dir=registry_dir,
        curated_dir=curated_dir,
        prompts_dir=prompts_dir,
        scene_policy_path=_pick_existing(configs_dir / "scene_policy.json", PROJECT_ROOT / "configs" / "scene_policy.json"),
        trust_policy_path=_pick_existing(configs_dir / "trust_policy.json", PROJECT_ROOT / "configs" / "trust_policy.json"),
        pipeline_thresholds_path=_pick_existing(
            configs_dir / "pipeline_thresholds.json",
            PROJECT_ROOT / "configs" / "pipeline_thresholds.json",
        ),
        grammar_spec_path=grammar_spec_path,
        grammar_rules_path=grammar_rules_path,
        scene_catalog_path=scene_catalog_path,
    )


ACTIVE_DIALECT_CONFIG = load_dialect_config()
