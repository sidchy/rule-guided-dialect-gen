from __future__ import annotations

from pathlib import Path

from .dialect import ACTIVE_DIALECT_CONFIG


ROOT = ACTIVE_DIALECT_CONFIG.project_root
DIALECT = ACTIVE_DIALECT_CONFIG.dialect_key
DIALECT_DIR = ACTIVE_DIALECT_CONFIG.dialect_dir
CONFIGS_DIR = ACTIVE_DIALECT_CONFIG.configs_dir
DATA_DIR = ACTIVE_DIALECT_CONFIG.data_dir
REGISTRY_DIR = ACTIVE_DIALECT_CONFIG.registry_dir
RUNS_DIR = ACTIVE_DIALECT_CONFIG.runs_dir
CURATED_DIR = ACTIVE_DIALECT_CONFIG.curated_dir
PROMPTS_DIR = ACTIVE_DIALECT_CONFIG.prompts_dir
DOCS_DIR = ROOT / "docs"
SCRIPTS_DIR = ROOT / "scripts"
SCENE_POLICY_PATH = ACTIVE_DIALECT_CONFIG.scene_policy_path
TRUST_POLICY_PATH = ACTIVE_DIALECT_CONFIG.trust_policy_path
PIPELINE_THRESHOLDS_PATH = ACTIVE_DIALECT_CONFIG.pipeline_thresholds_path
GRAMMAR_SPEC_PATH = ACTIVE_DIALECT_CONFIG.grammar_spec_path
GRAMMAR_RULES_PATH = ACTIVE_DIALECT_CONFIG.grammar_rules_path
SCENE_CATALOG_PATH = ACTIVE_DIALECT_CONFIG.scene_catalog_path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
