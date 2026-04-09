import json
import tempfile
from pathlib import Path

import wz_pipeline.domain_config as domain_module


def test_build_domain_context_merges_terms_and_filters_by_scene() -> None:
    payload = [
        {
            "domain_id": "medical",
            "label": "医疗",
            "scene_allowlist": ["health_medical"],
            "required_terms": ["挂号"],
            "preferred_terms": ["门诊"],
            "blocked_terms": ["偏方"],
            "prompt_notes": ["优先用就医场景常用说法。"],
            "review_notes": ["检查医疗词是否自然。"],
        },
        {
            "domain_id": "payments",
            "label": "支付",
            "required_terms": ["付款码"],
            "preferred_terms": ["收款码"],
            "blocked_terms": [],
            "prompt_notes": [],
            "review_notes": [],
        },
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        catalog_path = Path(tmp_dir) / "domain_catalog.json"
        catalog_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        original_path = domain_module.DOMAIN_CATALOG_PATH
        try:
            domain_module.DOMAIN_CATALOG_PATH = catalog_path
            domain_module.load_domain_catalog.cache_clear()
            context = domain_module.build_domain_context(["medical", "payments"], scene_id="health_medical")
        finally:
            domain_module.DOMAIN_CATALOG_PATH = original_path
            domain_module.load_domain_catalog.cache_clear()

    assert context["domain_ids"] == ["medical", "payments"]
    assert context["required_terms"] == ["挂号", "付款码"]
    assert context["preferred_terms"] == ["门诊", "收款码"]
    assert context["blocked_terms"] == ["偏方"]
    assert context["prompt_notes"] == ["优先用就医场景常用说法。"]


def test_build_domain_context_drops_scene_mismatched_domain() -> None:
    payload = [
        {
            "domain_id": "medical",
            "label": "医疗",
            "scene_allowlist": ["health_medical"],
            "required_terms": ["挂号"],
        }
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        catalog_path = Path(tmp_dir) / "domain_catalog.json"
        catalog_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        original_path = domain_module.DOMAIN_CATALOG_PATH
        try:
            domain_module.DOMAIN_CATALOG_PATH = catalog_path
            domain_module.load_domain_catalog.cache_clear()
            context = domain_module.build_domain_context(["medical"], scene_id="food_dining")
        finally:
            domain_module.DOMAIN_CATALOG_PATH = original_path
            domain_module.load_domain_catalog.cache_clear()

    assert context["domain_ids"] == []
    assert context["required_terms"] == []


def test_build_domain_context_inherits_parent_scene_allowlist() -> None:
    payload = [
        {
            "domain_id": "medical",
            "label": "医疗",
            "scene_allowlist": ["health_medical"],
            "required_terms": ["挂号"],
        }
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        catalog_path = Path(tmp_dir) / "domain_catalog.json"
        catalog_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        original_path = domain_module.DOMAIN_CATALOG_PATH
        try:
            domain_module.DOMAIN_CATALOG_PATH = catalog_path
            domain_module.load_domain_catalog.cache_clear()
            context = domain_module.build_domain_context(["medical"], scene_id="health_hospital")
        finally:
            domain_module.DOMAIN_CATALOG_PATH = original_path
            domain_module.load_domain_catalog.cache_clear()

    assert context["domain_ids"] == ["medical"]
    assert context["required_terms"] == ["挂号"]


def test_build_domain_context_inherits_digital_parent_allowlist() -> None:
    payload = [
        {
            "domain_id": "chat",
            "label": "沟通",
            "scene_allowlist": ["digital_chat"],
            "required_terms": ["消息"],
        }
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        catalog_path = Path(tmp_dir) / "domain_catalog.json"
        catalog_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        original_path = domain_module.DOMAIN_CATALOG_PATH
        try:
            domain_module.DOMAIN_CATALOG_PATH = catalog_path
            domain_module.load_domain_catalog.cache_clear()
            context = domain_module.build_domain_context(["chat"], scene_id="digital_ai_assistant")
        finally:
            domain_module.DOMAIN_CATALOG_PATH = original_path
            domain_module.load_domain_catalog.cache_clear()

    assert context["domain_ids"] == ["chat"]
    assert context["required_terms"] == ["消息"]
