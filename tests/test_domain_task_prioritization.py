import random

from wz_pipeline.pipelines import fewshot_batch as batch_module


def _word(surface: str, definition: str, scene_id: str) -> dict[str, object]:
    return {
        "wz_word": surface,
        "definition": definition,
        "scene_id": scene_id,
        "semantic_class": "noun",
        "slot_kind": "noun",
        "source_type": "test",
        "is_modern": False,
        "modern_priority": 0,
    }


def _example(surface: str, definition: str, scene_id: str, wz_sentence: str, zh_sentence: str) -> dict[str, str]:
    return {
        "wz_word": surface,
        "definition": definition,
        "scene_id": scene_id,
        "wz_sentence": wz_sentence,
        "zh_sentence": zh_sentence,
        "source_file": "test.jsonl",
    }


def test_select_core_and_support_words_prioritizes_domain_terms() -> None:
    originals = {
        "core_is_allowed": batch_module.core_is_allowed,
        "core_is_blocked": batch_module.core_is_blocked,
        "scene_match_score": batch_module.scene_match_score,
    }
    try:
        batch_module.core_is_allowed = lambda scene_id, wz_word: True
        batch_module.core_is_blocked = lambda scene_id, wz_word, definition, source_file="": False
        batch_module.scene_match_score = lambda scene_id, *texts: 2 if any(texts) else 0

        anchor = _example(
            "排队",
            "医院排队等候",
            "health_medical",
            "医院里排队等号。",
            "在医院排队挂号。",
        )
        examples = [
            anchor,
            _example(
                "排队",
                "医院排队等候",
                "health_medical",
                "门诊口排队看病。",
                "门诊口排队看病。",
            ),
        ]
        scene_words = [
            _word("挂号", "医院挂号", "health_medical"),
            _word("门诊", "医院门诊", "health_medical"),
        ]

        core_word, support_words = batch_module.select_core_and_support_words(
            anchor,
            scene_words,
            examples,
            "health_medical",
            rng=random.Random(0),
            domain_context={
                "domain_ids": ["medical"],
                "required_terms": ["挂号"],
                "preferred_terms": ["门诊"],
            },
        )
    finally:
        for name, value in originals.items():
            setattr(batch_module, name, value)

    assert core_word is not None
    assert core_word["wz_word"] == "挂号"
    assert [word["wz_word"] for word in support_words] == ["门诊"]


def test_create_tasks_balanced_skips_scenes_without_matching_domain() -> None:
    originals = {
        "build_domain_context": batch_module.build_domain_context,
        "build_task": batch_module.build_task,
        "core_is_allowed": batch_module.core_is_allowed,
        "core_is_blocked": batch_module.core_is_blocked,
        "scene_match_score": batch_module.scene_match_score,
        "PRIORITY_SCENES": batch_module.PRIORITY_SCENES,
        "MODERN_SIDECAR_SCENES": batch_module.MODERN_SIDECAR_SCENES,
    }
    try:
        batch_module.build_domain_context = lambda domain_ids, scene_id: (
            {
                "domain_ids": ["medical"],
                "domain_labels": ["医疗"],
                "required_terms": ["挂号"],
                "preferred_terms": ["门诊"],
                "blocked_terms": [],
                "prompt_notes": [],
                "review_notes": [],
            }
            if scene_id == "health_medical"
            else {
                "domain_ids": [],
                "domain_labels": [],
                "required_terms": [],
                "preferred_terms": [],
                "blocked_terms": [],
                "prompt_notes": [],
                "review_notes": [],
            }
        )
        batch_module.build_task = (
            lambda examples, core_word, support_words, scene_id, task_id, lane, core_tier, domain_ids, domain_context=None: {
                "task_id": task_id,
                "scene_id": scene_id,
                "core_word": core_word["wz_word"],
                "support_words": [word["wz_word"] for word in support_words],
                "domain_ids": list(domain_ids),
            }
        )
        batch_module.core_is_allowed = lambda scene_id, wz_word: True
        batch_module.core_is_blocked = lambda scene_id, wz_word, definition, source_file="": False
        batch_module.scene_match_score = lambda scene_id, *texts: 1 if any(texts) else 0
        batch_module.PRIORITY_SCENES = ["food_dining", "health_medical"]
        batch_module.MODERN_SIDECAR_SCENES = []

        examples_by_scene = {
            "food_dining": [
                _example("吃饭", "吃饭动作", "food_dining", "大家一道吃饭。", "大家一起吃饭。"),
                _example("吃饭", "吃饭动作", "food_dining", "中午去吃饭。", "中午去吃饭。"),
                _example("汤面", "热汤面", "food_dining", "今朝烧汤面。", "今天煮汤面。"),
            ],
            "health_medical": [
                _example("排队", "医院排队等候", "health_medical", "医院里排队等号。", "在医院排队挂号。"),
                _example("排队", "医院排队等候", "health_medical", "门诊口排队看病。", "门诊口排队看病。"),
                _example("挂号", "医院挂号", "health_medical", "先去挂号再看。", "先去挂号再看。"),
            ],
        }
        words_by_scene = {
            "food_dining": [
                _word("吃饭", "吃饭动作", "food_dining"),
                _word("汤面", "热汤面", "food_dining"),
                _word("米饭", "白米饭", "food_dining"),
            ],
            "health_medical": [
                _word("排队", "医院排队等候", "health_medical"),
                _word("挂号", "医院挂号", "health_medical"),
                _word("门诊", "医院门诊", "health_medical"),
            ],
        }

        tasks = batch_module.create_tasks_balanced(
            examples_by_scene,
            words_by_scene,
            1,
            seed=0,
            scene_filter=["food_dining", "health_medical"],
            domain_ids=["medical"],
        )
    finally:
        for name, value in originals.items():
            setattr(batch_module, name, value)

    assert len(tasks) == 1
    assert tasks[0]["scene_id"] == "health_medical"
    assert tasks[0]["domain_ids"] == ["medical"]
