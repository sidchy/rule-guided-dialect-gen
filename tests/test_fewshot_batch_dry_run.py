import csv
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

from wz_pipeline.jsonl import read_jsonl
from wz_pipeline.pipelines import fewshot_batch as batch_module
from wz_pipeline.runs import RunLayout


def _layout(root: Path, run_id: str) -> RunLayout:
    run_root = root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    return RunLayout(
        pipeline_name="fewshot_batch",
        run_id=run_id,
        root=run_root,
        config_path=run_root / "config.json",
        results_path=run_root / "results.jsonl",
        rule_gate_path=run_root / "rule_gate.jsonl",
        review_tsv_path=run_root / "review.tsv",
        review_results_path=run_root / "review_results.jsonl",
        summary_path=run_root / "summary.json",
        promotion_candidates_path=run_root / "promotion_candidates.jsonl",
    )


def _example(surface: str, definition: str, scene_id: str, wz_sentence: str, zh_sentence: str) -> dict[str, str]:
    return {
        "wz_word": surface,
        "definition": definition,
        "scene_id": scene_id,
        "wz_sentence": wz_sentence,
        "zh_sentence": zh_sentence,
        "source_file": "test.jsonl",
    }


def _word(
    surface: str,
    definition: str,
    scene_id: str,
    *,
    semantic_class: str = "noun",
    slot_kind: str = "noun",
    source_type: str = "test",
    modern_priority: int = 0,
) -> dict[str, object]:
    return {
        "wz_word": surface,
        "definition": definition,
        "scene_id": scene_id,
        "semantic_class": semantic_class,
        "slot_kind": slot_kind,
        "source_type": source_type,
        "is_modern": modern_priority > 0,
        "modern_priority": modern_priority,
    }


def _examples_by_scene() -> dict[str, list[dict[str, str]]]:
    return {
        "home_life": [
            _example("灶脚", "家里厨房角落", "home_life", "屋里灶脚还热个。", "家里灶脚还是热的。"),
            _example("灶脚", "家里厨房角落", "home_life", "灶脚边还摆牢物事。", "灶脚边还摆着东西。"),
            _example("眠床", "睡觉用床", "home_life", "眠床上头摆勒衫裤。", "床上放着衣服。"),
        ],
        "food_dining": [
            _example("汤面", "热汤面", "food_dining", "今朝烧汤面吃。", "今天煮汤面吃。"),
            _example("汤面", "热汤面", "food_dining", "汤面还烫口个。", "汤面还很烫。"),
            _example("吃饭", "吃饭动作", "food_dining", "大家一道吃饭去。", "大家一起去吃饭。"),
        ],
        "health_medical": [
            _example("排队", "医院排队等候", "health_medical", "医院里排队等号。", "在医院排队挂号。"),
            _example("排队", "医院排队等候", "health_medical", "门诊口排队看病。", "门诊口排队看病。"),
            _example("挂号", "医院挂号", "health_medical", "先去挂号再看。", "先去挂号再看。"),
        ],
    }


def _words_by_scene() -> dict[str, list[dict[str, object]]]:
    return {
        "home_life": [
            _word("灶脚", "家里厨房角落", "home_life"),
            _word("眠床", "睡觉用床", "home_life"),
            _word("衫裤", "身上衣裳", "home_life"),
        ],
        "food_dining": [
            _word("汤面", "热汤面", "food_dining"),
            _word("吃饭", "吃饭动作", "food_dining"),
            _word("筷箸", "吃饭筷子", "food_dining"),
        ],
        "health_medical": [
            _word("排队", "医院排队等候", "health_medical"),
            _word("挂号", "医院挂号", "health_medical"),
            _word("门诊", "医院门诊", "health_medical"),
        ],
    }


def _known_words() -> set[str]:
    return {
        "灶脚",
        "眠床",
        "衫裤",
        "汤面",
        "吃饭",
        "筷箸",
        "排队",
        "挂号",
        "门诊",
        "屋里",
        "桌边",
        "医院",
    }


def _fake_domain_context(domain_ids: list[str] | None, *, scene_id: str) -> dict[str, object]:
    selected = [item for item in (domain_ids or []) if str(item).strip() == "medical" and scene_id == "health_medical"]
    if not selected:
        return {
            "domain_ids": [],
            "domain_labels": [],
            "required_terms": [],
            "preferred_terms": [],
            "blocked_terms": [],
            "prompt_notes": [],
            "review_notes": [],
        }
    return {
        "domain_ids": ["medical"],
        "domain_labels": ["医疗"],
        "required_terms": ["挂号"],
        "preferred_terms": ["门诊"],
        "blocked_terms": ["偏方"],
        "prompt_notes": ["优先用看病场景用语。"],
        "review_notes": ["检查医疗词是否自然。"],
    }


def _fake_generation(task: dict[str, object]) -> list[dict[str, str]]:
    scene_prefix = {
        "home_life": ("屋里", "眠床"),
        "food_dining": ("桌边", "筷箸"),
        "health_medical": ("医院", "门诊"),
    }
    variant_terms = ["头桩", "二桩", "三桩", "四桩", "五桩", "六桩"]
    variant = variant_terms[sum(ord(ch) for ch in str(task["task_id"])) % len(variant_terms)]
    scene_word, fallback_support = scene_prefix[str(task["scene_id"])]
    core_word = str(task["core_word"])
    support_words = [str(item) for item in task.get("support_words", [])]
    support_word = support_words[0] if support_words else fallback_support
    domain_terms = [str(item) for item in task.get("domain_required_terms", [])]
    domain_word = domain_terms[0] if domain_terms else ""
    parts = [variant, scene_word, core_word]
    if support_word and support_word not in parts:
        parts.append(support_word)
    if domain_word and domain_word not in parts:
        parts.append(domain_word)
    parts.extend(["个事体", "阿拉", "一道", "慢慢", "讲清", "停当", "转去"])
    wz_sentence = "".join(parts)
    zh_sentence = f"{task['scene_id']} {core_word}"
    return [{"wz": wz_sentence, "zh": zh_sentence}]


def test_fewshot_batch_offline_dry_runs_cover_baseline_domain_and_feedback() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        runs_root = root / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        feedback_summary_path = root / "feedback_summary.json"
        feedback_summary_path.write_text(
            json.dumps(
                {
                    "run_id": "prev_demo",
                    "machine_metrics": {
                        "failure_analysis": {
                            "rule_fail_count": 2,
                            "bucket_counts": {"grammar": 2},
                            "primary_bucket": "grammar",
                            "top_grammar_reasons": [{"name": "particle_misuse", "count": 2}],
                            "top_domain_reasons": [],
                            "top_naturalness_reasons": [],
                        }
                    },
                    "extra": {
                        "scene_rule_pass_counts": {
                            "home_life": 2,
                            "food_dining": 1,
                            "health_medical": 1,
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        originals = {
            "prepare_run_layout": batch_module.prepare_run_layout,
            "register_run": batch_module.register_run,
            "build_client": batch_module.build_client,
            "load_sibling_dedup_state": batch_module.load_sibling_dedup_state,
            "load_wz_word_set": batch_module.load_wz_word_set,
            "load_examples_by_scene": batch_module.load_examples_by_scene,
            "load_words_by_scene": batch_module.load_words_by_scene,
            "request_generation": batch_module.request_generation,
            "core_is_allowed": batch_module.core_is_allowed,
            "core_is_blocked": batch_module.core_is_blocked,
            "word_looks_usable": batch_module.word_looks_usable,
            "example_looks_usable": batch_module.example_looks_usable,
            "core_word_looks_usable": batch_module.core_word_looks_usable,
            "scene_match_score": batch_module.scene_match_score,
            "build_domain_context": batch_module.build_domain_context,
            "grammar_validation_reasons": batch_module.grammar_validation_reasons,
            "relevant_spec_labels": batch_module.relevant_spec_labels,
            "should_attempt_grammar_repair": batch_module.should_attempt_grammar_repair,
            "detect_unsupported_surface_terms": batch_module.detect_unsupported_surface_terms,
            "EXTRACTED_SHORT": batch_module.EXTRACTED_SHORT,
            "EXTRACTED_LONG": batch_module.EXTRACTED_LONG,
        }
        registered_runs: list[dict[str, object]] = []
        try:
            batch_module.prepare_run_layout = lambda pipeline_name, run_id: _layout(runs_root / pipeline_name, run_id)
            batch_module.register_run = lambda entry: registered_runs.append(entry)
            batch_module.build_client = lambda provider: (object(), "fake-model", provider)
            batch_module.load_sibling_dedup_state = lambda runs_root, current_run_id: (set(), Counter())
            batch_module.load_wz_word_set = _known_words
            batch_module.load_examples_by_scene = _examples_by_scene
            batch_module.load_words_by_scene = _words_by_scene
            batch_module.request_generation = lambda client, model, task: _fake_generation(task)
            batch_module.core_is_allowed = lambda scene_id, wz_word: True
            batch_module.core_is_blocked = lambda scene_id, wz_word, definition, source_file="": False
            batch_module.word_looks_usable = lambda word: True
            batch_module.example_looks_usable = lambda example: True
            batch_module.core_word_looks_usable = lambda word, scene_id, anchor=None: True
            batch_module.scene_match_score = lambda scene_id, *texts: 1 if any(texts) else 0
            batch_module.build_domain_context = _fake_domain_context
            batch_module.grammar_validation_reasons = lambda text: []
            batch_module.relevant_spec_labels = lambda sentence, reasons: []
            batch_module.should_attempt_grammar_repair = lambda text: False
            batch_module.detect_unsupported_surface_terms = lambda text, protected_terms=None: []
            batch_module.EXTRACTED_SHORT = root / "missing_short.jsonl"
            batch_module.EXTRACTED_LONG = root / "missing_long.jsonl"

            run_specs = [
                ("baseline_demo", []),
                ("domain_demo", ["--domains", "medical"]),
                ("feedback_demo", ["--feedback-summary", str(feedback_summary_path)]),
            ]
            original_argv = sys.argv
            try:
                for run_id, extra_args in run_specs:
                    sys.argv = [
                        "wz-generate-fewshot-batch",
                        "--tasks",
                        "3",
                        "--provider",
                        "deepseek",
                        "--scenes",
                        "home_life,food_dining,health_medical",
                        "--sleep",
                        "0",
                        "--disable-grammar-repair",
                        "--run-id",
                        run_id,
                        *extra_args,
                    ]
                    batch_module.main()
            finally:
                sys.argv = original_argv
        finally:
            for name, value in originals.items():
                setattr(batch_module, name, value)

        baseline_root = runs_root / "fewshot_batch" / "baseline_demo"
        domain_root = runs_root / "fewshot_batch" / "domain_demo"
        feedback_root = runs_root / "fewshot_batch" / "feedback_demo"

        for run_root in (baseline_root, domain_root, feedback_root):
            assert (run_root / "config.json").exists()
            assert (run_root / "rule_gate.jsonl").exists()
            assert (run_root / "review.tsv").exists()
            assert (run_root / "summary.json").exists()

        baseline_config = json.loads((baseline_root / "config.json").read_text(encoding="utf-8"))
        baseline_summary = json.loads((baseline_root / "summary.json").read_text(encoding="utf-8"))
        baseline_rule_gate = read_jsonl(baseline_root / "rule_gate.jsonl")
        with (baseline_root / "review.tsv").open(encoding="utf-8", newline="") as handle:
            baseline_header = next(csv.reader(handle, delimiter="\t"))

        assert baseline_config["domains"] == []
        assert baseline_config["feedback"]["enabled"] is False
        assert "failure_analysis" in baseline_summary["machine_metrics"]
        assert baseline_summary["machine_metrics"]["rule_pass_count"] > 0
        assert baseline_summary["extra"]["scene_task_counts"]
        assert baseline_summary["extra"]["scene_target_quotas"]
        assert baseline_summary["extra"]["task_speech_act_counts"]
        assert baseline_summary["extra"]["scene_core_task_counts"]
        assert baseline_summary["extra"]["scene_support_task_counts"]
        assert baseline_summary["extra"]["scene_place_support_task_counts"] is not None
        assert baseline_summary["extra"]["scene_distinct_core_count"]
        assert baseline_summary["extra"]["scene_distinct_support_count"]
        assert "failure_buckets" in baseline_header
        assert "failure_primary_bucket" in baseline_header
        assert "review_focus" in baseline_header
        assert "target_speech_act" in baseline_header
        assert "task_speech_acts" in baseline_header
        assert baseline_rule_gate
        assert "validation" in baseline_rule_gate[0]
        assert "grammar_reasons" in baseline_rule_gate[0]["validation"]
        assert "domain_required_hits" in baseline_rule_gate[0]["validation"]
        assert baseline_rule_gate[0]["target_speech_act"]
        assert len(baseline_rule_gate[0]["task_speech_acts"]) == 3

        domain_config = json.loads((domain_root / "config.json").read_text(encoding="utf-8"))
        domain_summary = json.loads((domain_root / "summary.json").read_text(encoding="utf-8"))
        domain_rule_gate = read_jsonl(domain_root / "rule_gate.jsonl")
        assert domain_config["domains"] == ["medical"]
        assert domain_summary["machine_metrics"]["rule_pass_count"] > 0
        assert domain_rule_gate
        assert all(row["domain_ids"] == ["medical"] for row in domain_rule_gate)
        assert any(row["validation"]["domain_required_hits"] == ["挂号"] for row in domain_rule_gate)

        feedback_config = json.loads((feedback_root / "config.json").read_text(encoding="utf-8"))
        feedback_summary = json.loads((feedback_root / "summary.json").read_text(encoding="utf-8"))
        assert feedback_config["feedback"]["enabled"] is True
        assert feedback_config["feedback"]["source_summary_path"] == str(feedback_summary_path)
        assert feedback_config["feedback"]["primary_bucket"] == "grammar"
        assert feedback_config["feedback"]["anchor_pool_top_k"] == 2
        assert feedback_config["food_modern_trial_ratio"] == 0.125
        assert "failure_analysis" in feedback_summary["machine_metrics"]
        assert feedback_summary["machine_metrics"]["rule_pass_count"] > 0
        assert len(registered_runs) == 3


def test_validate_sentence_rejects_structural_duplicates() -> None:
    wz = "你行李恁重，我伉你相伴拎，省得你吃力显。"
    skeleton = batch_module.sentence_skeleton(wz, ["相伴", "共享单车", "车站大道"])
    validation = batch_module.validate_sentence(
        wz,
        "你行李这么重，我陪你一起拿，免得你太累。",
        known_words={"相伴", "共享单车", "车站大道", "行李", "吃力"},
        core_word="相伴",
        support_words=["共享单车", "车站大道"],
        existing_sentences=set(),
        existing_skeleton_counts=Counter({skeleton: 1}),
        candidate_skeleton=skeleton,
        scene_id="transport_trip",
        lane="mainline",
        core_tier="stable",
        approved_modern_terms=["共享单车", "车站大道"],
        banned_terms=[],
        domain_required_terms=[],
        domain_preferred_terms=[],
        domain_blocked_terms=[],
    )
    assert "structural_duplicate" in validation["reasons"]


def test_create_tasks_balanced_covers_all_speech_act_windows() -> None:
    originals = {
        "core_is_allowed": batch_module.core_is_allowed,
        "core_is_blocked": batch_module.core_is_blocked,
        "word_looks_usable": batch_module.word_looks_usable,
        "example_looks_usable": batch_module.example_looks_usable,
        "core_word_looks_usable": batch_module.core_word_looks_usable,
        "scene_match_score": batch_module.scene_match_score,
        "build_domain_context": batch_module.build_domain_context,
    }
    try:
        batch_module.core_is_allowed = lambda scene_id, wz_word: True
        batch_module.core_is_blocked = lambda scene_id, wz_word, definition, source_file="": False
        batch_module.word_looks_usable = lambda word: True
        batch_module.example_looks_usable = lambda example: True
        batch_module.core_word_looks_usable = lambda word, scene_id, anchor=None: True
        batch_module.scene_match_score = lambda scene_id, *texts: 1 if any(texts) else 0
        batch_module.build_domain_context = lambda domain_ids, scene_id: {
            "domain_ids": [],
            "domain_labels": [],
            "required_terms": [],
            "preferred_terms": [],
            "blocked_terms": [],
            "prompt_notes": [],
            "review_notes": [],
        }

        tasks = batch_module.create_tasks_balanced(
            examples_by_scene={"home_life": _examples_by_scene()["home_life"]},
            words_by_scene={"home_life": _words_by_scene()["home_life"]},
            num_tasks=5,
            seed=17,
            scene_filter=["home_life"],
        )
    finally:
        for name, value in originals.items():
            setattr(batch_module, name, value)

    assert len(tasks) == 5
    assert {tuple(task["speech_acts"]) for task in tasks} == {
        ("question", "complaint", "request"),
        ("complaint", "request", "narration"),
        ("request", "narration", "evaluation"),
        ("narration", "evaluation", "question"),
        ("evaluation", "question", "complaint"),
    }


def test_create_tasks_balanced_allows_example_only_scene() -> None:
    originals = {
        "core_is_allowed": batch_module.core_is_allowed,
        "core_is_blocked": batch_module.core_is_blocked,
        "word_looks_usable": batch_module.word_looks_usable,
        "example_looks_usable": batch_module.example_looks_usable,
        "core_word_looks_usable": batch_module.core_word_looks_usable,
        "scene_match_score": batch_module.scene_match_score,
        "build_domain_context": batch_module.build_domain_context,
    }
    try:
        batch_module.core_is_allowed = lambda scene_id, wz_word: True
        batch_module.core_is_blocked = lambda scene_id, wz_word, definition, source_file="": False
        batch_module.word_looks_usable = lambda word: True
        batch_module.example_looks_usable = lambda example: True
        batch_module.core_word_looks_usable = lambda word, scene_id, anchor=None: True
        batch_module.scene_match_score = lambda scene_id, *texts: 1 if any(texts) else 0
        batch_module.build_domain_context = lambda domain_ids, scene_id: {
            "domain_ids": [],
            "domain_labels": [],
            "required_terms": [],
            "preferred_terms": [],
            "blocked_terms": [],
            "prompt_notes": [],
            "review_notes": [],
        }

        tasks = batch_module.create_tasks_balanced(
            examples_by_scene={"health_medical": _examples_by_scene()["health_medical"]},
            words_by_scene={},
            num_tasks=5,
            seed=19,
            scene_filter=["health_medical"],
        )
    finally:
        for name, value in originals.items():
            setattr(batch_module, name, value)

    assert len(tasks) == 5
    assert {task["scene_id"] for task in tasks} == {"health_medical"}


def test_expand_scene_targets_splits_digital_chat_into_subscenes() -> None:
    targets = batch_module.expand_scene_targets(
        ["digital_chat"],
        "手机微信消息视频打电话装软件更新密码提醒",
        expand_digital=True,
    )
    assert "digital_chat" in targets
    assert "digital_ai_assistant" in targets
    assert "digital_messaging_call" in targets
    assert "digital_device_trouble" in targets
    assert "digital_app_operation" in targets


def test_stable_core_policy_expanded_for_mainline_scenes() -> None:
    assert len(batch_module.stable_core_terms("home_life")) >= 8
    assert len(batch_module.stable_core_terms("transport_trip")) >= 8
    assert len(batch_module.stable_core_terms("shopping_payment")) >= 8
    assert len(batch_module.stable_core_terms("weather_safety")) >= 8
    assert len(batch_module.stable_core_terms("food_dining")) >= 6
    assert len(batch_module.stable_core_terms("health_medical")) >= 6
    assert len(batch_module.stable_core_terms("work_study")) >= 6


def test_load_words_by_scene_caps_focus_replaceables_and_blocks_places() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        replaceable_path = root / "replaceable_lexicon.jsonl"
        rows = []
        for idx in range(20):
            rows.append(
                {
                    "wz_word": f"名词{idx:02d}",
                    "mandarin_headword": f"名词{idx:02d}",
                    "definition": f"家里名词{idx:02d}",
                    "semantic_class": "noun",
                    "slot_kind": "noun",
                    "primary_scene_id": "home_life",
                }
            )
        for idx in range(20):
            rows.append(
                {
                    "wz_word": f"动作{idx:02d}",
                    "mandarin_headword": f"动作{idx:02d}",
                    "definition": f"家里动作{idx:02d}",
                    "semantic_class": "action",
                    "slot_kind": "verb",
                    "primary_scene_id": "home_life",
                }
            )
        for idx in range(20):
            rows.append(
                {
                    "wz_word": f"设备{idx:02d}",
                    "mandarin_headword": f"设备{idx:02d}",
                    "definition": f"家里设备{idx:02d}",
                    "semantic_class": "device",
                    "slot_kind": "noun",
                    "primary_scene_id": "home_life",
                }
            )
        for idx in range(6):
            rows.append(
                {
                    "wz_word": f"地名{idx:02d}",
                    "mandarin_headword": f"地名{idx:02d}",
                    "definition": f"地点{idx:02d}",
                    "semantic_class": "place",
                    "slot_kind": "place",
                    "primary_scene_id": "home_life",
                }
            )
        replaceable_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

        originals = {
            "REPLACEABLE_LEXICON": batch_module.REPLACEABLE_LEXICON,
            "DENSE_WHITELIST": batch_module.DENSE_WHITELIST,
            "load_modern_words_from_assets": batch_module.load_modern_words_from_assets,
            "choose_scene_for_word": batch_module.choose_scene_for_word,
            "expand_scene_targets": batch_module.expand_scene_targets,
        }
        try:
            batch_module.REPLACEABLE_LEXICON = replaceable_path
            batch_module.DENSE_WHITELIST = root / "missing_dense.jsonl"
            batch_module.load_modern_words_from_assets = lambda by_scene, seen: None
            batch_module.choose_scene_for_word = lambda row: "home_life"
            batch_module.expand_scene_targets = lambda scene_ids, *texts, expand_digital=False: [str(scene_ids[0])]

            words_by_scene = batch_module.load_words_by_scene()
        finally:
            for name, value in originals.items():
                setattr(batch_module, name, value)

        replaceable_words = [
            row
            for row in words_by_scene["home_life"]
            if row.get("source_type") == "replaceable_lexicon"
        ]
        category_counts = Counter(
            batch_module.normalized_focus_replaceable_category(row)
            for row in replaceable_words
        )

        assert len(replaceable_words) == 36
        assert category_counts["noun"] == 12
        assert category_counts["action"] == 12
        assert category_counts["device"] == 12
        assert all(row.get("slot_kind") != "place" for row in replaceable_words)


def test_create_tasks_balanced_spreads_core_usage_within_scene_cap() -> None:
    scene_id = "home_life"
    examples_by_scene = {
        scene_id: [
            _example("大蛮阵", "家里阵仗大", scene_id, "屋里大蛮阵还未收拾停当。", "家里东西很多还没收拾好。"),
            _example("整理", "整理家里", scene_id, "你先整理桌面再出去。", "你先整理桌面再出去。"),
            _example("收拾", "收拾屋里", scene_id, "阿妈催我收拾房间。", "妈妈催我收拾房间。"),
            _example("晾起", "晾衣服起来", scene_id, "阳台里个衫裤快晾起。", "阳台里的衣服快晾起来。"),
            _example("冰箱", "家用冰箱", scene_id, "冰箱里还搁牢菜。", "冰箱里还放着菜。"),
            _example("空调", "家里空调", scene_id, "夜里空调莫开太冷。", "晚上空调别开太冷。"),
            _example("洗衣机", "家用洗衣机", scene_id, "洗衣机还勒转个。", "洗衣机还在转。"),
            _example("路由器", "家里路由器", scene_id, "路由器断电脱再插起。", "路由器断电后再插上。"),
        ]
    }
    words_by_scene = {
        scene_id: [
            _word("大蛮阵", "家里阵仗大", scene_id),
            _word("整理", "整理家里", scene_id, semantic_class="action", slot_kind="verb"),
            _word("收拾", "收拾屋里", scene_id, semantic_class="action", slot_kind="verb"),
            _word("晾起", "晾衣服起来", scene_id, semantic_class="action", slot_kind="verb"),
            _word("冰箱", "家用冰箱", scene_id, semantic_class="device"),
            _word("空调", "家里空调", scene_id, semantic_class="device"),
            _word("洗衣机", "家用洗衣机", scene_id, semantic_class="device"),
            _word("路由器", "家里路由器", scene_id, semantic_class="device"),
            _word("阳台", "家里阳台", scene_id),
            _word("厨房", "家里厨房", scene_id),
            _word("书房", "家里书房", scene_id),
            _word("热水器", "家里热水器", scene_id, semantic_class="device"),
        ]
    }

    originals = {
        "core_is_allowed": batch_module.core_is_allowed,
        "core_is_blocked": batch_module.core_is_blocked,
        "word_looks_usable": batch_module.word_looks_usable,
        "example_looks_usable": batch_module.example_looks_usable,
        "core_word_looks_usable": batch_module.core_word_looks_usable,
        "scene_match_score": batch_module.scene_match_score,
        "build_domain_context": batch_module.build_domain_context,
    }
    try:
        batch_module.core_is_allowed = lambda scene_id, wz_word: True
        batch_module.core_is_blocked = lambda scene_id, wz_word, definition, source_file="": False
        batch_module.word_looks_usable = lambda word: True
        batch_module.example_looks_usable = lambda example: True
        batch_module.core_word_looks_usable = lambda word, scene_id, anchor=None: True
        batch_module.scene_match_score = lambda scene_id, *texts: 1 if any(texts) else 0
        batch_module.build_domain_context = lambda domain_ids, scene_id: {
            "domain_ids": [],
            "domain_labels": [],
            "required_terms": [],
            "preferred_terms": [],
            "blocked_terms": [],
            "prompt_notes": [],
            "review_notes": [],
        }

        tasks = batch_module.create_tasks_balanced(
            examples_by_scene=examples_by_scene,
            words_by_scene=words_by_scene,
            num_tasks=12,
            seed=29,
            scene_filter=[scene_id],
        )
    finally:
        for name, value in originals.items():
            setattr(batch_module, name, value)

    core_counts = Counter(str(task.get("core_word") or "") for task in tasks)
    assert len(tasks) == 12
    assert len(core_counts) >= 4
    assert core_counts.most_common(1)[0][1] <= 5


def test_create_tasks_balanced_rotates_transport_place_supports() -> None:
    scene_id = "transport_trip"
    examples_by_scene = {
        scene_id: [
            _example("相伴", "一起同行", scene_id, "你伉我相伴去车站大道。", "你陪我一起去车站大道。"),
            _example("导航", "手机导航", scene_id, "导航讲还要转一道。", "导航说还要再转一次。"),
            _example("换乘", "中途换乘", scene_id, "今朝换乘两趟车。", "今天换乘两趟车。"),
            _example("打车", "叫车出门", scene_id, "雨落大就打车去。", "雨下大就打车去。"),
            _example("地铁", "搭地铁", scene_id, "地铁一站一站慢慢到。", "地铁一站一站慢慢到。"),
            _example("高铁", "坐高铁", scene_id, "高铁票今朝先看起。", "高铁票今天先看起来。"),
            _example("共享单车", "扫码骑车", scene_id, "共享单车扫起就行。", "共享单车扫码就可以。"),
            _example("网约车", "手机约车", scene_id, "网约车讲两分钟到。", "网约车说两分钟到。"),
        ]
    }
    words_by_scene = {
        scene_id: [
            _word("相伴", "一起同行", scene_id),
            _word("导航", "手机导航", scene_id, semantic_class="transport"),
            _word("换乘", "中途换乘", scene_id, semantic_class="transport"),
            _word("打车", "叫车出门", scene_id, semantic_class="transport"),
            _word("地铁", "搭地铁", scene_id, semantic_class="transport"),
            _word("高铁", "坐高铁", scene_id, semantic_class="transport"),
            _word("共享单车", "扫码骑车", scene_id, semantic_class="transport"),
            _word("网约车", "手机约车", scene_id, semantic_class="transport"),
            _word("车站大道", "地点名词", scene_id, semantic_class="place", slot_kind="place", source_type="place_name", modern_priority=3),
            _word("高铁站", "地点名词", scene_id, semantic_class="place", slot_kind="place", source_type="place_name", modern_priority=3),
            _word("地铁站", "地点名词", scene_id, semantic_class="place", slot_kind="place", source_type="place_name", modern_priority=3),
            _word("机场路", "地点名词", scene_id, semantic_class="place", slot_kind="place", source_type="place_name", modern_priority=3),
            _word("南塘街", "地点名词", scene_id, semantic_class="place", slot_kind="place", source_type="place_name", modern_priority=3),
            _word("瓯海大道", "地点名词", scene_id, semantic_class="place", slot_kind="place", source_type="place_name", modern_priority=3),
            _word("新城站", "地点名词", scene_id, semantic_class="place", slot_kind="place", source_type="place_name", modern_priority=3),
            _word("蒲鞋市", "地点名词", scene_id, semantic_class="place", slot_kind="place", source_type="place_name", modern_priority=3),
            _word("学院路", "地点名词", scene_id, semantic_class="place", slot_kind="place", source_type="place_name", modern_priority=3),
            _word("梧田站", "地点名词", scene_id, semantic_class="place", slot_kind="place", source_type="place_name", modern_priority=3),
            _word("车票", "车票", scene_id, semantic_class="noun"),
            _word("行李", "出门行李", scene_id, semantic_class="noun"),
        ]
    }

    originals = {
        "core_is_allowed": batch_module.core_is_allowed,
        "core_is_blocked": batch_module.core_is_blocked,
        "word_looks_usable": batch_module.word_looks_usable,
        "example_looks_usable": batch_module.example_looks_usable,
        "core_word_looks_usable": batch_module.core_word_looks_usable,
        "scene_match_score": batch_module.scene_match_score,
        "build_domain_context": batch_module.build_domain_context,
    }
    try:
        batch_module.core_is_allowed = lambda scene_id, wz_word: True
        batch_module.core_is_blocked = lambda scene_id, wz_word, definition, source_file="": False
        batch_module.word_looks_usable = lambda word: True
        batch_module.example_looks_usable = lambda example: True
        batch_module.core_word_looks_usable = lambda word, scene_id, anchor=None: True
        batch_module.scene_match_score = lambda scene_id, *texts: 1 if any(texts) else 0
        batch_module.build_domain_context = lambda domain_ids, scene_id: {
            "domain_ids": [],
            "domain_labels": [],
            "required_terms": [],
            "preferred_terms": [],
            "blocked_terms": [],
            "prompt_notes": [],
            "review_notes": [],
        }

        tasks = batch_module.create_tasks_balanced(
            examples_by_scene=examples_by_scene,
            words_by_scene=words_by_scene,
            num_tasks=20,
            seed=41,
            scene_filter=[scene_id],
        )
    finally:
        for name, value in originals.items():
            setattr(batch_module, name, value)

    lexical_summary = batch_module.summarize_task_lexical_diversity(tasks)
    place_counts = lexical_summary["scene_place_support_task_counts"][scene_id]

    assert len(tasks) == 20
    assert sum(place_counts.values()) >= 10
    assert len(place_counts) >= 8
    assert max(place_counts.values()) <= 3
