import csv
import json
import sys
import tempfile
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
        assert "failure_buckets" in baseline_header
        assert "failure_primary_bucket" in baseline_header
        assert "review_focus" in baseline_header
        assert baseline_rule_gate
        assert "validation" in baseline_rule_gate[0]
        assert "grammar_reasons" in baseline_rule_gate[0]["validation"]
        assert "domain_required_hits" in baseline_rule_gate[0]["validation"]

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


def test_word_looks_usable_respects_runtime_blocklists() -> None:
    assert not batch_module.word_looks_usable({"wz_word": "一时一刻", "definition": "一会儿"})
    assert not batch_module.word_looks_usable({"wz_word": "挤皁", "definition": "挤压皂液"})
    assert not batch_module.word_looks_usable(
        {"wz_word": "物事", "definition": "东西", "source_file": "活色生香温州话.xlsx"}
    )


def test_example_looks_usable_respects_runtime_blocklists() -> None:
    assert not batch_module.example_looks_usable(
        {"wz_word": "洞洞丝儿", "definition": "裂开小洞", "source_file": "test.jsonl"}
    )
    assert batch_module.example_looks_usable(
        {"wz_word": "物事", "definition": "东西", "source_file": "test.jsonl"}
    )


def test_core_is_blocked_respects_runtime_blocklists() -> None:
    assert batch_module.core_is_blocked("shopping_payment", "百来番钿", "金额说法")
    assert batch_module.core_is_blocked("home_life", "物事", "东西", "活色生香温州话.xlsx")


def test_prompt_examples_for_task_filters_recovery_bad_skeletons() -> None:
    examples = [
        {
            "wz_word": "相伴",
            "definition": "一起走",
            "scene_id": "transport_trip",
            "wz_sentence": "我伉你相伴走，该地方好嬉显罢。",
            "zh_sentence": "我陪你一起走，这个地方很好玩。",
            "source_file": "test.jsonl",
        },
        {
            "wz_word": "相伴",
            "definition": "一起走",
            "scene_id": "transport_trip",
            "wz_sentence": "你𧟰愁寻不着，我伉你相伴走。",
            "zh_sentence": "你怕找不到，我陪你一起走。",
            "source_file": "test.jsonl",
        },
    ]
    prompt_examples, contaminated = batch_module.prompt_examples_for_task(
        "transport_trip", examples, set()
    )
    assert contaminated is True
    assert len(prompt_examples) == 1
    assert prompt_examples[0]["wz_sentence"] == "你𧟰愁寻不着，我伉你相伴走。"


def test_prompt_examples_for_task_prefers_empty_over_contaminated_examples() -> None:
    examples = [
        {
            "wz_word": "相伴",
            "definition": "一起走",
            "scene_id": "transport_trip",
            "wz_sentence": "我伉你相伴走，该地方好嬉显罢。",
            "zh_sentence": "我陪你一起走，这个地方很好玩。",
            "source_file": "test.jsonl",
        }
    ]
    prompt_examples, contaminated = batch_module.prompt_examples_for_task(
        "transport_trip", examples, set()
    )
    assert contaminated is True
    assert prompt_examples == []


def test_prompt_examples_for_task_dedupes_same_skeleton_examples() -> None:
    examples = [
        {
            "wz_word": "合着",
            "definition": "划算",
            "scene_id": "shopping_payment",
            "wz_sentence": "该件衣裳五十番钿买来，真合着显。",
            "zh_sentence": "这件衣服五十块买来，很划算。",
            "source_file": "test.jsonl",
        },
        {
            "wz_word": "合着",
            "definition": "划算",
            "scene_id": "shopping_payment",
            "wz_sentence": "该件衣裳一百番钿买来，真合着显。",
            "zh_sentence": "这件衣服一百块买来，很划算。",
            "source_file": "test.jsonl",
        },
    ]
    prompt_examples, contaminated = batch_module.prompt_examples_for_task(
        "shopping_payment", examples, set()
    )
    assert contaminated is True
    assert prompt_examples == []


def test_dedupe_generated_candidates_limits_and_collapses_same_skeleton() -> None:
    candidates = [
        {"wz": "该件衣裳五十番钿买来，真合着显。", "zh": "这件衣服五十块买来，很划算。"},
        {"wz": "该件衣裳一百番钿买来，真合着显。", "zh": "这件衣服一百块买来，很划算。"},
        {"wz": "你买恁多物事，五十番钿合着不？", "zh": "你买这么多东西，五十块划算吗？"},
        {"wz": "该件衣裳两百番钿买来，真合着。", "zh": "这件衣服两百块买来，很划算。"},
    ]
    deduped = batch_module.dedupe_generated_candidates(
        candidates,
        scene_id="shopping_payment",
        core_word="合着",
        support_words=["五十番钿"],
        approved_modern_terms=[],
    )
    assert len(deduped) <= batch_module.GENERATED_SENTENCE_LIMIT
    assert "该件衣裳五十番钿买来，真合着显。" not in {row["wz"] for row in deduped}
    assert "该件衣裳一百番钿买来，真合着显。" not in {row["wz"] for row in deduped}
    assert "该件衣裳两百番钿买来，真合着。" in {row["wz"] for row in deduped}


def test_dedupe_generated_candidates_filters_recovery_repeaters() -> None:
    candidates = [
        {"wz": "屋里大蛮阵，一日个行用也交关多，真难熬。", "zh": "家里人多，一天的开销很多，真难熬。"},
        {"wz": "天色恁好，有太阳，我走外转嬉嬉。", "zh": "天气很好，有太阳，我出去玩玩。"},
        {"wz": "我伉你相伴走车站大道，你𧟰愁寻不着路。", "zh": "我陪你走车站大道，你不用怕找不到路。"},
    ]
    deduped = batch_module.dedupe_generated_candidates(
        candidates,
        scene_id="transport_trip",
        core_word="相伴",
        support_words=["车站大道"],
        approved_modern_terms=[],
    )
    assert [row["wz"] for row in deduped] == ["我伉你相伴走车站大道，你𧟰愁寻不着路。"]


def test_validate_sentence_flags_global_explicit_block_terms() -> None:
    val = batch_module.validate_sentence(
        "我未吃饭，肚饿起罢，赶紧煮饭。",
        "我还没吃饭，肚子饿了，赶紧做饭。",
        {"我", "未", "吃饭", "煮饭"},
        "吃饭",
        [],
        set(),
        scene_id="food_dining",
        lane="mainline",
        core_tier="stable",
        approved_modern_terms=[],
        banned_terms=[],
        domain_required_terms=[],
        domain_preferred_terms=[],
        domain_blocked_terms=[],
    )
    assert "banned_terms:赶紧" in val["reasons"]


def test_safe_lengthen_sentence_adds_scene_or_core_tail_near_threshold() -> None:
    sentence = "天色恁恶，大家著埭屋里。"
    lengthened = batch_module.safe_lengthen_sentence(
        sentence,
        scene_id="weather_safety",
        core_word="天色",
    )
    assert lengthened != batch_module.clean_wz(sentence)
    assert len(batch_module.clean_wz(lengthened)) >= batch_module.MIN_SENTENCE_LENGTH
    assert len(batch_module.clean_wz(lengthened)) <= batch_module.MAX_SENTENCE_LENGTH
    assert any(tail in lengthened for tail in ("大家小心。", "避雨要紧。"))


def test_safe_lengthen_sentence_skips_bad_bare_completion_marker() -> None:
    sentence = "天色恁个样子，我著埭屋里不出去爻。"
    assert (
        batch_module.safe_lengthen_sentence(
            sentence,
            scene_id="weather_safety",
            core_word="天色",
        )
        == batch_module.clean_wz(sentence)
    )
