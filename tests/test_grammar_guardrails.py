from wz_pipeline.grammar_guardrails import (
    build_grammar_repair_user_prompt,
    grammar_validation_reasons,
    should_attempt_grammar_repair,
)


def test_grammar_validation_accepts_basic_completion_pattern() -> None:
    assert grammar_validation_reasons("饭吃爻罢。") == []


def test_grammar_validation_rejects_known_high_risk_patterns() -> None:
    assert "completion_object_order" in grammar_validation_reasons("跳爻舞再走。")
    assert "stative_progressive" in grammar_validation_reasons("我著埭坐。")
    assert "dynamic_postposed_zhedai" in grammar_validation_reasons("渠走著埭。")


def test_grammar_validation_rejects_mandarin_negation_and_bad_followup() -> None:
    reasons = grammar_validation_reasons("西瓜切爻摆桌能界。")
    assert any(reason.startswith("completion_followed_by_clause:") for reason in reasons)
    assert "mandarin_negation" in grammar_validation_reasons("我没有吃。")
    assert "northern_wu_lexeme_gaotou" in grammar_validation_reasons("黑板高头写个字。")


def test_should_attempt_grammar_repair_focuses_on_particles() -> None:
    assert should_attempt_grammar_repair("饭吃爻罢。") is True
    assert should_attempt_grammar_repair("屋里真热。") is False


def test_build_grammar_repair_user_prompt_contains_context() -> None:
    prompt = build_grammar_repair_user_prompt(
        scene_id="food_dining",
        core_word="西瓜",
        support_words=["冰箱"],
        wz_sentence="西瓜切爻摆桌能界。",
        zh_sentence="西瓜切好摆上桌了。",
        reasons=["completion_followed_by_clause:摆"],
    )
    assert "food_dining" in prompt
    assert "西瓜" in prompt
    assert "completion_followed_by_clause:摆" in prompt
