from wz_pipeline.grammar_guardrails import grammar_validation_reasons


def test_recovery_guardrails_flag_old_bad_ba_and_as_templates() -> None:
    assert "ba_after_request_marker" in grammar_validation_reasons("恁个天色，预警讲会落大雨，大家快走归罢。")
    assert "ba_clause_chain_after_completion" in grammar_validation_reasons("我外卖点爻罢，等渠送来就吃饭。")
    assert "a_shi_template_drift" in grammar_validation_reasons("该部手机阿是又打电话打不底。")


def test_recovery_guardrails_flag_modal_with_qi_or_luqu() -> None:
    assert "future_start_marker_after_modal" in grammar_validation_reasons("我未吃晚饭，想点外卖吃起。")
    assert "future_continuative_marker_after_modal" in grammar_validation_reasons("外卖生意好，渠想继续做落去。")


def test_recovery_guardrails_flag_scalar_wo_and_bad_xian_shapes() -> None:
    assert "wo_used_for_scalar_also" in grammar_validation_reasons("我沃还未到，你急阿乜。")
    assert "subjective_emotion_xianxian" in grammar_validation_reasons("我急显急，讲也讲不清。")
    assert "bare_xian_degree" in grammar_validation_reasons("屋里大蛮阵，靠阿爸一个人做事干，困难显。")
    assert "bare_xian_degree" in grammar_validation_reasons("屋里大蛮阵，靠阿爸一个人做事干，困难显显。")
    assert "bare_xian_degree" in grammar_validation_reasons("我伉你相伴走车站大道，该条路好走显罢。")


def test_recovery_guardrails_flag_old_start_marker_and_northern_wu_forms() -> None:
    assert "bare_start_marker_qi" in grammar_validation_reasons("我伉你相伴走起，该条路我熟显罢。")
    assert "completion_ba_motion_or_setup" in grammar_validation_reasons("你𧟰愁寻不着高铁站，我伉你相伴走爻罢。")
    assert "northern_wu_lexeme_jiaoguan" in grammar_validation_reasons("外卖吃起落去，钞票用爻交关快。")


def test_recovery_guardrails_keep_basic_completion_sentence() -> None:
    assert grammar_validation_reasons("饭吃爻罢。") == []
