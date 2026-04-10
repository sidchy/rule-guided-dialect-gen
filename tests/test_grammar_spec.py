from wz_pipeline.grammar_spec import generation_spec_excerpt, relevant_spec_labels


def test_generation_spec_excerpt_reads_markdown_sections() -> None:
    excerpt = generation_spec_excerpt(max_lines=1)
    assert "生成总原则" in excerpt
    assert "批量生成硬约束清单" in excerpt
    assert "不推荐直接生成的高风险式" in excerpt
    assert "完成体：`爻`" in excerpt
    assert "否定句：`冇`" in excerpt or "`冇` 的否定用法" in excerpt


def test_relevant_spec_labels_map_sentence_markers_and_reasons() -> None:
    labels = relevant_spec_labels("饭吃爻罢，渠还未走归。", ["sentence_final_ba_overused"])
    assert "2. 生成总原则" in labels[0]
    assert "3.1 完成体：`爻`" in labels
    assert "3.4 已然体：重读 `罢` 与句末轻读 `罢`" in labels
    assert "9.2 `未`" in labels


def test_relevant_spec_labels_cover_backported_reason_mappings() -> None:
    labels = relevant_spec_labels("你阿乜妆个物事？", ["a_mie_as_how_question", "mao_for_unfinished_event"])
    assert "13. 不推荐直接生成的高风险式" in labels
    assert "9.2 `未`" in labels
