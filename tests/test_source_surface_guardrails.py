from wz_pipeline.source_surface_guardrails import detect_unsupported_surface_terms, source_surface_count


def _mock_stats() -> dict:
    return {
        "term_counts": {
            "该个": 12,
            "车钻": 6,
            "料车": 8,
            "走来": 20,
            "走去": 18,
            "寻不着": 4,
        },
        "ngram_counts": {
            "该个": 12,
            "车钻": 6,
            "料车": 8,
            "走来": 20,
            "走去": 18,
            "寻不着": 4,
            "电脑": 4,
        },
        "trusted_terms": frozenset({"该个", "车钻", "料车", "走来", "走去", "寻不着", "电脑"}),
        "trusted_single_chars": frozenset({"我", "放", "个", "真"}),
        "max_term_len": 3,
    }


def test_detect_unsupported_surface_terms_flags_absent_surface() -> None:
    findings = detect_unsupported_surface_terms(
        "该个车钻放料车高头，我走来走去寻不着。",
        protected_terms=["料车", "车钻"],
        stats=_mock_stats(),
    )
    assert findings == [{"surface": "高头", "source_count": 0}]


def test_detect_unsupported_surface_terms_allows_present_source_forms() -> None:
    findings = detect_unsupported_surface_terms(
        "我走来走去寻不着电脑。",
        protected_terms=["电脑"],
        stats=_mock_stats(),
    )
    assert findings == []


def test_source_surface_count_reads_term_or_ngram_hits() -> None:
    assert source_surface_count("走来", stats=_mock_stats()) == 20
    assert source_surface_count("高头", stats=_mock_stats()) == 0
