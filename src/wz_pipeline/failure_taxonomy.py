from __future__ import annotations

from typing import Iterable


FAILURE_BUCKET_ORDER = ("grammar", "domain", "naturalness")

RULE_REASON_BUCKET_MAP = {
    "grammar": "grammar",
    "mandarin_markers": "grammar",
    "source_surface_missing": "grammar",
    "domain_required_terms_missing": "domain",
    "domain_blocked_terms": "domain",
}


def reason_key(reason: str | None) -> str:
    return str(reason or "").split(":", 1)[0].strip()


def classify_rule_reason_bucket(reason: str | None) -> str:
    key = reason_key(reason)
    if not key:
        return ""
    if key.startswith("domain_"):
        return "domain"
    return RULE_REASON_BUCKET_MAP.get(key, "naturalness")


def classify_rule_fail_buckets(reasons: Iterable[str] | None) -> list[str]:
    buckets = {
        bucket
        for reason in (reasons or [])
        for bucket in [classify_rule_reason_bucket(reason)]
        if bucket
    }
    return [bucket for bucket in FAILURE_BUCKET_ORDER if bucket in buckets]


def classify_critic_fail_buckets(
    *,
    critic_pass: bool,
    critic_issue_types: list[str] | None,
    critic_grammar_pass: bool,
    critic_grammar_issue_types: list[str] | None,
    critic_domain_pass: bool,
    critic_domain_issue_types: list[str] | None,
) -> list[str]:
    buckets: set[str] = set()
    if critic_grammar_issue_types or not critic_grammar_pass:
        buckets.add("grammar")
    if critic_domain_issue_types or not critic_domain_pass:
        buckets.add("domain")
    if critic_issue_types or (not critic_pass and not buckets):
        buckets.add("naturalness")
    return [bucket for bucket in FAILURE_BUCKET_ORDER if bucket in buckets]
