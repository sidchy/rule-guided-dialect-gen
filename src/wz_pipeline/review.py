from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from .contracts import apply_contract, normalize_support_words, stable_sample_id
from .grammar_config import load_grammar_config
from .grammar_spec import relevant_spec_labels
from .jsonl import write_jsonl


REVIEW_COLUMNS = [
    "sample_id",
    "scene_id",
    "lane",
    "core_word",
    "support_words",
    "domain_ids",
    "domain_required_terms",
    "domain_blocked_terms",
    "domain_review_notes",
    "wz_sentence",
    "zh_sentence",
    "rule_gate_status",
    "grammar_markers",
    "grammar_auto_flags",
    "grammar_spec_sections",
    "human_review_status",
    "review_reason",
    "reviewer",
    "reviewed_at",
]

PASS_VALUES = {"pass", "通过", "yes", "y"}
FAIL_VALUES = {"fail", "不通过", "no", "n"}
PENDING_VALUES = {"pending", "待定", ""}
RULE_PASS_VALUES = {"pass"}
RULE_FAIL_VALUES = {"fail"}
GRAMMAR_REVIEW_MARKERS = ("爻", "罢", "著埭", "落去", "未", "冇", "不")


def parse_selector_arg(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _normalize_candidate_words(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split("|") if item.strip()]
    return []


def _resolve_core_word(row: dict[str, Any]) -> str:
    core_word = str(row.get("core_word", "")).strip()
    if core_word:
        return core_word
    target_words = _normalize_candidate_words(row.get("target_words"))
    return target_words[0] if target_words else ""


def _resolve_support_words(row: dict[str, Any], core_word: str) -> list[str]:
    support_words = normalize_support_words(row.get("support_words"))
    if support_words:
        return support_words
    target_words = _normalize_candidate_words(row.get("target_words"))
    if not target_words:
        return []
    if core_word and target_words and target_words[0] == core_word:
        return target_words[1:]
    return target_words


def row_mentions_any_term(row: dict[str, Any], terms: set[str]) -> bool:
    if not terms:
        return False
    core_word = _resolve_core_word(row)
    support_words = _resolve_support_words(row, core_word)
    target_words = _normalize_candidate_words(row.get("target_words"))
    haystacks = [
        str(row.get("wz_sentence", "")),
        core_word,
        " ".join(support_words),
        " ".join(target_words),
    ]
    return any(term and any(term in haystack for haystack in haystacks) for term in terms)


def exclude_rows_with_terms(rows: list[dict[str, Any]], terms: set[str]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if not row_mentions_any_term(row, terms)]


def build_grammar_review_metadata(row: dict[str, Any]) -> dict[str, str]:
    sentence = str(row.get("wz_sentence", "")).strip()
    validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
    grammar_reasons = validation.get("grammar_reasons", [])
    review_markers = load_grammar_config().get("review_markers")
    if review_markers is None:
        review_markers = GRAMMAR_REVIEW_MARKERS
    markers = [
        marker
        for marker in review_markers
        if str(marker) and str(marker) in sentence
    ]
    sections = relevant_spec_labels(sentence, list(grammar_reasons))
    if markers and not grammar_reasons:
        grammar_reasons = ["manual_particle_review"]
    return {
        "grammar_markers": "|".join(markers),
        "grammar_auto_flags": "|".join(str(reason) for reason in grammar_reasons if str(reason).strip()),
        "grammar_spec_sections": "|".join(sections),
    }


def export_review_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            core_word = _resolve_core_word(row)
            support_words = _resolve_support_words(row, core_word)
            grammar_metadata = build_grammar_review_metadata(row)
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id and row.get("origin_run_id") and row.get("pipeline_name"):
                sample_id = stable_sample_id(
                    {
                        "origin_run_id": row.get("origin_run_id", ""),
                        "pipeline_name": row.get("pipeline_name", ""),
                        "scene_id": row.get("scene_id", ""),
                        "lane": row.get("lane", ""),
                        "core_word": core_word,
                        "support_words": support_words,
                        "wz_sentence": row.get("wz_sentence", ""),
                        "zh_sentence": row.get("zh_sentence", ""),
                        "source_row_id": row.get("source_row_id", ""),
                    },
                    prefix=str(row.get("sample_prefix") or "gen"),
                )
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "scene_id": row.get("scene_id", ""),
                    "lane": row.get("lane", ""),
                    "core_word": core_word,
                    "support_words": "|".join(support_words),
                    "domain_ids": "|".join(str(item).strip() for item in row.get("domain_ids", []) if str(item).strip()),
                    "domain_required_terms": "|".join(
                        str(item).strip() for item in row.get("domain_required_terms", []) if str(item).strip()
                    ),
                    "domain_blocked_terms": "|".join(
                        str(item).strip() for item in row.get("domain_blocked_terms", []) if str(item).strip()
                    ),
                    "domain_review_notes": " | ".join(
                        str(item).strip() for item in row.get("domain_review_notes", []) if str(item).strip()
                    ),
                    "wz_sentence": row.get("wz_sentence", ""),
                    "zh_sentence": row.get("zh_sentence", ""),
                    "rule_gate_status": normalize_rule_gate_status(
                        str(row.get("rule_gate_status", ""))
                    ),
                    "grammar_markers": grammar_metadata["grammar_markers"],
                    "grammar_auto_flags": grammar_metadata["grammar_auto_flags"],
                    "grammar_spec_sections": grammar_metadata["grammar_spec_sections"],
                    "human_review_status": row.get("human_review_status", "pending"),
                    "review_reason": row.get("review_reason", ""),
                    "reviewer": row.get("reviewer", ""),
                    "reviewed_at": row.get("reviewed_at", ""),
                }
            )


def _normalize_review_status(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in PASS_VALUES:
        return "pass"
    if normalized in FAIL_VALUES:
        return "fail"
    if normalized in PENDING_VALUES:
        return "pending"
    return "pending"


def normalize_rule_gate_status(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in RULE_PASS_VALUES:
        return "pass"
    if normalized in RULE_FAIL_VALUES:
        return "fail"
    return "not_run"


def prepare_review_rows(
    rows: list[dict[str, Any]],
    *,
    origin_run_id: str = "",
    pipeline_name: str = "",
    sample_prefix: str = "gen",
    default_rule_gate_status: str = "not_run",
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["origin_run_id"] = str(item.get("origin_run_id") or origin_run_id)
        item["pipeline_name"] = str(item.get("pipeline_name") or pipeline_name)
        item["core_word"] = _resolve_core_word(item)
        item["support_words"] = _resolve_support_words(item, item["core_word"])
        item["rule_gate_status"] = normalize_rule_gate_status(
            str(item.get("rule_gate_status") or default_rule_gate_status)
        )
        item["human_review_status"] = _normalize_review_status(
            str(item.get("human_review_status", "pending"))
        )
        item["sample_prefix"] = str(item.get("sample_prefix") or sample_prefix)
        if not str(item.get("sample_id", "")).strip() and item["origin_run_id"] and item["pipeline_name"]:
            item["sample_id"] = stable_sample_id(
                {
                    "origin_run_id": item["origin_run_id"],
                    "pipeline_name": item["pipeline_name"],
                    "scene_id": item.get("scene_id", ""),
                    "lane": item.get("lane", ""),
                    "core_word": item["core_word"],
                    "support_words": item["support_words"],
                    "wz_sentence": item.get("wz_sentence", ""),
                    "zh_sentence": item.get("zh_sentence", ""),
                    "source_row_id": item.get("source_row_id", ""),
                },
                prefix=item["sample_prefix"],
            )
        prepared.append(item)
    return prepared


def select_review_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int | None = None,
    seed: int = 42,
    rule_pass_only: bool = True,
) -> list[dict[str, Any]]:
    filtered = [
        dict(row)
        for row in rows
        if not rule_pass_only or normalize_rule_gate_status(str(row.get("rule_gate_status", "pass"))) == "pass"
    ]
    if limit is None or limit <= 0 or limit >= len(filtered):
        return filtered

    rng = random.Random(seed)
    scene_groups: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in filtered:
        scene = str(row.get("scene_id", "") or "unknown")
        subgroup = (str(row.get("lane", "")), str(row.get("core_word", "")))
        scene_groups[scene][subgroup].append(row)

    scene_order = sorted(scene_groups)
    rng.shuffle(scene_order)
    scene_cycles: dict[str, deque[tuple[str, str]]] = {}
    for scene, subgroup_map in scene_groups.items():
        subgroup_order = list(subgroup_map)
        rng.shuffle(subgroup_order)
        scene_cycles[scene] = deque(subgroup_order)
        for subgroup_rows in subgroup_map.values():
            rng.shuffle(subgroup_rows)

    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        progressed = False
        for scene in scene_order:
            cycle = scene_cycles.get(scene)
            if not cycle:
                continue
            subgroup_map = scene_groups[scene]
            rotations = len(cycle)
            for _ in range(rotations):
                subgroup = cycle[0]
                subgroup_rows = subgroup_map.get(subgroup, [])
                if subgroup_rows:
                    selected.append(subgroup_rows.pop())
                    cycle.rotate(-1)
                    progressed = True
                    if not subgroup_rows:
                        subgroup_map.pop(subgroup, None)
                        try:
                            cycle.remove(subgroup)
                        except ValueError:
                            pass
                    break
                try:
                    cycle.popleft()
                except IndexError:
                    break
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def filter_review_rows(
    rows: list[dict[str, Any]],
    *,
    scenes: set[str] | None = None,
    lanes: set[str] | None = None,
    rule_pass_only: bool = True,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    scene_filter = scenes or set()
    lane_filter = lanes or set()
    for row in rows:
        scene_id = str(row.get("scene_id", "")).strip()
        lane = str(row.get("lane", "")).strip()
        if scene_filter and scene_id not in scene_filter:
            continue
        if lane_filter and lane not in lane_filter:
            continue
        if rule_pass_only and normalize_rule_gate_status(str(row.get("rule_gate_status", "pass"))) != "pass":
            continue
        selected.append(dict(row))
    return selected


def limit_review_rows_per_scene(
    rows: list[dict[str, Any]],
    *,
    limit_per_scene: int,
    seed: int = 42,
    rule_pass_only: bool = True,
) -> list[dict[str, Any]]:
    if limit_per_scene <= 0:
        return list(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("scene_id", "")).strip() or "unknown"].append(dict(row))
    limited: list[dict[str, Any]] = []
    for offset, scene_id in enumerate(sorted(grouped)):
        limited.extend(
            select_review_rows(
                grouped[scene_id],
                limit=limit_per_scene,
                seed=seed + offset,
                rule_pass_only=rule_pass_only,
            )
        )
    return limited


def import_review_tsv(
    path: Path,
    *,
    origin_run_id: str,
    pipeline_name: str,
    policy_version: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            status = _normalize_review_status(str(row.get("human_review_status", "")))
            rule_gate_status = normalize_rule_gate_status(str(row.get("rule_gate_status", "")))
            structured = apply_contract(
                {
                    "sample_id": row.get("sample_id", ""),
                    "scene_id": row.get("scene_id", ""),
                    "lane": row.get("lane", ""),
                    "core_word": row.get("core_word", ""),
                    "support_words": row.get("support_words", ""),
                    "wz_sentence": row.get("wz_sentence", ""),
                    "zh_sentence": row.get("zh_sentence", ""),
                    "review_reason": row.get("review_reason", "").strip(),
                    "reviewer": row.get("reviewer", "").strip(),
                    "reviewed_at": row.get("reviewed_at", "").strip() or datetime.now().isoformat(),
                },
                source_type="generated_candidate",
                origin_run_id=origin_run_id,
                pipeline_name=pipeline_name,
                policy_version=policy_version,
                scene_id=str(row.get("scene_id", "")),
                lane=str(row.get("lane", "")),
                core_word=str(row.get("core_word", "")),
                support_words=str(row.get("support_words", "")),
                rule_gate_status=rule_gate_status,
                human_review_status=status,
                trust_tier="reviewed" if status in {"pass", "fail"} else "candidate",
                upstream_eligible=False,
            )
            rows.append(structured)
    return rows


def write_review_results(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def build_human_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reviewed = [row for row in rows if row.get("human_review_status") in {"pass", "fail"}]
    passed = [row for row in reviewed if row.get("human_review_status") == "pass"]
    scene_counts: dict[str, Counter[str]] = defaultdict(Counter)
    lane_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in reviewed:
        scene_counts[row.get("scene_id", "")][row["human_review_status"]] += 1
        lane_counts[row.get("lane", "")][row["human_review_status"]] += 1
    return {
        "reviewed_count": len(reviewed),
        "human_pass_count": len(passed),
        "scene_human_pass_rate": {
            scene: round(counter["pass"] / max(1, counter["pass"] + counter["fail"]), 4)
            for scene, counter in scene_counts.items()
        },
        "lane_human_pass_rate": {
            lane: round(counter["pass"] / max(1, counter["pass"] + counter["fail"]), 4)
            for lane, counter in lane_counts.items()
        },
    }
