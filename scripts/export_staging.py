#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "data" / "staging"

PUNCT_TRANSLATION = str.maketrans(
    {
        "\u3000": " ",
        "\t": " ",
        "\r": " ",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
)

PINYIN_PAREN_RE = re.compile(
    r"[(（]\s*[A-Za-zāáǎàōóǒòēéěèīíǐìūúǔùȳǘǚǜüŋɦʔ]+(?:\s+[A-Za-zāáǎàōóǒòēéěèīíǐìūúǔùȳǘǚǜüŋɦʔ]+)*\s*[)）]"
)
MULTISPACE_RE = re.compile(r"\s+")


def clean_cell(value: Any) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def normalize_text(value: str | None) -> str | None:
    if not value:
        return None
    text = value.translate(PUNCT_TRANSLATION)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text or None


def strip_pronunciation(value: str | None) -> str | None:
    if not value:
        return None
    text = PINYIN_PAREN_RE.sub("", value)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_for_training(value: str | None) -> str | None:
    if not value:
        return None
    return normalize_text(strip_pronunciation(value))


def load_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    keep_columns = [c for c in df.columns if not str(c).startswith("Unnamed:")]
    df = df[keep_columns].copy()
    for col in df.columns:
        df[col] = df[col].map(clean_cell)
    return df


def base_record(
    *,
    source_file: str,
    source_sheet: str,
    source_row_id: int,
    record_type: str,
) -> dict[str, Any]:
    return {
        "source_file": source_file,
        "source_sheet": source_sheet,
        "source_row_id": source_row_id,
        "record_type": record_type,
        "wz_text_raw": None,
        "wz_text_norm": None,
        "wz_text_train": None,
        "zh_text_raw": None,
        "zh_text_norm": None,
        "zh_text_train": None,
        "wz_word_raw": None,
        "wz_word_norm": None,
        "wz_word_train": None,
        "mandarin_headword_raw": None,
        "definition_raw": None,
        "definition_norm": None,
        "example_wz_raw": None,
        "example_wz_norm": None,
        "example_wz_train": None,
        "example_zh_raw": None,
        "example_zh_norm": None,
        "example_zh_train": None,
        "notes_raw": None,
        "exclude_from_v1": False,
        "exclude_reasons": [],
    }


def finalize_record(record: dict[str, Any]) -> dict[str, Any]:
    text_fields = [
        "wz_text_raw",
        "zh_text_raw",
        "wz_word_raw",
        "definition_raw",
        "example_wz_raw",
        "example_zh_raw",
        "mandarin_headword_raw",
    ]
    for field in text_fields:
        raw = record.get(field)
        norm_field = field.replace("_raw", "_norm")
        train_field = field.replace("_raw", "_train")
        if norm_field in record:
            record[norm_field] = normalize_text(raw)
        if train_field in record:
            record[train_field] = normalize_for_training(raw)
    record["exclude_reasons"] = sorted(set(record["exclude_reasons"]))
    return record


def export_lin_teacher(path: Path) -> list[dict[str, Any]]:
    df = load_excel(path)
    sheet = "完整语料库（含问题标注）"
    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        record = base_record(
            source_file=path.name,
            source_sheet=sheet,
            source_row_id=idx + 2,
            record_type="sentence_pair",
        )
        record["wz_text_raw"] = row.get("clean_text")
        record["zh_text_raw"] = row.get("mandarin_translation")
        if not record["wz_text_raw"]:
            record["exclude_from_v1"] = True
            record["exclude_reasons"].append("missing_wz_text")
        if not record["zh_text_raw"]:
            record["exclude_from_v1"] = True
            record["exclude_reasons"].append("missing_zh_text")
        records.append(finalize_record(record))
    return records


def export_dictionary(path: Path) -> list[dict[str, Any]]:
    df = load_excel(path)
    sheet = "工作表1"
    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        if not row.get("词条简体"):
            continue
        record = base_record(
            source_file=path.name,
            source_sheet=sheet,
            source_row_id=idx + 2,
            record_type="dictionary_entry",
        )
        record["wz_word_raw"] = row.get("词条简体")
        record["definition_raw"] = row.get("释义简体")
        record["example_wz_raw"] = row.get("方言例句")
        record["example_zh_raw"] = row.get("普通话翻译")
        record["notes_raw"] = row.get("备注")
        if not record["definition_raw"]:
            record["exclude_reasons"].append("missing_definition")
        if record["example_wz_raw"] and not record["example_zh_raw"]:
            record["exclude_reasons"].append("example_missing_translation")
        records.append(finalize_record(record))
    return records


def classify_huose_type(type_value: str | None) -> str:
    if not type_value:
        return "fixed_expression"
    if type_value in {"二字词语", "三字词语", "四字词语"}:
        return "lexicon_entry"
    return "fixed_expression"


def export_huose(path: Path) -> list[dict[str, Any]]:
    df = load_excel(path)
    sheet = "【无单字】活色生香温州话_3.20"
    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        if not row.get("原文"):
            continue
        record = base_record(
            source_file=path.name,
            source_sheet=sheet,
            source_row_id=idx + 2,
            record_type=classify_huose_type(row.get("类型")),
        )
        record["wz_word_raw"] = row.get("原文")
        record["definition_raw"] = row.get("普通话释义")
        record["notes_raw"] = row.get("类型")
        if not record["definition_raw"]:
            record["exclude_from_v1"] = True
            record["exclude_reasons"].append("huose_missing_definition")
        records.append(finalize_record(record))
    return records


def export_lexicon(path: Path) -> list[dict[str, Any]]:
    df = load_excel(path)
    sheet = "Sheet1"
    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        if not row.get("词条"):
            continue
        record = base_record(
            source_file=path.name,
            source_sheet=sheet,
            source_row_id=idx + 2,
            record_type="lexicon_entry",
        )
        record["wz_word_raw"] = row.get("词条")
        record["definition_raw"] = row.get("释义")
        if not record["definition_raw"]:
            record["exclude_reasons"].append("missing_definition")
        records.append(finalize_record(record))
    return records


def build_exam_record(
    *,
    path: Path,
    sheet: str,
    row_id: int,
    wz_word: str | None,
    definition: str | None,
    example_wz: str | None = None,
    example_zh: str | None = None,
) -> dict[str, Any]:
    record_type = "dictionary_entry" if example_wz or example_zh else "lexicon_entry"
    record = base_record(
        source_file=path.name,
        source_sheet=sheet,
        source_row_id=row_id,
        record_type=record_type,
    )
    record["wz_word_raw"] = wz_word
    record["definition_raw"] = definition
    record["example_wz_raw"] = example_wz
    record["example_zh_raw"] = example_zh
    if not record["wz_word_raw"]:
        record["exclude_from_v1"] = True
        record["exclude_reasons"].append("missing_wz_word")
    if not record["definition_raw"]:
        record["exclude_reasons"].append("missing_definition")
    if record["example_wz_raw"] and not record["example_zh_raw"]:
        record["exclude_reasons"].append("example_missing_translation")
    return finalize_record(record)


def flush_exam_entry(
    *,
    path: Path,
    sheet: str,
    entry: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not entry or not any([entry.get("wz_word"), entry.get("definition"), entry.get("examples")]):
        return []
    examples = entry.get("examples") or []
    if examples:
        return [
            build_exam_record(
                path=path,
                sheet=sheet,
                row_id=example["row_id"],
                wz_word=entry.get("wz_word"),
                definition=entry.get("definition"),
                example_wz=example.get("wz"),
                example_zh=example.get("zh"),
            )
            for example in examples
        ]
    return [
        build_exam_record(
            path=path,
            sheet=sheet,
            row_id=entry["row_id"],
            wz_word=entry.get("wz_word"),
            definition=entry.get("definition"),
        )
    ]


def export_term_exam(path: Path) -> list[dict[str, Any]]:
    excel = pd.ExcelFile(path)
    records: list[dict[str, Any]] = []
    for sheet in excel.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        current_entry: dict[str, Any] | None = None
        for idx, raw_row in df.iterrows():
            values = [clean_cell(value) for value in raw_row.tolist()[:4]]
            while len(values) < 4:
                values.append(None)
            wz_word, definition, example_wz, example_zh = values
            if not any(values):
                continue

            starts_new_entry = bool(wz_word or definition)
            has_example_pair = bool(example_wz and example_zh)

            if starts_new_entry:
                records.extend(flush_exam_entry(path=path, sheet=sheet, entry=current_entry))
                current_entry = {
                    "row_id": idx + 1,
                    "wz_word": wz_word,
                    "definition": definition,
                    "examples": [],
                }
                if has_example_pair:
                    current_entry["examples"].append({"row_id": idx + 1, "wz": example_wz, "zh": example_zh})
                elif example_wz or example_zh:
                    current_entry["examples"].append({"row_id": idx + 1, "wz": example_wz, "zh": example_zh})
                continue

            if current_entry is None:
                continue
            if example_wz or example_zh:
                current_entry["examples"].append({"row_id": idx + 1, "wz": example_wz, "zh": example_zh})

        records.extend(flush_exam_entry(path=path, sheet=sheet, entry=current_entry))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(r["record_type"] for r in records)
    excluded = sum(1 for r in records if r["exclude_from_v1"])
    by_source = Counter(r["source_file"] for r in records)
    reasons = Counter()
    for record in records:
        for reason in record["exclude_reasons"]:
            reasons[reason] += 1
    return {
        "total_records": len(records),
        "excluded_from_v1": excluded,
        "kept_for_v1": len(records) - excluded,
        "record_type_counts": dict(by_type),
        "source_counts": dict(by_source),
        "exclude_reason_counts": dict(reasons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export unified staging records from Wenzhou Excel files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write staging outputs.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    records.extend(export_dictionary(ROOT / "【20260327最终版】温州方言词典.xlsx"))
    records.extend(export_huose(ROOT / "【终版】活色生香温州话.xlsx"))
    records.extend(export_term_exam(ROOT / "副本温州话词语考释1-4199(1)(1).xlsx"))
    records.extend(export_term_exam(ROOT / "温州话词语考释8595-12651.xlsx"))
    records.extend(export_lexicon(ROOT / "温州话资源库-词汇-323交付版.xlsx"))

    jsonl_path = output_dir / "staging_records.jsonl"
    summary_path = output_dir / "staging_summary.json"
    csv_path = output_dir / "staging_records.csv"

    write_jsonl(jsonl_path, records)
    pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(build_summary(records), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(records)} records to {jsonl_path}")
    print(f"Wrote CSV to {csv_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
