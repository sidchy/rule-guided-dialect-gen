# 人工复核工作流

## 1. 导出审查包

使用 `wz-export-review-package` 从某个 run 的 `rule_gate.jsonl` 导出标准 `review.tsv`。

## 2. 填写审查结果

必填列：

- `sample_id`
- `human_review_status`
- `review_reason`
- `reviewer`
- `reviewed_at`

`human_review_status` 支持：

- `pass`
- `fail`
- `pending`

也兼容中文：

- `通过`
- `不通过`
- `待定`

## 3. 回写结构化结果

使用 `wz-import-review-results` 把 TSV 回写到 run 目录内，并生成机器可消费的 review 结果。

## 4. Promotion

使用 `wz-promote-reviewed-samples` 只提升满足以下条件的样本：

- `rule_gate_status=pass`
- `human_review_status=pass`

通过的生成样本写入 `curated/generated_reviewed.jsonl`。

