# 温州话语料工程回环

唯一合法主线：

`原始来源/人工来源 -> 结构化派生 -> 批次运行 -> 规则门 -> 人工复核 -> curated 高置信池 -> 后续扩写/训练`

## 核心规则

1. `generated_candidate` 默认 `upstream_eligible=false`。
2. 只有 `trust_tier in {source, curated}` 的样本允许进入下一轮生成、anchor、seed 或训练导出。
3. `batch_passed.jsonl` 只代表规则门通过，不代表可用训练样本。
4. 所有新 run 必须落在 `runs/<pipeline_name>/<run_id>/` 下，并包含 `config.json`、`results.jsonl`、`rule_gate.jsonl`、`review.tsv`、`summary.json`、`promotion_candidates.jsonl`。
5. 历史实验产物先登记到 `registry/legacy_runs.jsonl`，只有经过 review + promotion 后才允许进入 `curated/`。

## 场景执行策略

当前产品范围仍覆盖多场景语料，但工程 rollout 分层执行：

- `mainline`: `home_life`, `food_dining`, `transport_trip`, `shopping_payment`, `weather_safety`
- `sidecar`: `digital_chat`, `work_study`
- `candidate`: `daily_chat`, `health_medical`

统一配置见 [`configs/scene_policy.json`](../configs/scene_policy.json)。

## 放量门槛

- `mainline`: 至少 50 条已审样本，且人工通过率不低于 0.70，每个主 scene 至少 5 条人工通过
- `sidecar`: 至少 20 条已审样本，并满足 sidecar anchor threshold
- `candidate`: 先做人审和资产盘点，不默认参与当前放量
