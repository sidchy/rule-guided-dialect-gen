# 当前要求

## 当前唯一主目标

当前项目的唯一主目标是构建高质量、可追溯、可复核的温州话生成/改写语料资产。

当前优先能力：

- `wz_rewrite_keep_skeleton`
- `scene_conditioned_wz_generation`
- `batch_sentence_expansion`

当前不再作为默认主线：

- 开放域 `zh_to_wz_sentence` 微调

## 当前语法真相源

生成时涉及功能词、体貌、否定、处置句、语气词时，优先遵守：

- [`温州话生成语法规范_潘悟云原书67-83页.md`](./温州话生成语法规范_潘悟云原书67-83页.md)

## 工程硬规则

1. `generated_candidate` 默认不能回流上游。
2. 只有 `source` 和 `curated` 样本可以进入 anchor、seed、prompt example 或训练导出。
3. `batch_passed.jsonl` 只是规则通过，不代表最终可用。
4. 历史 few-shot / dense / compound 资产必须先登记，再 `review + promotion`，才能进入 `curated`。
5. 机器指标和人工指标必须分开汇报，不允许把 `pass_rate` 当成人工通过率。

## 场景要求

这些场景都仍然属于需求范围：

- `home_life`
- `food_dining`
- `transport_trip`
- `shopping_payment`
- `weather_safety`
- `digital_chat`
- `work_study`
- `daily_chat`
- `health_medical`

但当前工程 rollout 不同层级：

- `mainline`
  - `home_life`
  - `food_dining`
  - `transport_trip`
  - `shopping_payment`
  - `weather_safety`
- `sidecar`
  - `digital_chat`
  - `work_study`
- `candidate`
  - `daily_chat`
  - `health_medical`

这表示：

- `sidecar` 和 `candidate` 不是被删除
- 它们只是当前不和 mainline 一起默认放量
- 只有质量闭环跑通、anchor/人审条件满足后，才会升级

## 当前优先 review 顺序

1. `fewshot_batch_deepseek_mainline_v4`
2. `fewshot_batch_deepseek_mainline_v3`
3. `work_study` 相关包
4. `daily_chat` / `health_medical`
5. `digital_chat`

`digital_chat` 排在后面不是因为不需要，而是因为当前可信 anchor 还不足。
