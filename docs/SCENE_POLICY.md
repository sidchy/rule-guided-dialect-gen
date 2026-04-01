# Scene Policy

当前 scene policy 的唯一配置文件是：

- [`configs/scene_policy.json`](../configs/scene_policy.json)

## 分层

- `mainline_focus_scenes`
  - 当前默认 mainline 放量场景
- `sidecar_scenes`
  - 仍属需求范围，但需要额外 anchor / review gate
- `candidate_scenes`
  - 需求存在，但当前不默认进入放量
- `default_scenes`
  - few-shot batch 默认使用的场景
- `priority_scenes`
  - 任务分配和回退时使用的全量优先顺序

## 当前配置

- `mainline_focus_scenes`
  - `home_life`
  - `food_dining`
  - `transport_trip`
  - `shopping_payment`
  - `weather_safety`
- `sidecar_scenes`
  - `digital_chat`
  - `work_study`
- `candidate_scenes`
  - `daily_chat`
  - `health_medical`

## 解释

产品层上，上述场景都属于需求场景。工程层上，只有 `mainline_focus_scenes` 会默认进入当前批量生成主线。其余场景需要先满足各自 gate，再决定是否升级。
