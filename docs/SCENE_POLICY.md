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
  - `health_medical`
  - `work_study`
  - `digital_ai_assistant`
  - `digital_messaging_call`
  - `digital_device_trouble`
  - `digital_app_operation`
- `sidecar_scenes`
  - 当前为空
- `candidate_scenes`
  - `daily_chat`
  - `digital_chat`

## 解释

产品层上，上述场景都属于需求场景。工程层上，`mainline_focus_scenes` 会默认进入当前批量生成主线。

当前 scene policy 有两个新增约定：

- 允许把一个宽父场景拆成多个主线子场景。例如 `digital_chat` 不再直接放量，而是拆成 `digital_ai_assistant / digital_messaging_call / digital_device_trouble / digital_app_operation`
- `default_scenes` 和 `priority_scenes` 建议直接放子场景，而不是只放父场景。这样 allocator、speech-act 轮转和 review 汇总都能在子场景粒度上均衡

如果后续某个新场景仍需要额外 anchor gate 或专项 review，再放回 `sidecar_scenes`。
