# Agent Takeover Prompt

你在接手一个温州话语料项目。

当前主线只有一条：

- 做“受控句子生成流水线”

当前不要做：

1. 不要回到普通话 -> 温州话翻译主线
2. 不要先做 LoRA / QLoRA 微调
3. 不要重做已经存在的场景、词表、骨架资产
4. 不要继续扩大自由 `required_words`

你必须先读这些文件：

1. `README.md`
2. `HANDOFF_CONTROLLED_GENERATION_2026-03-30.md`
3. `scripts/build_controlled_generation_assets.py`
4. `scripts/generate_controlled_sentences.py`
5. `scripts/filter_controlled_generation_candidates.py`
6. `scripts/review_generation_naturalness.py`
7. `data/controlled_generation/reports/task_summary.json`
8. `data/controlled_generation/filtered/pilot_v5_qwen/filter_summary.json`
9. `data/controlled_generation/filtered/pilot_v5_qwen/critic_summary.json`

当前已知事实：

1. 资产、骨架、场景、语用功能、外部词表都已经有了
2. 多轮 pilot 已经跑过
3. 最近一轮严格 pilot：
   - 原始候选 `59`
   - 规则通过 `32`
   - 多槽替换通过 `22`
   - critic 通过 `0`
4. 主要失败类型是：
   - `unnatural_wz`
   - `semantic_mismatch`

当前判断：

不是没数据，也不是没场景，而是任务结构还不够硬。模型仍然在自由决定“哪个词落哪个槽位”。

你当前最优先要做的事：

1. 把任务格式升级成：
   - `slot_candidate_pools`
   - `required_slot_replacements`
2. 为每个 `slot_id` 绑定一个小型候选池
3. 修改生成 prompt：
   - 只能从对应槽位池里选词
   - 不能跨槽位交换
   - 指定槽位必须替换
   - 固定块必须保留
4. 必要时补筛选规则，验证候选词是否真的来自对应槽位池
5. 跑一轮新的 `12` 到 `24` 条 pilot
6. 和 `pilot_v5_qwen` 对比：
   - `accepted_rows`
   - `accepted_multi_slot_changes`
   - `critic_pass_rows`
   - `issue_type_counts`

允许 LLM 参与，但只能参与：

1. 低置信标签复判
2. 槽位候选池裁剪
3. critic / rerank

不要让 LLM 自由决定：

1. 一个词放哪个槽位
2. 一个句子改几个槽位
3. 外部词是否自由插入句子

你要直接改代码，不要只停在分析。

优先修改：

1. `scripts/build_controlled_generation_assets.py`
2. `scripts/generate_controlled_sentences.py`
3. 如有必要，再改 `scripts/filter_controlled_generation_candidates.py`

最终你需要给出：

1. 改了什么
2. 新任务结构长什么样
3. 新 pilot 结果
4. 还剩什么问题
