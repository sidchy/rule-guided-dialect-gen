# 受控生成流水线交接文档

日期：2026-03-30

## 1. 当前状态

这个仓库已经清掉了旧 `stage1` 翻译 / 微调路线的主文件。当前默认主线只有一条：

- 基于现有温州话句子，构建“受控句子生成流水线”

当前不要把主任务理解成：

1. 普通话 -> 温州话通用翻译
2. 先做 LoRA / QLoRA 微调
3. 自由扩写大量温州话句子

当前要做的是：

1. 从现有句子抽骨架
2. 标出可替换实词槽位
3. 按场景和语义角色构造任务
4. 让模型在强约束下改写
5. 自动筛选
6. 只保留适合 AI 日常对话和语音训练的句子

## 2. 当前仓库里真正有用的内容

### 2.1 数据

当前主线相关目录：

1. `data/cleaned/`
   已清洗、去重后的主记录
2. `data/llm_review/`
   词头抽取和相关复核结果
3. `data/controlled_generation/`
   当前主线全部资产、任务、候选、筛选和报告

### 2.2 脚本

当前主线核心脚本：

1. `scripts/build_controlled_generation_assets.py`
2. `scripts/build_example_mined_lexicon.py`
3. `scripts/tag_scene_semantics.py`
4. `scripts/review_example_mined_scene_tags.py`
5. `scripts/build_generation_pilot.py`
6. `scripts/generate_controlled_sentences.py`
7. `scripts/filter_controlled_generation_candidates.py`
8. `scripts/review_generation_naturalness.py`

保留但不是当前主线核心：

1. `scripts/export_staging.py`
2. `scripts/dedup_staging.py`
3. `scripts/extract_headwords_heuristic.py`
4. `scripts/extract_mandarin_headwords.py`

## 3. 已经完成到什么程度

### 3.1 资产已经建起来了

当前已经有：

1. 可替换实词表
   `data/controlled_generation/assets/replaceable_lexicon.jsonl`
2. 高频生活词补充表
   `data/controlled_generation/assets/example_mined_lexicon.jsonl`
   `data/controlled_generation/assets/example_mined_lexicon_reviewed.jsonl`
3. 外部新词和地名总表
   `data/controlled_generation/assets/external_term_catalog.json`
4. 场景 taxonomy
   `data/controlled_generation/assets/scene_taxonomy.json`
5. 语用功能 taxonomy
   `data/controlled_generation/assets/speech_function_taxonomy.json`
6. 骨架模板库
   `data/controlled_generation/skeletons/skeleton_templates.jsonl`
7. 全量任务
   `data/controlled_generation/tasks/generation_tasks_full.jsonl`
8. pilot 任务
   `data/controlled_generation/tasks/generation_tasks_pilot_v2.jsonl`

### 3.2 已经做过的约束

当前系统已经不是最早那种“拿个骨架就塞词”的版本了，已经做过这些收紧：

1. 先做 `topic_scene`
2. 再做 `speech_function`
3. 词条、骨架、外部词都带场景标签和置信度
4. 高频生活词单独挖掘，并做过低置信重标注
5. 外部词要过：
   - 主场景匹配
   - 槽位语义类匹配
6. 骨架里只要还有 `unknown` 槽位，就不注现代词 / 外部词
7. 取消了“找不到词就回退整池乱挑”
8. 两槽模板默认至少改两个槽位
9. 规则筛选之后还有一层 LLM critic

## 4. 当前最重要的实验结论

最近一轮严格 pilot 是：

1. `data/controlled_generation/filtered/pilot_v5_qwen/filter_summary.json`
2. `data/controlled_generation/filtered/pilot_v5_qwen/critic_summary.json`

结论：

1. 原始候选：`59`
2. 规则通过：`32`
3. 多槽替换通过：`22`
4. critic 真正通过：`0`

critic 失败类型：

1. `unnatural_wz`: `31`
2. `semantic_mismatch`: `29`
3. `forced_insertion`: `7`
4. `scene_mismatch`: `1`

这说明当前问题不是：

1. 没有数据
2. 没有场景
3. 没有约束

当前真正的问题是：

1. 任务结构仍然给模型太多自由度
2. 模型仍然在自己决定“哪个词落哪个槽位”
3. 多槽改写时，形式上过关，语义和口语自然度却不够

## 5. 当前最缺的是什么

当前最缺的不是更多词，而是“槽位级别的强绑定任务结构”。

具体缺口：

1. 缺少 `slot_id -> candidate_pool`
2. 缺少 `required_slot_replacements`
3. 缺少“只能从该槽位池选词”的 prompt
4. 缺少“模型是否真的用了指定槽位候选”的校验
5. 缺少基于新任务结构的下一轮 pilot

## 6. 下一步应该直接做什么

### 6.1 首要任务

把当前任务格式从：

- `required_words`
- `preferred_words`

升级成：

- `slot_candidate_pools`
- `required_slot_replacements`

推荐结构：

```json
{
  "slot_candidate_pools": {
    "SLOT_1": [
      {
        "term": "医院",
        "semantic_class": "institution",
        "source_type": "lexicon"
      },
      {
        "term": "鹿城区人民医院",
        "semantic_class": "institution",
        "source_type": "place_name"
      }
    ],
    "SLOT_2": [
      {
        "term": "医师",
        "semantic_class": "person",
        "source_type": "lexicon"
      }
    ]
  },
  "required_slot_replacements": ["SLOT_1", "SLOT_2"]
}
```

### 6.2 第二步

修改生成 prompt，明确写死：

1. 只能从对应槽位池中选词
2. 不能跨槽位交换候选词
3. 指定槽位必须替换
4. 固定块必须保留
5. 不允许自由加塞候选池之外的新核心实词

### 6.3 第三步

如有必要，在筛选脚本里新增：

1. 候选词是否来自对应 `slot_candidate_pool`
2. 是否出现了跨槽位使用
3. 是否仍满足多槽替换要求

### 6.4 第四步

重跑一轮新的小规模 pilot，只要 `12` 到 `24` 条任务。

对比对象固定为 `pilot_v5_qwen`，比较：

1. `accepted_rows`
2. `accepted_multi_slot_changes`
3. `critic_pass_rows`
4. `issue_type_counts`

## 7. LLM 应该如何参与

LLM 可以参与，但只能参与“受约束决策”。

允许 LLM 参与：

1. 低置信标签复判
2. 为某个槽位从更大池子里裁剪 3 到 8 个最合适候选词
3. 规则通过样本的 critic / rerank

不要让 LLM 自由决定：

1. 一个词该进哪个槽位
2. 一个句子改几个槽位
3. 一个外部词该不该随便出现在句子里

## 8. 接手 agent 的工作边界

接手 agent 不要做这些事：

1. 不要恢复旧 `stage1` 路线
2. 不要重做已有的场景与词表体系
3. 不要把任务改回自由 required words
4. 不要先去做微调

接手 agent 应该直接改这些文件：

1. `scripts/build_controlled_generation_assets.py`
2. `scripts/generate_controlled_sentences.py`
3. 必要时 `scripts/filter_controlled_generation_candidates.py`

## 9. 接手前必读

1. `README.md`
2. `HANDOFF_CONTROLLED_GENERATION_2026-03-30.md`
3. `prompts/agent_takeover_controlled_generation_2026-03-30.md`
4. `scripts/build_controlled_generation_assets.py`
5. `scripts/generate_controlled_sentences.py`
6. `scripts/filter_controlled_generation_candidates.py`
7. `scripts/review_generation_naturalness.py`
8. `data/controlled_generation/reports/task_summary.json`
9. `data/controlled_generation/filtered/pilot_v5_qwen/filter_summary.json`
10. `data/controlled_generation/filtered/pilot_v5_qwen/critic_summary.json`

## 10. 一句话结论

当前项目不是缺语料，而是缺“槽位绑定候选池”的下一层任务结构。下一步不要再盲目放大生成量，先把任务结构改对，再跑新的小规模验证。
