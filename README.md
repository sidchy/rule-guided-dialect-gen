# Dialect Speech Corpus Toolkit / 方言语音语料工具箱

用大模型从方言词典和例句出发，批量生成语音训练用的方言句子。  
支持语法规则自动校验 + 人工审核闭环 + 数据溯源。  
目前已在**温州话**上验证，设计为可泛化到任意方言。

---

## 这是什么

如果你有一份方言词典（带例句和释义），想把它扩充成几千上万条适合语音训练的方言句子，这个工具箱能帮你：

1. **从词典里提取种子句子** — 自动清洗、去重、按长度筛选
2. **用 LLM 批量生成新句子** — 以种子句子为锚点，受控地生成带场景的新句子
3. **自动语法校验** — 用你定义的方言语法规则过滤明显错误
4. **导出人工审核包** — TSV 格式，专家逐句审核
5. **审核通过后晋级为训练数据** — 严格的 trust tier 保障数据质量

```
词典 → 种子句子 → LLM 生成 → 规则校验 → 人工审核 → 训练语料
         ↑                                              │
         └──────── 审核通过的句子可以反哺下一轮 ─────────┘
```

---

## 快速开始

### 1. 安装

```bash
git clone <repo-url>
cd dialect-corpus-toolkit
pip install -e .
```

### 2. 创建你的方言项目

```bash
cp -r dialects/_template dialects/你的方言
# 例如：cp -r dialects/_template dialects/cantonese
```

### 3. 准备最基础的数据

至少需要一份清洗后的词典 JSONL 文件，放到 `dialects/你的方言/data/cleaned/` 目录下。

每行格式（字段名可配置）：
```json
{
  "wz_word_train": "方言词",
  "definition_norm": "释义",
  "example_wz_train": "方言例句",
  "example_zh_train": "普通话翻译"
}
```

### 4. 配置 LLM API

复制并编辑环境变量文件：
```bash
cp .env.example .env
# 编辑 .env，填入你的 API 密钥
```

### 5. 编辑方言配置

```bash
# 编辑主配置
vim dialects/你的方言/dialect.yaml
```

最少需要改：`dialect_name`、`dialect_code`、`sentence_length`。

### 6. 跑你的第一批

```bash
# Step 1: 从词典提取种子句子
wz-extract-training-sentences

# Step 2: 批量生成
wz-generate-fewshot-batch --tasks 20 --provider deepseek

# 如果想让上一轮失败分析反哺这一轮配额和采样
wz-generate-fewshot-batch --tasks 20 --provider deepseek --feedback-run-id <上一次的 run_id>

# Step 3: 导出审核包
wz-export-review-package --run-id <上一步输出的 run_id>
```

---

## 你的方言项目长什么样

```
dialects/你的方言/
│
├── dialect.yaml              ← 【入口1】主配置文件
│
├── configs/                  ← 【入口2-4】场景、领域和策略配置
│   ├── scene_policy.json        场景分层（哪些先做、哪些后做）
│   ├── scene_catalog.json       场景详情（关键词、现代词汇）
│   ├── domain_catalog.json      领域词表（专用词、禁用词、备注）
│   ├── trust_policy.json        数据信任层级（通常不改）
│   └── pipeline_thresholds.json 质量门槛
│
├── grammar/                  ← 【入口5-6】语法规则
│   ├── grammar_spec.md          你的语法参考文档
│   └── grammar_rules.yaml       机器可执行的语法校验规则
│
├── prompts/                  ← 【入口7】LLM Prompt
│   ├── generation_system.md     生成 prompt 系统消息
│   ├── generation_user.md       生成 prompt 用户消息模板
│   └── repair_system.md         语法修补 prompt
│
├── data/                     ← 【入口8】你的数据
│   ├── source_dictionaries/     原始词典文件（xlsx 等）
│   ├── cleaned/                 清洗后的 JSONL
│   ├── extracted_training_sentences/  种子句子
│   └── controlled_generation/
│       └── assets/              词表、场景种子等
│
├── runs/                     ← 每次运行的完整产物（自动生成）
├── registry/                 ← 运行记录（自动生成）
└── curated/                  ← 最终审核通过的训练语料
```

---

## 配置解析与联动规则

这部分决定“文件放在哪”“配置改了以后哪条 pipeline 会受影响”。

### 1. 当前激活哪个方言

- 运行时优先读取环境变量 `WZ_PIPELINE_DIALECT`
- 如果没设，默认用 `wenzhou`
- 所以切换方言最直接的方式是：

```bash
export WZ_PIPELINE_DIALECT=你的方言目录名
```

### 2. 路径怎么解析

- `dialect.yaml` 里的相对路径，默认都相对于 `dialects/<dialect>/` 解析
- 如果本地方言 overlay 文件或目录不存在，运行时会回退到项目根目录已有资源
- 当前会自动回退的主要目录有：`configs/`、`data/`、`runs/`、`registry/`、`curated/`、`prompts/`
- 温州话还额外对 `grammar_spec.md` 和 `scene_catalog.json` 做了兼容回退
- `domain_catalog.json` 不建议依赖隐式回退；如果你要用 `--domains`，最好在方言目录里明确提供这个文件

### 3. 哪些配置影响哪条链路

| 配置/文件 | 影响的阶段 | 作用 |
|------|------|------|
| `dialect.yaml` | 全链路 | 路径、句长、普通话 marker、表面词校验、task balance |
| `configs/scene_policy.json` | 生成 | 决定哪些场景默认跑、哪些是 sidecar/candidate |
| `configs/scene_catalog.json` | 生成 | 场景关键词、意图、现代词汇提示 |
| `configs/domain_catalog.json` | 生成 + review | `--domains` 选中的专用词、禁用词、备注 |
| `grammar/grammar_spec.md` | 生成 + repair + review | 给 LLM 和人工审核提供语法章节摘录 |
| `grammar/grammar_rules.yaml` | 校验 + repair + review | 机器可执行语法规则、修补触发、章节映射 |
| `data/cleaned/cleaned_records_primary.jsonl` | 抽种子 + 兜底词表 | 原始清洗词典主输入 |
| `data/extracted_training_sentences/*.jsonl` | 生成 | few-shot 例句来源 |
| `data/controlled_generation/assets/replaceable_lexicon.jsonl` | 生成 | 场景词表、核心词/辅助词候选 |
| `data/controlled_generation/assets/external_term_catalog.json` | 生成 | 现代生活词、地名等外部词补充 |
| `summary.json`（上一轮） | 生成 | `--feedback-run-id/--feedback-summary` 的回流输入 |

### 4. 当前 prompt 的真实状态

- `prompts/` 目录已经有运行时路径入口，但当前 `fewshot_batch` 还不是直接读取 `generation_system.md / generation_user.md / repair_system.md`
- 目前主链路是“代码内拼 prompt + 注入 `grammar_spec.md` 摘录 + 注入 domain/scene/task 上下文”
- 所以如果你现在只改 `prompts/*.md`，**不会**自动改变 `fewshot_batch` 的生成行为

---

## 8 个入口详解

### 入口 1: `dialect.yaml` — 方言身份证

这是整个项目的核心配置文件，包含：

| 字段 | 说明 | 必填 |
|------|------|------|
| `dialect_name` | 方言名称（如"粤语"、"闽南语"） | 是 |
| `dialect_code` | 英文短代码（如 yue、nan） | 是 |
| `sentence_length.min/max` | 生成句子的字数范围 | 是 |
| `task_balance.max_prompt_variants_per_word_combo` | 同一词组允许的 prompt 变体上限，用来减少弱场景配额空转 | 否 |
| `mandarin_markers` | 普通话标记词列表 | 是 |
| `stopwords` | 停用词 | 否 |
| `bad_definition_hints` | 释义中的排除关键词 | 否 |
| `boundary_single_chars` | 表面词校验的边界虚词 | 否 |
| `llm.primary_provider` | 主力 LLM 提供商 | 是 |

完整字段参考：`dialects/_template/dialect.yaml`（每个字段都有注释）。

### 入口 2: `configs/scene_policy.json` — 场景分层

定义你的方言项目要覆盖哪些生活场景，以及它们的优先级：

```json
{
  "mainline_focus_scenes": ["daily_life", "food", "transport"],
  "sidecar_scenes": ["work"],
  "candidate_scenes": ["medical"]
}
```

- **mainline**：默认进入批量生成的场景
- **sidecar**：需要额外 gate 才能生成的场景
- **candidate**：暂不生成，等主线稳定后再开

### 入口 3: `configs/scene_catalog.json` — 场景详情

每个场景的关键词、现代词汇、意图描述。用于将词典条目分配到场景、以及在 prompt 中提供场景上下文。

### 入口 4: `configs/domain_catalog.json` — 领域词表

当用户在运行时指定 `--domains` 时，会从这里读取该轮生成和审查所需的领域专用词、禁用词和补充说明。

适合放：

- 医疗、支付、学校、办公等领域的专用说法
- 必须出现或优先出现的词
- 不该混入的词
- 给生成和审核的备注

运行时如果带 `--feedback-run-id` 或 `--feedback-summary`，上一轮 `summary.json` 里的失败分析会继续影响这一轮的配额和采样。
如果指定的 summary 找不到，pipeline 会直接报错退出，不会静默回退。

### 入口 5: `grammar/grammar_spec.md` — 语法参考文档

放你的方言语法参考。可以是语法书的摘录、论文的整理、或者你自己写的规范。

**作用**：
- 生成和修补时，会自动摘取相关章节给 LLM 参考
- 作为人工审核的对照标准

**没有语法文档怎么办？** 可以先留空，pipeline 会跳过语法校验。先跑通流程，再逐步补充规则。

### 入口 6: `grammar/grammar_rules.yaml` — 机器可执行的语法规则

将语法规则写成正则表达式，让机器自动检测。例如：

```yaml
review_markers: ["咗", "紧", "住"]  # 粤语体标记
high_risk_patterns:
  - name: "mandarin_aspect"
    pattern: "(?:了|着|过)"
    description: "普通话体标记混入"
```

**不确定怎么写？** 每个字段在模板文件中都有注释和温州话的示例。

### 入口 7: `prompts/` — LLM Prompt 模板

三个 prompt 文件：

| 文件 | 用途 | 何时调用 |
|------|------|----------|
| `generation_system.md` | 告诉 LLM "你是谁、规则是什么" | 每次生成 |
| `generation_user.md` | 给 LLM 具体的任务（例句、核心词等） | 每个 task |
| `repair_system.md` | 告诉修补 LLM 如何最小幅度修改 | 触发修补时 |

**当前状态**：
- 这些路径已经进入运行时配置层
- 但 `fewshot_batch` 目前还没有直接从这三个 markdown 文件读 prompt
- 当前主链路仍以内置 prompt 模板为主，再叠加 grammar / domain / scene 上下文

### 入口 8: `data/source_dictionaries/` — 原始数据

把你的方言词典文件（xlsx、csv、jsonl 等）放在这里。

**最低要求**：至少一份包含以下信息的文件：
- 方言词
- 释义
- 方言例句
- 普通话翻译

---

## 原始文件格式速查

这里列的是当前主 pipeline 真正会读到的格式。不是所有字段都必填，但这些最小字段最好保证。

### 1. `data/cleaned/cleaned_records_primary.jsonl`

用途：
- `wz-extract-training-sentences` 的直接输入
- `fewshot_batch` 在缺词表时的兜底来源

最小建议字段：

```json
{
  "wz_word_train": "方言词",
  "definition_norm": "释义",
  "example_wz_train": "方言例句",
  "example_zh_train": "普通话翻译",
  "source_file": "原始文件名",
  "source_row_id": 123
}
```

说明：
- `example_wz_train` 为空的行不会进入种子句子池
- `wz_word_train` 最好是 2-4 字的可复用词面，便于进入生成候选
- `source_file / source_row_id` 不是强制，但强烈建议保留，方便追溯

### 2. `data/extracted_training_sentences/short_8_20.jsonl`

用途：
- `fewshot_batch` 当前主要把它当 few-shot 例句池

由 `wz-extract-training-sentences` 自动产出，典型字段：

```json
{
  "sentence_id": "dict_ex_xxx",
  "wz_sentence": "方言句子",
  "zh_sentence": "普通话翻译",
  "wz_word": "来源词",
  "definition": "来源释义",
  "char_len": 16,
  "length_bucket": "short",
  "source_file": "原始文件名",
  "source_row_id": 123
}
```

### 3. `data/controlled_generation/assets/replaceable_lexicon.jsonl`

用途：
- `fewshot_batch` 的主词表来源，用来给每个场景挑核心词和辅助词

最小建议字段：

```json
{
  "wz_word": "方言词",
  "mandarin_headword": "普通话词头",
  "definition": "释义",
  "primary_scene_id": "scene_id",
  "semantic_class": "noun",
  "slot_kind": "noun"
}
```

说明：
- 实际文件可以带更多打标字段，例如 `scene_tags`、`scene_scores`、`confidence`
- 当前生成主链最关心的是：词面、释义、主场景、语义类别、slot 类别

### 4. `data/controlled_generation/assets/external_term_catalog.json`

用途：
- 给生成链路补现代生活词、设备名、地名等词汇

当前支持的主要键：

```json
{
  "example_mined_terms": [
    {
      "term": "笔记本",
      "scene_tags": ["digital_chat", "work_study"],
      "primary_topic_scene": "digital_chat",
      "semantic_class": "device"
    }
  ],
  "place_names": [
    {
      "term": "七都",
      "scene_tags": ["transport_trip", "daily_chat"],
      "primary_scene_id": "transport_trip"
    }
  ]
}
```

### 5. `configs/scene_policy.json`

最小格式：

```json
{
  "mainline_focus_scenes": ["scene_1", "scene_2"],
  "sidecar_scenes": ["scene_3"],
  "candidate_scenes": ["scene_4"],
  "default_scenes": ["scene_1", "scene_2"],
  "priority_scenes": ["scene_1", "scene_2", "scene_3", "scene_4"]
}
```

联动规则：
- `default_scenes` 是 CLI 不传 `--scenes` 时的默认集合
- `priority_scenes` 决定 task allocator 的全局顺序和可调度场景池
- `sidecar_scenes` 只在额外 gate 满足时开放

### 6. `configs/scene_catalog.json`

最小格式：

```json
[
  {
    "scene_id": "scene_1",
    "label": "场景中文名",
    "intents": ["这个场景下会说什么"],
    "zh_keywords": ["普通话关键词"],
    "dialect_keywords": ["方言关键词"],
    "default_required_modern_words": [],
    "default_preferred_modern_words": []
  }
]
```

### 7. `configs/domain_catalog.json`

最小格式：

```json
[
  {
    "domain_id": "medical",
    "label": "医疗",
    "scene_allowlist": ["health_medical"],
    "required_terms": ["挂号"],
    "preferred_terms": ["门诊"],
    "blocked_terms": ["偏方"],
    "prompt_notes": ["生成时优先用看病场景用语。"],
    "review_notes": ["审查时检查专用词是否自然。"]
  }
]
```

联动规则：
- 只有运行时显式传 `--domains medical` 才会生效
- 生效后会同时进入 prompt、validation、review TSV 和 LLM critic
- 如果传了不存在的 `domain_id`，pipeline 会直接报错

### 8. `grammar/grammar_rules.yaml`

最小格式：

```yaml
review_markers: []
repair_trigger_pattern: ""
completion_marker: ""
allowed_after_completion: ""
max_completion_count: 1
sentence_final_particle: ""
max_sentence_final_count: 1
high_risk_patterns: []
modal_verbs_before_completion:
  pattern: ""
marker_to_sections: {}
reason_to_sections: {}
```

联动规则：
- `high_risk_patterns` 进入机器校验
- `repair_trigger_pattern` 决定是否触发 LLM repair
- `marker_to_sections / reason_to_sections` 决定从 `grammar_spec.md` 摘哪几段给 repair 和 review

### LLM API 配置 — `.env` 文件

```bash
# .env 文件格式（项目根目录）
# 支持 Qwen (通义千问)、DeepSeek、或任何 OpenAI 兼容 API

# Qwen / 通义千问 (通过 DashScope)
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-plus

# DeepSeek
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

---

## Pipeline 步骤详解

### Step 1: 从词典提取种子句子

```bash
wz-extract-training-sentences
```

**做了什么**：
- 读取 `data/cleaned/cleaned_records_primary.jsonl`
- 清洗括号注释、规范空白
- 按长度筛选（默认 8-30 字）
- 去重
- 输出到 `data/extracted_training_sentences/`

**输出**：
- `short_8_20.jsonl` — 短句（用作 few-shot 例句）
- `long_20_30.jsonl` — 长句
- `all_in_range.jsonl` — 全部

### Step 2: 批量生成新句子

```bash
wz-generate-fewshot-batch --tasks 200 --provider deepseek
```

这一阶段固定会写出：
- `config.json`：包含 `domains` 和 `feedback`
- `summary.json`：包含 `machine_metrics.failure_analysis`
- `review.tsv`：包含 `failure_buckets`、`failure_primary_bucket`、`review_focus`

**做了什么**：
- 按场景均衡分配生成任务
- 为每个任务选择 anchor 例句 + 核心词 + 辅助词
- 调用 LLM 生成 5 句
- 语法校验（grammar guardrails）
- 表面词校验（source surface guardrails）
- 如果检测到高风险功能词，尝试 LLM 修补

**关键参数**：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--tasks N` | 生成任务数（每个任务产 5 句） | 200 |
| `--provider` | LLM 提供商 | deepseek |
| `--scenes` | 指定场景（逗号分隔） | 全部 mainline |
| `--domains` | 指定领域（逗号分隔），读取 `domain_catalog.json` | 空 |
| `--feedback-run-id` | 用上一轮 `summary.json` 做配额/采样回流 | 空 |
| `--feedback-summary` | 直接指定上一轮 `summary.json` 路径 | 空 |
| `--food-modern-trial-ratio` | `food_dining` 中现代词 trial lane 占比 | 0.25 |
| `--disable-grammar-repair` | 跳过高风险功能词的二次 repair | 否 |
| `--resume` | 从断点恢复 | 否 |
| `--run-id` | 指定 run ID | 自动生成 |

**输出**（在 `runs/fewshot_batch/<run_id>/` 下）：
- `results.jsonl` — 全部生成结果（含通过和失败）
- `rule_gate.jsonl` — 通过规则校验的句子
- `review.tsv` — 人工审核用的 TSV
- `summary.json` — 运行统计

### Step 3: 导出人工审核包

```bash
wz-export-review-package --run-id <run_id>
```

生成标准 TSV 文件，包含：
- 方言句子、普通话翻译
- 场景、核心词、辅助词
- 规则校验状态、语法标记
- 空白的"专家判定"和"备注"列 — 由审核人填写

### Step 4: 导入审核结果

审核人填完 TSV 后：

```bash
wz-import-review-results \
  --input <填好的.tsv> \
  --output <输出.jsonl> \
  --origin-run-id <run_id> \
  --pipeline-name fewshot_batch \
  --policy-version <版本号>
```

### Step 5: 晋级为训练数据

```bash
wz-promote-reviewed-samples \
  --input <审核结果.jsonl> \
  --promotion-candidates-output <候选.jsonl> \
  --curated-output curated/generated_reviewed.jsonl \
  --policy-version <版本号> \
  --origin-run-id <run_id>
```

**晋级条件（硬规则）**：`rule_gate_status == pass` 且 `human_review_status == pass`

### Step 6: 导出训练数据

```bash
wz-build-training-candidates
```

合并 `curated/seed_sentences.jsonl` + `curated/generated_reviewed.jsonl`，输出最终的 `curated/training_candidates.jsonl`。

---

## 数据治理

### Trust Tier 机制

每条数据都有一个信任层级：

| 来源 | Trust Tier | 能否用于训练 | 能否回流生成 |
|------|-----------|-------------|-------------|
| 词典原文提取 | `source` | 能 | 能 |
| LLM 生成（未审核） | `candidate` | **不能** | **不能** |
| 人工审核通过 | `curated` | 能 | 能 |

**核心规则**：`generated_candidate` 永远不能回流上游。只有 `source` 和 `curated` 数据可以进入下一轮生成的 prompt、anchor、seed 或训练导出。

### 每次运行的产物

每次 pipeline 运行都会在 `runs/<pipeline_name>/<run_id>/` 下产生一套标准文件：

```
config.json                — 运行配置（可复现）
results.jsonl              — 全部结果
rule_gate.jsonl            — 规则校验通过的
review.tsv                 — 人工审核用
summary.json               — 机器指标 + 人工指标
promotion_candidates.jsonl — 晋级候选
```

所有运行都注册到 `registry/run_registry.jsonl`，支持回溯。

---

## 架构概览

### 框架代码 vs 方言数据

```
项目根目录/
├── src/wz_pipeline/          ← 框架代码（方言无关的 pipeline 逻辑）
│   ├── contracts.py             数据合约 & trust tier
│   ├── runs.py                  运行管理
│   ├── registry.py              运行注册
│   ├── review.py                审核工作流
│   ├── promotion.py             晋级逻辑
│   ├── grammar_guardrails.py    语法校验
│   ├── grammar_spec.py          语法文档解析
│   ├── source_surface_guardrails.py  词面校验
│   ├── pipelines/               Pipeline 实现
│   └── cli/                     命令行入口
│
├── dialects/                 ← 方言数据（每个方言一个目录）
│   ├── _template/               空模板（复制这个开始）
│   └── wenzhou/                 温州话（已验证的实例）
│
├── configs/                  ← 当前温州话的配置（后续迁入 dialects/wenzhou/）
├── data/                     ← 当前温州话的数据（后续迁入 dialects/wenzhou/）
├── runs/                     ← 当前运行产物
├── curated/                  ← 当前审核通过的数据
├── registry/                 ← 当前运行记录
├── scripts/                  ← 遗留脚本（逐步迁入框架包）
├── tests/                    ← 测试
└── docs/                     ← 文档
```

> **当前状态**：框架代码已开始从 `dialects/<dialect>/dialect.yaml` 读取运行时配置，并支持本地方言 overlay 不存在时回退到根目录现有资源。
> 仍有一部分温州话启发式常量尚未完全移出代码（详见 `docs/AUDIT_REPORT.md`），但生成主链、语法摘录和 review metadata 已接入配置层。

---

## 当前已验证方言：温州话

| 指标 | 数值 |
|------|------|
| 种子句子 | 3,674 |
| 生成试验（最大批次） | 115 句，93% 规则通过 |
| 人工审核通过 | 0（尚未完成首轮人工审核） |
| 语法规则覆盖 | 约 5%（仅 7 个字面 pattern） |

已知问题和改进建议详见 [`docs/AUDIT_REPORT.md`](docs/AUDIT_REPORT.md)。

---

## 开发者指南

### 运行测试

```bash
pip install -e ".[dev]"
pytest tests/
```

### 当前 CLI 入口

| 命令 | 功能 |
|------|------|
| `wz-extract-training-sentences` | 从词典提取种子句子 |
| `wz-generate-fewshot-batch` | 批量生成新句子 |
| `wz-build-modern-anchor-candidates` | 识别现代词锚点 |
| `wz-export-review-package` | 导出人工审核包 |
| `wz-import-review-results` | 导入审核结果 |
| `wz-promote-reviewed-samples` | 晋级审核通过的样本 |
| `wz-register-legacy-run` | 注册历史运行 |
| `wz-materialize-seed-curated` | 将种子句子写入 curated |
| `wz-build-training-candidates` | 构建最终训练数据 |

### 重构路线图

1. **P0**：泛化语法检测 + 完成首轮人工审核
2. **P1**：加独立 LLM 审核 pass + 失败分析回流
3. **P2**：拆分 `fewshot_batch.py` + 配置外部化（读取 `dialect.yaml`）
4. **P3**：端到端测试 + prompt fuzz 测试

详见 [`docs/AUDIT_REPORT.md`](docs/AUDIT_REPORT.md)。

---

## FAQ

**Q: 我没有方言语法参考文档怎么办？**  
A: 先把 `grammar/grammar_spec.md` 和 `grammar_rules.yaml` 留空。Pipeline 会跳过语法校验，直接进入人工审核。你可以从人工审核中积累常见错误，逐步补充规则。

**Q: 我的方言没有体貌标记（如"了/着/过"的对应物）怎么办？**  
A: `grammar_rules.yaml` 中的 `review_markers`、`completion_marker` 等字段全部留空。语法校验只检查你定义的规则——如果没定义，就不检查。

**Q: 怎么换成别的 LLM？**  
A: 只要兼容 OpenAI API 格式就行。在 `.env` 中配置 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY`，指向你的 API。已测试的提供商：通义千问（Qwen）、DeepSeek。

**Q: 需要多少种子句子才够？**  
A: 实测温州话 3,674 句种子可以支撑有效的 few-shot 生成。建议至少 500 句。句子越多、覆盖的词汇和句式越广，生成质量越好。

**Q: 生成的质量怎么样？**  
A: 高度依赖你的方言数据质量和语法规则覆盖度。温州话实测机器规则通过率 ~93%，但真实的人工通过率还需要验证。建议先跑小批量（20-50 task）+ 完整人工审核，确认质量后再扩量。

**Q: 能用于商业项目吗？**  
A: 工具箱本身开源。但你使用的 LLM API 可能有自己的使用条款，请查阅你所使用的 API 提供商的政策。
