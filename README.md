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
├── configs/                  ← 【入口2-3】场景和策略配置
│   ├── scene_policy.json        场景分层（哪些先做、哪些后做）
│   ├── scene_catalog.json       场景详情（关键词、现代词汇）
│   ├── trust_policy.json        数据信任层级（通常不改）
│   └── pipeline_thresholds.json 质量门槛
│
├── grammar/                  ← 【入口4-5】语法规则
│   ├── grammar_spec.md          你的语法参考文档
│   └── grammar_rules.yaml       机器可执行的语法校验规则
│
├── prompts/                  ← 【入口6】LLM Prompt
│   ├── generation_system.md     生成 prompt 系统消息
│   ├── generation_user.md       生成 prompt 用户消息模板
│   └── repair_system.md         语法修补 prompt
│
├── data/                     ← 【入口7】你的数据
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

## 7 个入口详解

### 入口 1: `dialect.yaml` — 方言身份证

这是整个项目的核心配置文件，包含：

| 字段 | 说明 | 必填 |
|------|------|------|
| `dialect_name` | 方言名称（如"粤语"、"闽南语"） | 是 |
| `dialect_code` | 英文短代码（如 yue、nan） | 是 |
| `sentence_length.min/max` | 生成句子的字数范围 | 是 |
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

### 入口 4: `grammar/grammar_spec.md` — 语法参考文档

放你的方言语法参考。可以是语法书的摘录、论文的整理、或者你自己写的规范。

**作用**：
- 生成和修补时，会自动摘取相关章节给 LLM 参考
- 作为人工审核的对照标准

**没有语法文档怎么办？** 可以先留空，pipeline 会跳过语法校验。先跑通流程，再逐步补充规则。

### 入口 5: `grammar/grammar_rules.yaml` — 机器可执行的语法规则

将语法规则写成正则表达式，让机器自动检测。例如：

```yaml
review_markers: ["咗", "紧", "住"]  # 粤语体标记
high_risk_patterns:
  - name: "mandarin_aspect"
    pattern: "(?:了|着|过)"
    description: "普通话体标记混入"
```

**不确定怎么写？** 每个字段在模板文件中都有注释和温州话的示例。

### 入口 6: `prompts/` — LLM Prompt 模板

三个 prompt 文件：

| 文件 | 用途 | 何时调用 |
|------|------|----------|
| `generation_system.md` | 告诉 LLM "你是谁、规则是什么" | 每次生成 |
| `generation_user.md` | 给 LLM 具体的任务（例句、核心词等） | 每个 task |
| `repair_system.md` | 告诉修补 LLM 如何最小幅度修改 | 触发修补时 |

### 入口 7: `data/source_dictionaries/` — 原始数据

把你的方言词典文件（xlsx、csv、jsonl 等）放在这里。

**最低要求**：至少一份包含以下信息的文件：
- 方言词
- 释义
- 方言例句
- 普通话翻译

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

> **当前状态**：框架代码中仍有温州话特定的硬编码值（详见 `docs/AUDIT_REPORT.md`）。
> `dialects/wenzhou/dialect.yaml` 已提取了所有硬编码值作为参考。
> 后续代码重构将让框架代码从 `dialect.yaml` 读取配置，实现真正的方言无关。

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
