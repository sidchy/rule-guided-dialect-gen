# 温州话语料 Pipeline 审计报告

> 审计日期：2026-04-01
> 审计范围：整个 `微调用温州话语料` 项目的 pipeline 设计、代码实现、规则覆盖、闭环可靠性

---

## 总结

| 维度 | 结论 |
|------|------|
| 路线是否成立 | **成立**。用词典 seed 喂 LLM 做受控生成 + 规则约束 + 人工校验，是低资源方言语料扩充的合理路线 |
| 最大风险 | 语法校验覆盖率严重不足（代码只实现了文档规则的 ~5%），导致 93% pass rate 虚高 |
| 自审核闭环 | **不可靠**。生成和修补用同一模型，无独立评判；rule gate 只有 7 个字面 pattern |
| 最缺的质控环节 | (1) 温/普语义对齐检测 (2) 泛化的语法校验 (3) 失败原因回流机制 |
| 代码支撑后续扩量 | 骨架良好（trust tier、run registry、review workflow），但方言特定内容与框架代码耦合，且主文件 1824 行难以维护 |

---

## A. 当前 Pipeline 总结

```
词典 XLSX
    │
    ▼
extract_training_sentences ──► seed pool (3674句, 8-30字)
    │                              │
    │          ┌───────────────────┤
    │          ▼                   ▼
    │  modern_anchor_candidates   fewshot_batch (DeepSeek/Qwen)
    │  (识别现代词锚点)            │ ← SYSTEM_PROMPT + 语法规范摘录
    │                             │ ← grammar_guardrails 校验
    │                             │ ← grammar_repair 二次修补
    │                             │ ← source_surface_guardrails 词面校验
    │                             ▼
    │                       rule_gate.jsonl (机器通过)
    │                             │
    │                       export_review_tsv → 人工审核 → import_review
    │                             │
    │                       promote → curated/generated_reviewed.jsonl
    │                             │
    └─────────────────────► build_training_candidates → 合并输出
```

**当前数据量：**

| 数据集 | 数量 | 状态 |
|--------|------|------|
| 种子句子 (seed) | 3,674 | 可用于训练 |
| 生成句子 (pilot demo) | 10 句，5 过 (50%) | 未人工审核 |
| 生成句子 (legacy v4) | 115 句，107 过 (93%) | 未人工审核 |
| 人工审核通过 | **0** | curated 生成池为空 |

**现状**：Pipeline 搭好了，但还没跑完一个完整闭环——零句生成句子通过了人工审核。

---

## B. 五大核心问题

### B1. 语法规则覆盖是"样板匹配"，不是真正的语法校验

**严重性：高** | **位置：** `src/wz_pipeline/grammar_guardrails.py:31-42`

`grammar_guardrails.py` 只有 7 个硬编码字面 pattern：

```python
HIGH_RISK_LITERAL_PATTERNS = [
    ("completion_object_order", re.compile(r"跳爻舞")),       # 只能捉到"跳爻舞"
    ("stative_progressive", re.compile(r"著埭(?:坐|晓得|快活)")), # 只有3个动词
    ("dynamic_postposed_zhedai", re.compile(r"(?:走|笑|飞)著埭")),
    ("qishi_object_order", re.compile(r"唱歌起罢")),
    ("continuative_object_order", re.compile(r"一直开会落去")),
    ("mandarin_negation", re.compile(r"(?:没有|沒有)")),
    ("northern_wu_lexeme_gaotou", re.compile(r"高头")),
]
```

这些是语法文档里举的**原始例子**，不是泛化规则。同类型错误完全通不过检测：

| 能检测到 | 检测不到（同类型） |
|----------|-------------------|
| `跳爻舞` | `吃爻饭`、`唱爻歌`、`写爻字` |
| `著埭坐/晓得/快活` | `著埭想`、`著埭知道` |
| `唱歌起罢` | `跳舞起罢`、`读书起罢` |

语法文档 §12 列了"宾语前移"为硬规则（V+O+体标记 → O+V+体标记），但代码只匹配了文档举的那一个原句。

**影响：93% pass rate 不可信——大量同类语法错误被放行。**

---

### B2. "自审核"用同一模型做生成和修补，无独立评判

**严重性：高** | **位置：** `src/wz_pipeline/pipelines/fewshot_batch.py:1283-1335`

生成和 grammar repair 都用 DeepSeek-chat：

- 生成：`temperature=0.6`，`fewshot_batch.py:1293`
- 修补：`temperature=0.2`，`fewshot_batch.py:1314`
- 修补后重新进 validate_sentence，但 validate_sentence 的 grammar check 就是那 7 个字面 pattern

**问题链**：
1. 模型产出的温州话本身可能有语法错误
2. 同一模型自纠无法收敛到正确（缺乏外部参照）
3. Rule gate 太弱，无法拦截修补后仍有问题的句子
4. 没有独立第三方审核 pass

**这不是"自审核闭环"，而是"同一个学生做题、同一个学生改卷"。**

---

### B3. 无温州话/普通话语义对齐检测

**严重性：高** | **位置：** `src/wz_pipeline/pipelines/fewshot_batch.py:1349-1425`

`validate_sentence()` 检查的内容：

- ✅ 字数（20-30）
- ✅ 核心词是否存在
- ✅ 禁止词
- ✅ 表面词存在性
- ✅ 少量普通话标记词
- ❌ **不检查** `wz_sentence` 和 `zh_sentence` 语义是否一致
- ❌ **不检查** 温州话句子是否有意义
- ❌ **不检查** 温州话句子内部语义连贯性

实际样例中已有可疑对齐：

> 温州话：`大蛮阵个家庭，洗衣机用爻罢，只好手洗能界。`
> 普通话：`人口多的家庭，洗衣机坏了，只能手洗的时候。`

"用爻罢"翻译成"坏了"是否正确？代码完全无法判断。

**影响：人工校验量不可能"少量"——必须逐句对照温/普对。**

---

### B4. 规则文档 510 行，代码只实现 ~5%

**严重性：高** | **位置：** `docs/温州话生成语法规范_潘悟云原书67-83页.md` vs `grammar_guardrails.py` + `grammar_spec.py`

语法文档覆盖了：

| 文档章节 | 内容 | 代码实现 |
|----------|------|----------|
| §3.1 完成体 `爻` | 宾语前移、不能跟情态动词、exhaustive 语义 | 部分（3个pattern + 情态动词检测） |
| §3.2 进行体 `著埭+V` | 不能配静态动词 | 3个静态动词字面匹配 |
| §3.3 持续体 `V+著埭` | 不能配动态动词 | 3个动态动词字面匹配 |
| §3.4 已然体 `罢` | 重读/轻读区分 | 只检查数量≤1 |
| §3.5 起始体 `起` | 宾语前移 | 1个字面匹配 |
| §3.6 继续体 `落去` | 宾语前移 | 1个字面匹配 |
| §4 `有`字句 | 存在/事件肯定 | **零实现** |
| §5 处置句 `逮` | 宾语前置 | **零实现** |
| §6 被动/致使 `咐/财` | 句式结构 | **零实现** |
| §7 双宾语 | 定量宾语要求 | **零实现** |
| §8 补语 | 结果/程度补语 | **零实现** |
| §9 否定 `不/未/冇` | 三分系统不可混用 | 只查字面 `没有` |
| §10 反问句 `啊不/啊未/啊冇` | 疑问类型 | **零实现** |
| §11 句末语气词 | 功能性使用 | **零实现** |

`grammar_spec.py` 的 `relevant_spec_excerpt()` 只是把规范文本塞进 prompt 让 LLM "参照"——这不是可执行的校验。

---

### B5. Pilot 50% vs Legacy 93%——生成稳定性存疑

**严重性：中** | **位置：** `runs/fewshot_batch/fewshot_mainline_pilot_demo_20260401/summary.json`

| 运行 | 句数 | 通过率 | 失败原因 |
|------|------|--------|----------|
| pilot demo | 10 | 50% | 全部 `too_short` (18-19字) |
| legacy v4 | 115 | 93% | 8句失败 |

同一 pipeline、同一模型（deepseek-chat），差距巨大。原因：

1. pilot 只有 2 个 task（anchor 选择空间小）
2. 模型对 "20-30 字" 硬约束的执行力随 anchor/例句质量波动
3. 没有 retry 机制让短句重新生成

**启示：pass rate 高度依赖 anchor 和例句选择，而不是一个稳定的产出指标。**

---

## C. 方案层面问题

### C1. "抽骨架 + 换实词"已偏离为自由生成

`prompts/controlled_generation_rewrite_v1.md` 定义了 `[SLOT_x]` 骨架替换方案，但 `fewshot_batch.py` 实际采用的是完全自由生成：给 LLM 几个例句 + 核心词，让它"生成 5 句 20-30 字的温州话口语长句"。

两套 prompt 并存但互不相干。骨架替换方案的可控性更好，但被搁置了。

### C2. 场景分层过于复杂但缺数据支撑

9 个场景（5 mainline + 2 sidecar + 2 candidate），每个场景有独立的 priority terms、deny terms、support pool sizes、modern gate。但 seed pool 的场景分配靠 keyword matching（`scene_match_score`），precision 未经验证。

如果场景标注错了，整个分层就是在错误标签上做精细化。

### C3. 无失败原因反馈回路

当前设计中没有任何机制让人工审核结果反馈到生成策略：

- 通过的句子进 curated pool
- 失败的句子的失败原因没有被结构化收集
- 没有用于调整 prompt 或规则

"闭环"只闭了数据流，没闭信号流。

---

## D. 工程/代码问题

### D1. fewshot_batch.py 1824 行单文件

**位置：** `src/wz_pipeline/pipelines/fewshot_batch.py`

一个文件包含：task building、data loading、validation、generation、checkpoint、CLI、场景词表、定义过滤器。无法独立测试、无法独立复用。

### D2. 对 legacy 脚本有硬依赖

```python
# fewshot_batch.py:31-37
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(LEGACY_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_SCRIPTS_DIR))

from build_controlled_generation_assets import ROOT, SCENE_CATALOG, SEMANTIC_CLASS_HINTS, build_scene_payload
from generate_controlled_sentences import build_client
```

核心 pipeline 通过 `sys.path.insert` 引入 `scripts/` 目录下没有 `__init__.py` 的遗留脚本。不可测试、不可打包。

### D3. 方言特定常量硬编码在 Python 源码

以下内容全部硬编码在 `fewshot_batch.py` 中：

- `FOOD_SUPPORT_PRIORITY_TERMS`（12个词）
- `HOME_SUPPORT_PRIORITY_TERMS`（13个词）
- `SHOPPING_SUPPORT_PRIORITY_TERMS`（16个词）
- `TRANSPORT_SUPPORT_PRIORITY_TERMS`（17个词）
- `WEATHER_SUPPORT_PRIORITY_TERMS`（15个词）
- `SHOPPING_OLD_MONEY_TERMS`、`SHOPPING_AMOUNT_TERMS`
- `FOOD_BEVERAGE_TERMS`、`FOOD_STAPLE_TERMS`、`FOOD_ACTION_TERMS`
- `MANDARIN_MARKERS`（14个普通话标记词）
- `BAD_DEFINITION_HINTS`、`ARCHAIC_DEFINITION_HINTS`、`STOPWORDS`
- `SYSTEM_PROMPT`（完整的中文 prompt，20行）
- `POLICY_VERSION`

修改一个场景的词表需要改 Python 源码。换一个方言需要重写整个文件。

### D4. 测试覆盖不足

`tests/` 有 8 个文件，但：

- `test_grammar_guardrails.py` 只测那 7 个字面 pattern（测了就等于没测——测的是"能匹配自己写的例子"）
- 没有端到端测试
- 没有对 prompt 输出格式的 fuzz 测试
- 没有对 `validate_sentence` 的边界条件测试

### D5. 重复的清洗逻辑

- `fewshot_batch.py:205` 的 `clean_wz()`：`PAREN_RE.sub("", text)` + `re.sub(r"[\s\u3000]+", "", text)`
- `source_surface_guardrails.py:33` 的 `clean_surface()`：`PAREN_RE.sub("", text)` + `re.sub(r"[\s\u3000]+", "", text)`

几乎相同的逻辑写了两遍。

---

## E. 修改建议

### P0（必须先做）

| 建议 | 原因 |
|------|------|
| 泛化 grammar_guardrails 的宾语前移检测 | 这是当前最大漏检项。维护一个 VO 组合表，检测 `V+爻+O`（应为 `O+V+爻`），同理覆盖 `起`、`落去` | 
| 跑一批 200 句 + 完整人工审核 | 没有真实 human pass rate，所有后续决策都是盲的 |

### P1（尽快做）

| 建议 | 原因 |
|------|------|
| 加独立 LLM 审核 pass | 在 rule_gate 之后、人工之前，用不同模型/不同 prompt 审核：温/普对齐 + 句子通顺 + 语法标记。能把人工量降一个数量级 |
| 建 fail_analysis 统计和回流机制 | 每批 run 的失败原因按类分类统计，用于调整下一批 prompt 和规则 |

### P2（系统性改善）

| 建议 | 原因 |
|------|------|
| 拆分 fewshot_batch.py | 至少拆成 task_builder / generation / validation / data_loader / cli_main |
| 方言常量外部化到配置文件 | 场景词表、停用词、普通话标记 → JSON/YAML；换方言不需要改代码 |
| 消除 sys.path hack | 将 `build_client` 和 `build_scene_payload` 移入 pipeline 包 |

### P3（长期质量）

| 建议 | 原因 |
|------|------|
| 端到端测试 | 给定 input → 验证完整 pipeline 输出格式和基本约束 |
| prompt 输出 fuzz 测试 | LLM 返回非标准 JSON 时系统不应崩溃 |
| `著埭` 动/静限制的动词分类表 | 不只是 3 个动词，应维护动态/静态动词列表 |

---

## F. 开源泛化建议

当前项目的框架层逻辑（trust tier、run registry、review workflow、contracts）已经是方言无关的。主要耦合点在：

1. **语法规则** → 提取到 `grammar_rules.yaml`（每个方言自己写）
2. **场景词表** → 提取到 `scene_catalog.json`（每个方言自己配）
3. **LLM Prompt** → 提取到 `prompts/` 目录下的模板文件
4. **清洗参数** → 提取到 `dialect.yaml` 主配置

目标架构：**一个方言 = 一个数据目录**。框架代码（`src/dialect_pipeline/`）不含任何方言特定内容。

详细设计见 `README.md` 中的框架结构说明。
