# 温州话微调数据集与训练方案

## 0. 2026-03-30 方案转向

从 2026-03-30 起，本项目的当前优先目标已经从“完整的普通话 -> 温州话翻译微调”转向为：

- 基于现有句子骨架，保持句式、语气、结构大致不变
- 通过替换实词，批量生成新的温州话句子
- 在生成时支持加入更现代生活、更场景化的需求与词汇要求
- 产出适用于 AI 日常对话和后续语音训练的大批量句子

这意味着当前最重要的能力不再是：

- 通用普通话 -> 温州话翻译

而是：

1. `wz_rewrite_keep_skeleton`
   保持骨架，只替换实词
2. `scene_conditioned_wz_generation`
   给定场景、词汇约束和句式骨架，生成新句子
3. `batch_sentence_expansion`
   基于一条骨架批量扩写出一组可用于语音训练的新句子

### 0.1 当前判断

这个方向比“直接靠 LoRA 教会模型完整普通话-温州话翻译”更可行，原因是：

1. 任务更窄，约束更强
2. 现有语料更适合抽“骨架”和“可替换词位”
3. 最终用途是生成可控语音训练句，而不是做开放域翻译

### 0.2 旧方案如何处理

此前已经生成的 `stage1` 翻译/解释训练集不作废，但地位调整如下：

1. `zh_to_wz_sentence` 不再是核心目标，而是辅助材料
2. `zh_to_wz_word` 和 `wz_explain` 主要用于补充词汇映射与词义约束
3. 后续真正的主训练集应转向“骨架改写 + 场景词约束生成”

### 0.3 当前推荐的训练目标

如果继续微调，建议模型重点学习以下两类指令：

1. 给定一条温州话句子：
   保留句式骨架和语气，只替换实词，生成若干条新句子
2. 给定：
   - 一个骨架句
   - 一个生活场景
   - 一组要求出现的现代词汇
   生成符合骨架的新温州话句子

### 0.4 当前推荐的系统结构

当前更适合采用：

- 受控生成微调
- 词汇约束检索
- 批量生成与筛选

而不是把主要精力放在“通用翻译能力”上。

### 0.5 当前执行策略更新

从当前阶段开始，默认采用：

- 非微调优先

也就是说，第一步不再默认先做 LoRA 或 QLoRA，而是优先搭建一套可直接产句子的受控生成流水线：

1. 骨架库
2. 槽位体系
3. 实词表
4. 场景表
5. 批量生成 prompt
6. 自动筛选、去重与打分

### 0.6 为什么先不微调

当前判断是：

1. 任务已经从“开放翻译”收窄为“受控改写与批量造句”
2. 骨架、场景、词汇要求都可以在推理时显式提供
3. 对这个目标，prompt 设计和数据流水线的重要性高于先做微调
4. 如果基础模型已经能稳定守骨架并完成实词替换，就没有必要先投入微调成本

### 0.7 微调何时再考虑

只有在以下情况下，再把微调重新提到优先级前面：

1. 模型经常改坏骨架
2. 模型生成结果太像普通话，不像温州话
3. 场景词和现代词汇融入不自然
4. 批量生成合格率太低，筛选成本过高

在那之前，默认先验证“基础模型 + 受控生成流水线”是否已经足够。

### 0.8 当前主线落地物

围绕“受控生成”主线，当前代码与数据目录应新增并优先维护以下产物：

1. `data/controlled_generation/assets/replaceable_lexicon.jsonl`
   可替换实词表，提供槽位候选与替换池
2. `data/controlled_generation/assets/scene_catalog.json`
   场景表，提供场景标签、关键词和现代词种子
3. `data/controlled_generation/skeletons/skeleton_templates.jsonl`
   骨架模板库，保存原句、骨架模板、槽位信息和质量字段
4. `data/controlled_generation/tasks/generation_tasks_*.jsonl`
   生成任务输入，供 API 或后续模型批量产句
5. `data/controlled_generation/candidates/*.jsonl`
   原始生成候选
6. `data/controlled_generation/filtered/*.jsonl`
   自动筛选后的通过样本与复核样本

### 0.9 当前交接文档

当前受控生成主线的阶段总结、问题边界、缺失项和接手建议，统一维护在：

- `HANDOFF_CONTROLLED_GENERATION_2026-03-30.md`

当前仓库已经移除了旧 `stage1` 翻译 / 微调主线的核心文件，默认只保留受控生成主线相关脚本和资产。

如果后续由新的 agent 继续推进，优先阅读该文档，再看：

- `data/controlled_generation/reports/task_summary.json`
- `data/controlled_generation/filtered/pilot_v5_qwen/filter_summary.json`
- `data/controlled_generation/filtered/pilot_v5_qwen/critic_summary.json`

## 1. 项目目标

本项目的目标不是训练一个“通用温州话大模型”，而是基于现有中文基座模型，构建一套面向温州话生成与改写的高质量 SFT 数据集，并据此微调一个实用模型。

第一版模型需要重点覆盖以下能力：

1. 普通话 -> 温州话句子翻译
2. 普通话词 -> 温州话词翻译
3. 温州话句子改写：保持骨架、语气、句式尽量不变，只替换实词
4. 温州话词义解释

其中主任务是：

1. `zh_to_wz_sentence`
2. `wz_rewrite_keep_skeleton`

辅助任务是：

1. `zh_to_wz_word`
2. `wz_explain`

## 2. 当前语料概况

当前目录包含 4 份 Excel 语料：

1. `【3.25（2）】林老师.xlsx`
   主要是温州话-普通话句子平行语料
2. `【20260327最终版】温州方言词典.xlsx`
   主要是词条、释义、方言例句、普通话翻译
3. `【终版】活色生香温州话.xlsx`
   主要是词语、俗语、上下句等固定表达
4. `温州话资源库-词汇-323交付版.xlsx`
   主要是词汇及简短释义

这些语料是异质数据，不能直接拼接训练，必须先经过统一结构化、规则清洗、去重、LLM 辅助抽取、任务化构造，再生成最终训练集。

## 3. 模型与训练路线

### 3.1 推荐模型

第一版推荐使用：

- `Qwen/Qwen3.5-9B-Base`

理由：

1. `Base` 更适合作为定制 SFT 的底座
2. 本项目任务集中在方言生成与受控改写，不需要多模态能力
3. 9B 量级足以覆盖当前数据规模，不必一开始上更大模型

### 3.2 推荐训练方式

第一版推荐：

- `QLoRA SFT`
- `QLoRA SFT + 轻量 RAG`

不推荐：

- 全量微调

原因：

1. 单张 RTX 5090 32GB 更适合跑 `QLoRA`
2. 当前数据量级更适合 PEFT，而不是 full fine-tune
3. 先跑一个稳定 baseline，再决定是否升级训练策略
4. 方言生成与改写能力主要通过微调学习，长尾词义和词典知识更适合由检索补充

### 3.3 显卡与资源假设

默认训练环境：

- 单张 `RTX 5090 32GB`
- `bf16` 计算
- 4-bit quantized base model

### 3.4 训练框架

优先采用：

- `ms-swift`
- `LLaMA-Factory`

备选：

- `Axolotl`
- `Unsloth`

当前同时提供 `ms-swift` 与 `LLaMA-Factory` 两套入口；如果只选一种，优先使用 `LLaMA-Factory`。

### 3.5 架构结论

第一版整体架构固定为：

- 主体：`Qwen/Qwen3.5-9B-Base + QLoRA`
- 辅助：词典型轻量 `RAG`

两者职责分工如下：

1. 微调负责：
   - 普通话 -> 温州话生成
   - 普通话词 -> 温州话词映射
   - 温州话骨架改写
   - 温州话表达习惯和输出倾向
2. RAG 负责：
   - 长尾词条
   - 多义词消歧
   - 固定表达和俗语释义
   - 可追溯、可更新的词典知识

第一版默认不做“纯 RAG 替代微调”，也不做全量微调。

## 4. 数据集设计

### 4.1 最终任务类型

最终训练集按以下任务组织：

1. `zh_to_wz_sentence`
   普通话句子翻译成温州话句子
2. `zh_to_wz_word`
   普通话词或短语翻译成温州话词或短语
3. `wz_rewrite_keep_skeleton`
   保持骨架、语气、句式不变，只替换实词
4. `wz_explain`
   解释温州话词条或固定表达的意思

### 4.2 推荐任务配比

第一版训练集建议按以下比例混合：

1. `zh_to_wz_sentence`: `45%`
2. `wz_rewrite_keep_skeleton`: `25%`
3. `zh_to_wz_word`: `20%`
4. `wz_explain`: `10%`

### 4.3 训练阶段建议

建议分两阶段训练：

第一阶段：

1. `zh_to_wz_sentence`
2. `zh_to_wz_word`
3. `wz_explain`

第二阶段：

1. 在第一阶段 adapter 基础上继续训练
2. 加入 `wz_rewrite_keep_skeleton`

这样更容易先学稳基础映射，再学习受控改写。

## 5. 数据清洗总流程

整体流程采用：

- 规则清洗为主
- LLM 仅处理难点抽取、判断、复核与受控生成

说明：

1. 如果数据明显不干净、规则无法稳定裁决，可以调用本地 `.env` 配置的模型 API 做辅助清洗与复核
2. API 主要用于低置信度、冲突、格式异常、释义抽取失败等疑难样本，不作为全量主清洗手段

总流程分为 7 个阶段：

1. 原始入库
2. 规则标准化
3. 去重与冲突归并
4. LLM 结构化抽取
5. 按任务筛样本
6. 构造骨架改写样本
7. 组装最终 SFT 数据集

## 6. 阶段 1：原始入库

将 4 个 Excel 统一导出为结构化中间表，不在此阶段做语义改写。

每条记录至少保留以下字段：

- `source_file`
- `source_sheet`
- `source_row_id`
- `record_type`
- `wz_text_raw`
- `zh_text_raw`
- `definition_raw`
- `example_wz_raw`
- `example_zh_raw`
- `notes_raw`

`record_type` 初步分类为：

- `sentence_pair`
- `dictionary_entry`
- `lexicon_entry`
- `fixed_expression`

## 7. 阶段 2：规则标准化

这一阶段全部使用程序处理，不调用大模型。

### 7.1 统一规则

包括：

1. 删除空列、伪列、`Unnamed:*`
2. 统一空白字符与换行
3. 统一全角半角标点
4. 清理首尾空格
5. 去掉 Excel 中的空字符串与 `NaN`
6. 保留原文备份，不覆盖原始字段

### 7.2 去括号注音

训练主文本默认去掉括号注音，例如：

- `匄（hà）` -> `匄`
- `狃（niǎo）` -> `狃`

但必须保留一份原文备份字段，后续如需追溯仍可使用。

建议保留：

- `text_raw`
- `text_norm`

### 7.3 规范化后的核心字段

对所有可能参与训练的文本保留规范化版本：

- `wz_text_norm`
- `zh_text_norm`
- `definition_norm`
- `example_wz_norm`
- `example_zh_norm`

## 8. 阶段 3：去重与冲突归并

这一阶段也优先使用规则，不依赖大模型。

### 8.1 去重类型

1. 完全重复
2. 近重复
3. 冲突重复

### 8.2 完全重复

直接删除：

- 文本完全一致
- 或经规范化后完全一致

### 8.3 近重复

仅差异包括：

1. 标点
2. 空格
3. 括号注音
4. 少量格式符号

这类归并为同一条。

### 8.4 冲突重复

比如：

1. 同一温州话词条对应多个普通话义项
2. 同一温州话句子对应多个不同普通话翻译
3. 同一词条在不同来源中释义层级不同

这类不直接删除，保留并打标进入复核池。

建议字段：

- `dedup_key`
- `dup_group_id`
- `conflict_flag`
- `review_status`

## 9. 阶段 4：LLM 结构化抽取

这一阶段开始使用本地 `.env` 中的大模型 API。

LLM 只用于结构化抽取、判断和复核，不用于自由改写全量数据。

### 9.1 词条抽取目标

对词典与词汇类条目，抽取以下字段：

- `wz_word`
- `mandarin_headword`
- `definition`
- `is_fixed_expression`
- `extract_confidence`
- `needs_manual_review`

### 9.2 关于 `mandarin_headword`

注意：

1. 不是每条都必须抽出 `mandarin_headword`
2. 抽不出来时宁可留空，也不要硬造
3. `mandarin_headword` 和 `definition` 必须分开

例子：

- `阿宝儿`
- `mandarin_headword = 傻瓜`
- `definition = 对愚昧可笑者的谑称`

### 9.3 哪些词条可进入 `zh_to_wz_word`

入选条件：

1. 有 `wz_word`
2. 有高置信度 `mandarin_headword`
3. `mandarin_headword` 像词或短语，而不是一整句解释

### 9.4 LLM 适合做的工作

1. 从释义中抽核心普通话词头
2. 判断某条是否适合做词级翻译任务
3. 判断固定表达还是普通词条
4. 对冲突样本给出复核建议

### 9.5 LLM 不适合做的工作

1. 全量自由改写原始文本
2. 给所有词条强行脑补普通话对应词
3. 静默修正温州话原文
4. 无约束批量重写训练数据

## 10. 阶段 5：按任务筛样本

### 10.1 `zh_to_wz_word`

来源：

1. 词典词条
2. 资源库词汇
3. 活色生香中可稳定对齐的短词、固定短语

过滤规则：

1. [【终版】活色生香温州话.xlsx](/Users/sidneychai/Documents/微调用温州话语料/【终版】活色生香温州话.xlsx) 中如果某条只有 `原文`、没有 `普通话释义`，默认剔除
2. 不将“只有原文、没有释义”的活色生香条目用于词级翻译训练

补充说明：

1. 如果某些词条不适合作为稳定的 `zh_to_wz_word` 监督样本，但有明确释义，可优先进入后续 RAG 索引

入选要求：

1. `mandarin_headword` 可用
2. 温州话词条清晰
3. 不是纯解释句

### 10.2 `zh_to_wz_sentence`

来源：

1. 林老师句对反转
2. 词典例句反转

入选要求：

1. 温州话句子完整
2. 普通话句子完整
3. 不是明显噪声或格式错误
4. 不是无效重复

### 10.3 `wz_explain`

来源：

1. 词典释义
2. 活色生香释义
3. 资源库释义

即使没有 `mandarin_headword`，只要释义可读，也可保留为解释任务。

补充规则：

1. 活色生香中仅有 `原文`、缺失 `普通话释义` 的记录，不进入 `wz_explain`
2. 活色生香中缺释义的上下句、俗语、字数句，默认不进入第一版训练集

补充说明：

1. 这类未进入第一版训练集的条目，后续如补充到词典释义后，可再考虑入库或进入 RAG

### 10.4 `wz_rewrite_keep_skeleton`

该任务不直接来自原始 Excel，而是后续构造的增强数据。

## 11. 阶段 6：构造骨架改写样本

这是最需要控制质量的阶段。

目标是：

- 保持句式骨架、语气、结构尽量不变
- 只替换实词
- 改写后内容不同，但句法框架仍然清楚可见

### 11.1 推荐生成方式

采用两步法：

1. 先识别句中可替换实词
2. 再在约束下生成改写句

### 11.2 可替换对象

优先考虑：

1. 名词
2. 动词
3. 形容词

尽量不替换：

1. 虚词
2. 语气词
3. 结构助词
4. 句法骨架成分

### 11.3 生成后校验

每条改写样本应打标：

- `skeleton_preserved`
- `content_changed`
- `fluency_score`
- `review_needed`

### 11.4 该阶段的原则

1. 不追求大批量
2. 先追求样本自然、约束清晰
3. 如果改写看起来像自由造句，则剔除

## 12. 轻量 RAG 设计

第一版 RAG 不追求复杂系统，只做词典型检索增强。

### 12.1 RAG 的用途

RAG 主要用于补充以下能力：

1. 长尾词翻译
2. 多义词释义
3. 固定表达、俗语解释
4. 需要引用词典定义的查询

### 12.2 不建议用 RAG 替代的能力

以下任务仍应主要由微调模型负责：

1. 普通话整句转温州话
2. 温州话骨架改写
3. 输出整体风格与语气控制

### 12.3 第一版索引粒度

建议至少建立 3 类索引单元：

1. 词条级
   - `wz_word`
   - `mandarin_headword`
   - `definition`
2. 例句级
   - `example_wz`
   - `example_zh`
   - 关联词条
3. 固定表达级
   - `expression`
   - `definition`
   - 可选类型标签

### 12.4 建议的索引字段

每条索引记录建议保留：

- `doc_id`
- `source_file`
- `record_type`
- `wz_word`
- `mandarin_headword`
- `definition`
- `example_wz`
- `example_zh`
- `tags`
- `quality_score`

### 12.5 触发检索的场景

推理时建议按需触发检索，而不是所有请求都查。

优先触发 RAG 的情况：

1. 用户显式问“这个词是什么意思”
2. 用户问某个普通话词怎么说温州话
3. 输入中出现罕见词、固定表达、俗语
4. 模型对多义词需要消歧

默认不优先触发 RAG 的情况：

1. 普通短句翻译
2. 温州话骨架改写
3. 明显是高频、基础表达的任务

### 12.6 第一版实现原则

1. 先做轻量检索，不做复杂 agent 流
2. 检索内容作为上下文补充，不替代模型主输出
3. RAG 索引应从清洗后的词典和例句数据中构建
4. 检索命中结果必须可追溯到原始来源

## 13. 最终数据格式

最终训练数据统一为 `messages` 格式。

### 12.1 示例：普通话句子 -> 温州话句子

```json
{
  "task": "zh_to_wz_sentence",
  "messages": [
    {"role": "system", "content": "你是温州话助手。输出自然、简洁，不要解释。"},
    {"role": "user", "content": "把这句普通话翻成温州话：奶奶讲故事给你听。你知道吗？"},
    {"role": "assistant", "content": "娘娘讲故事匄你听。你晓得晓不得？"}
  ],
  "source": ["林老师"],
  "quality_score": 1.0
}
```

### 12.2 示例：普通话词 -> 温州话词

```json
{
  "task": "zh_to_wz_word",
  "messages": [
    {"role": "system", "content": "你是温州话助手。回答简洁准确。"},
    {"role": "user", "content": "把“傻瓜”翻成温州话。"},
    {"role": "assistant", "content": "阿宝儿。"}
  ],
  "source": ["词典"],
  "quality_score": 0.95
}
```

### 12.3 示例：温州话解释

```json
{
  "task": "wz_explain",
  "messages": [
    {"role": "system", "content": "你是温州话助手。回答简洁准确。"},
    {"role": "user", "content": "温州话“阿宝儿”是什么意思？"},
    {"role": "assistant", "content": "傻瓜。对愚昧可笑者的谑称。"}
  ],
  "source": ["词典"],
  "quality_score": 0.95
}
```

### 12.4 示例：骨架改写

```json
{
  "task": "wz_rewrite_keep_skeleton",
  "messages": [
    {"role": "system", "content": "你是温州话助手。严格保持句式骨架，只替换实词，不要解释。"},
    {"role": "user", "content": "把下面这句温州话改写成另一句温州话，保留句式和语气，只替换实词：娘娘讲故事匄你听。"},
    {"role": "assistant", "content": "阿叔讲新闻匄你听。"}
  ],
  "source": ["构造数据"],
  "quality_score": 0.9
}
```

## 14. 数据目录建议

建议后续建立如下目录结构：

```text
data/
  raw/
  staging/
  cleaned/
  llm_review/
prompts/
scripts/
```

各目录建议用途：

- `data/raw/`: 原始 Excel 或导出的原始表
- `data/staging/`: 规则清洗后的中间表
- `data/cleaned/`: 去重、归并后的结构化表
- `data/llm_review/`: LLM 抽取和复核结果
- `prompts/`: LLM 清洗和抽取 prompt
- `scripts/`: 数据处理脚本

后续如加入 RAG，可补充：

- `data/rag/`: 检索索引与导出文档

## 15. 数据切分规则

必须保留固定测试集，不允许在训练中混入近重复样本。

推荐切分：

1. `train`: `80%`
2. `valid`: `10%`
3. `test_locked`: `10%`

### 14.1 切分原则

1. 优先按去重后的组切分，而不是按单条切分
2. 同一条句子的近重复版本必须落在同一集合
3. 同一词条的多个来源版本尽量落在同一集合
4. `test_locked` 必须人工抽检

### 14.2 特别注意

如果某条词典例句已经进入句子任务，则与其高度近似的其它版本不能再随意分散到不同集合。

## 16. 质量控制

建议每个阶段都输出抽检池。

重点抽检以下样本：

1. `mandarin_headword` 抽取失败或低置信度词条
2. 多义词条
3. 温州话和普通话几乎一样的句对
4. LLM 生成的骨架改写样本
5. 跨来源重复但释义不一致的词条
6. 活色生香中格式异常、释义残缺、字段不完整的样本

建议保留以下质量字段：

- `quality_score`
- `source_count`
- `conflict_flag`
- `llm_generated`
- `needs_manual_review`

## 17. 第一版训练配置建议

## 18. 受控生成流水线 v1

这一节覆盖 2026-03-30 之后的当前主任务。

### 18.1 骨架模板格式

推荐字段：

```json
{
  "skeleton_id": "sk_xxx",
  "source_record_id": "dictionary_entry::知觉",
  "source_type": "dictionary_example",
  "source_wz_sentence": "我睏不深个,外转有响动就会知觉个",
  "source_zh_sentence": "我睡不沉的,外面一有动静就会有感觉的.",
  "scene_id": "home_life",
  "scene_label": "居家生活",
  "skeleton_template": "我睏不深个,外转有响动就会[SLOT_1]个",
  "fixed_chunks": ["我睏不深个,外转有响动就会", "个"],
  "slots": [
    {
      "slot_id": "SLOT_1",
      "surface_wz": "知觉",
      "surface_zh": "发觉",
      "slot_kind": "verb",
      "span_start": 12,
      "span_end": 14,
      "lexicon_id": "dictionary_entry::知觉||发觉||我睏不深个,外转有响动就会知觉个||我睡不沉的,外面一有动静就会有感觉的."
    }
  ],
  "slot_count": 1,
  "skeleton_quality_score": 0.93
}
```

约束原则：

1. `skeleton_template` 中只有槽位可被模型改写
2. `fixed_chunks` 必须是生成后要尽量原样保留的片段
3. 单条模板优先保留 `1` 到 `3` 个槽位
4. 槽位覆盖率过高的句子不进入第一批任务

### 18.2 槽位标注格式

推荐字段：

```json
{
  "slot_id": "SLOT_1",
  "surface_wz": "手机",
  "surface_zh": "手机",
  "slot_kind": "noun",
  "replaceable": true,
  "lexicon_id": "lex_xxx",
  "replacement_pool_key": "noun::device",
  "confidence": 0.88
}
```

当前版本不强依赖精确词性标注，但至少要区分：

1. `noun`
2. `verb`
3. `adjective`
4. `time`
5. `place`
6. `unknown`
7. `function`

其中：

1. `function` 默认不允许进入可替换槽位
2. `unknown` 可以保留，但优先级低于前三类

### 18.3 实词表格式

推荐字段：

```json
{
  "lexicon_id": "dictionary_entry::主顾||顾客||||",
  "wz_word": "主顾",
  "mandarin_headword": "顾客",
  "definition": "顾客",
  "slot_kind": "noun",
  "confidence": 0.92,
  "source_file": "【20260327最终版】温州方言词典.xlsx",
  "usable_as_slot": true
}
```

实词表只保留：

1. 有稳定 `mandarin_headword`
2. 不是固定表达整句
3. 不是量词、助词、纯语气词
4. 能自然嵌入句子槽位

### 18.4 场景表格式

推荐字段：

```json
{
  "scene_id": "digital_chat",
  "label": "数字沟通",
  "zh_keywords": ["电话", "手机", "微信", "视频", "消息"],
  "wz_keywords": ["电话", "录音"],
  "default_required_modern_words": ["手机"],
  "default_preferred_modern_words": ["微信", "视频", "消息"]
}
```

说明：

1. 场景表先走规则关键词，不先依赖模型分类
2. 场景标签主要服务于任务条件构造，而不是追求分类学完美
3. 第一版优先做十个以内高频场景

### 18.5 生成任务输入格式

推荐字段：

```json
{
  "task_id": "gen_xxx",
  "task_type": "scene_conditioned_wz_generation",
  "skeleton_id": "sk_xxx",
  "scene_id": "digital_chat",
  "scene_label": "数字沟通",
  "source_wz_sentence": "你个电话是几倈哦?",
  "skeleton_template": "你个[SLOT_1]是几倈哦?",
  "slots": [
    {
      "slot_id": "SLOT_1",
      "surface_wz": "电话",
      "surface_zh": "电话",
      "slot_kind": "noun"
    }
  ],
  "required_words": ["手机"],
  "preferred_words": ["微信", "视频"],
  "generation_count": 4
}
```

### 18.6 自动筛选质量字段

每条候选至少保留：

```json
{
  "skeleton_preserved": true,
  "required_words_hit": 1,
  "preferred_words_hit": 2,
  "slot_values_changed": 1,
  "content_changed": true,
  "duplicate": false,
  "length_ratio": 1.08,
  "quality_score": 0.91,
  "needs_manual_review": false
}
```

第一版筛选原则：

1. 固定片段未保留则直接打回
2. 必须词未命中则直接打回
3. 原句完全不变则直接打回
4. 只在槽位内变化，优先通过
5. 改动跑出骨架或像自由造句，进入复核池

### 18.7 规则和 API 的职责分工

规则优先完成：

1. 可替换实词表构建
2. 骨架模板抽取
3. 槽位标注
4. 场景标签初分
5. 自动去重
6. 骨架保持校验
7. 必须词命中检查

API 优先完成：

1. 基于任务输入生成候选句
2. 对低分候选做二次改写
3. 对边界样本做自然度复核

### 18.8 第一批试运行建议

不要一开始跑上万条。先按下面顺序做：

1. 规则抽出 `100` 到 `300` 条高质量骨架模板
2. 每条模板只保留 `1` 到 `2` 个槽位
3. 每条任务生成 `3` 到 `4` 个候选
4. 先看自动通过率，再决定是否扩容

### 18.9 当前建议命令

```bash
python3 scripts/build_example_mined_lexicon.py
python3 scripts/build_controlled_generation_assets.py
python3 scripts/tag_scene_semantics.py
python3 scripts/review_example_mined_scene_tags.py --provider qwen --threshold 0.6 --limit 8
python3 scripts/build_controlled_generation_assets.py --example-mined-lexicon data/controlled_generation/assets/example_mined_lexicon_reviewed.jsonl
python3 scripts/tag_scene_semantics.py --example-mined data/controlled_generation/assets/example_mined_lexicon_reviewed.jsonl
python3 scripts/generate_controlled_sentences.py --dry-run --limit 5
python3 scripts/filter_controlled_generation_candidates.py
```

其中新增的 `example_mined` 链路负责：

1. 从现有例句里补充“苹果、香蕉、手机、电脑、牛奶、鸡蛋”这一类高频现代生活词
2. 先用规则打初始 `topic_scene`
3. 再把低置信度词项单独交给 API 做场景重标注
4. 最后将 reviewed 词表重新并入主构建，避免把泛词硬塞进不合适的骨架

### 18.10 场景优先原则

当前实现必须从“场景先行”出发，而不是先抽骨架再硬塞词。

正确顺序应为：

1. 先定义普通话语义场景
2. 再给骨架打场景标签
3. 再给可替换实词打场景标签
4. 再将现代词和地名按场景匹配进任务
5. 最后才做受控生成

建议第一版固定场景池至少包括：

1. `transport_trip`
   交通规划、路线询问、出行问答
2. `shopping_payment`
   网购、支付、退款、快递
3. `life_knowledge`
   日常知识、生活经验、常识解释
4. `daily_chat`
   闲聊、情绪表达、轻对话
5. `digital_chat`
   手机、消息、视频、账号、设备
6. `food_dining`
   吃饭、点单、口味、外卖
7. `home_life`
   家务、家居、家庭生活
8. `work_study`
   工作协同、学习、会议、文档
9. `health_medical`
   健康、看病、挂号、消毒
10. `weather_safety`
   天气、预警、安全提醒

骨架和词条都要至少保留：

```json
{
  "scene_tags": ["daily_chat", "home_life"],
  "scene_scores": {
    "daily_chat": 3,
    "home_life": 1
  },
  "primary_scene_id": "daily_chat",
  "primary_scene_confidence": 0.75
}
```

任务构造规则更新为：

1. 默认只允许同场景或相邻场景的词条进入同一个骨架
2. 外部“新事物名词”和“地名”必须先打场景标签，再按比例注入
3. 场景冲突的词条和骨架不直接配对
4. 如果场景不明确，宁可降级为 `review`，不要硬造

### 18.11 主题场景与句子功能分层

场景标签不应只有一层。至少分为：

1. `topic_scene`
   主题领域
2. `speech_function`
   句子功能

例如：

- “怎么坐地铁去南站？”
  - `topic_scene = transport_trip`
  - `speech_function = question`
- “这个订单怎么退款？”
  - `topic_scene = shopping_payment`
  - `speech_function = question`
- “你不用恁爱心清,屋里个事干阿着管个.”
  - `topic_scene = home_life`
  - `speech_function = reminder`

推荐字段：

```json
{
  "topic_scene_tags": ["transport_trip", "daily_chat"],
  "primary_topic_scene": "transport_trip",
  "topic_scene_confidence": 0.72,
  "speech_function_tags": ["question", "planning"],
  "primary_speech_function": "question",
  "speech_function_confidence": 0.81
}
```

标注优先级：

1. 普通话来源句
2. 温州话来源句
3. 锚点词、例句和释义
4. 抽象骨架本身

原则：

1. `topic_scene` 主要看语义主题
2. `speech_function` 主要看句式、语气、标点
3. 骨架优先继承来源句标签，不强行单独主判
4. 允许多标签，不强行只留一个标签

## 19. 旧训练路线说明

此前仓库中保留过一套 `stage1` 翻译 / 解释 SFT 数据与 `QLoRA` 训练脚本。

自 2026-03-30 方案转向后，这条路线已经不再是当前主线，相关文件已从仓库中移除，包括：

1. `data/final/stage1_*`
2. `scripts/build_sft_datasets.py`
3. `configs/ms_swift_stage1_qlora.sh`
4. `configs/llamafactory_stage1_*`

当前默认不维护这套训练入口。后续如果重新评估微调，也应基于受控生成主线重新设计数据，而不是恢复旧 `stage1` 资产。

## 20. 推理要求

推理时默认不需要模型输出思维链。

要求：

1. 输出直接答案
2. 不解释过程
3. 优先自然、短促、符合温州话表达
4. 骨架改写任务必须守约束
5. 如果启用 RAG，检索结果只作为辅助上下文，不应机械照抄
6. 词义类和长尾词查询可以优先走“检索增强 + 生成”

## 21. 本项目当前明确不做的事情

第一版暂不做：

1. 全量微调
2. RLHF / DPO
3. 多轮对话复杂人格训练
4. 长上下文训练
5. 自由生成式“扩写大量温州话语料”

## 22. 建议的下一步执行顺序

按当前主线，建议按以下顺序推进：

1. 编写 Excel -> 中间表导出脚本
2. 编写规则清洗与去括号注音脚本
3. 编写去重与冲突归并脚本
4. 编写 LLM 抽取 `mandarin_headword` 的脚本
5. 构建受控生成骨架、槽位和词表资产
6. 做场景与语用功能标签
7. 生成小规模 pilot 任务
8. 跑 API 生成、规则筛选和 critic 复核
9. 继续收紧任务结构，直到自然度稳定
10. 只有在基础模型受控生成效果不够时，再重新评估微调

## 23. 当前已实现脚本

目前已经实现以下脚本：

1. `scripts/export_staging.py`
   将 4 个 Excel 导出为统一的 `staging` 中间表，并完成第一层规则清洗
2. `scripts/dedup_staging.py`
   对 `staging` 层做 exact duplicate 标记、冲突标记和第一版候选筛选
3. `scripts/extract_mandarin_headwords.py`
   使用 `.env` 中配置的 API 模型，为词条抽取 `mandarin_headword`
4. `scripts/build_controlled_generation_assets.py`
   构建骨架模板、可替换词表、外部词表和生成任务
5. `scripts/build_example_mined_lexicon.py`
   从例句中补充高频现代生活词
6. `scripts/tag_scene_semantics.py`
   为词条、骨架、外部词打 `topic_scene` 和 `speech_function`
7. `scripts/generate_controlled_sentences.py`
   按受控任务调用模型生成候选
8. `scripts/filter_controlled_generation_candidates.py`
   做骨架保持、槽位改动和命中率筛选
9. `scripts/review_generation_naturalness.py`
   对规则通过样本做自然度 critic

### 23.1 当前产物

当前已生成：

- `data/staging/staging_records.jsonl`
- `data/staging/staging_records.csv`
- `data/staging/staging_summary.json`
- `data/cleaned/cleaned_records_all.jsonl`
- `data/cleaned/cleaned_records_primary.jsonl`
- `data/cleaned/cleaned_records_primary.csv`
- `data/cleaned/review_candidates.jsonl`
- `data/cleaned/cleaned_summary.json`
- `data/llm_review/mandarin_headwords_heuristic.jsonl`
- `data/llm_review/mandarin_headwords.jsonl`
- `data/controlled_generation/assets/*`
- `data/controlled_generation/skeletons/*`
- `data/controlled_generation/tasks/*`
- `data/controlled_generation/candidates/*`
- `data/controlled_generation/filtered/*`
- `data/controlled_generation/reports/*`
- `data/controlled_generation/tags/*`

### 23.2 当前运行顺序

```bash
python3 scripts/export_staging.py
python3 scripts/dedup_staging.py
python3 scripts/extract_headwords_heuristic.py
python3 scripts/extract_mandarin_headwords.py --limit 100
python3 scripts/build_example_mined_lexicon.py
python3 scripts/build_controlled_generation_assets.py
python3 scripts/tag_scene_semantics.py
python3 scripts/review_example_mined_scene_tags.py --provider qwen --threshold 0.6 --limit 8
python3 scripts/build_generation_pilot.py
python3 scripts/generate_controlled_sentences.py --input data/controlled_generation/tasks/generation_tasks_pilot_v2.jsonl --provider qwen
python3 scripts/filter_controlled_generation_candidates.py --input data/controlled_generation/candidates/pilot_v5_qwen_raw.jsonl --output-dir data/controlled_generation/filtered/pilot_v5_qwen
python3 scripts/review_generation_naturalness.py --input data/controlled_generation/filtered/pilot_v5_qwen/generation_candidates_accepted.jsonl --output data/controlled_generation/filtered/pilot_v5_qwen/generation_candidates_critic.jsonl --summary data/controlled_generation/filtered/pilot_v5_qwen/critic_summary.json --provider qwen
```

说明：

1. `scripts/extract_headwords_heuristic.py` 不调用 API，用于先铺底抽词头
2. `scripts/extract_mandarin_headwords.py` 会调用 `.env` 中配置的 API，只处理启发式未解决或低置信度词条
3. 建议先用小 `limit` 试跑，确认抽词头质量
4. API 返回结果写入 `data/llm_review/mandarin_headwords.jsonl`
5. 当前仓库默认以受控生成验证为主，不再内置旧 `stage1` 微调脚本

## 24. 当前方案结论

当前路线固定为：

- 模型：`Qwen/Qwen3.5-9B-Base`
- 训练：`QLoRA SFT`
- 架构：`微调 + 轻量 RAG`
- 框架：`ms-swift`
- 数据：多任务 `messages` 格式
- 清洗：规则优先，LLM 只做抽取、判断、复核、受控生成
- 训练节奏：先基础映射，后骨架改写
- 推理：高频翻译与改写优先走微调模型，长尾词义和词典查询按需触发检索

后续所有脚本和目录设计默认围绕本 README 执行。
