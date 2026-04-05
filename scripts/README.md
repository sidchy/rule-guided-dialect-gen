# Scripts

`scripts/` 目录目前包含两类内容：

## 1. 当前优先维护入口

这些脚本是薄包装入口，调用 `src/wz_pipeline/` 中的当前主线逻辑：

- `extract_dictionary_training_sentences.py`
- `generate_long_sentences_fewshot_batch.py`
- `build_modern_anchor_candidates.py`
- `build_controlled_generation_assets.py`
- `export_review_package.py`
- `import_review_results.py`
- `promote_reviewed_samples.py`
- `register_legacy_run.py`
- `materialize_seed_curated.py`
- `build_training_candidates.py`

## 2. 历史实验脚本

这些脚本保留是为了复盘、兼容历史资产或后续迁移，不代表当前默认主线：

- dense slot 相关
- compound 拼接相关
- 早期 controlled generation 试验脚本
- 各类 review/筛样辅助脚本

当前处理原则：

- 可以保留
- 不删除历史资产
- 不把它们默认当成主线入口
- 如需复用，先明确登记为 `legacy`，再决定是否迁移进 `src/wz_pipeline/`

补充说明：

- `generate_long_sentences_fewshot_batch.py` 当前已经支持 scene 子场景拆分、speech-act 轮转、跨 run 去重和 task 多样性审计。
- `build_controlled_generation_assets.py` 负责产出 scene catalog / modern-word catalog 这类主线资产；如果要扩场景，优先改这里和 `configs/scene_policy.json`，而不是直接在 pipeline 里写死。
