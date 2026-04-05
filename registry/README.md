# Registry

本目录记录项目的元数据，不直接存放批次内容。

- `run_registry.jsonl`: 新 run 清单
- `legacy_runs.jsonl`: 历史实验资产登记
- `promotion_log.jsonl`: promotion 记录

约束：

- 这里只登记路径、run_id、provider、policy_version 等元数据。
- 方言本地词表、语法材料、运行产物和审核内容不应直接提交到 `registry/`。
