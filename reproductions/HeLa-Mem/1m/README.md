# PersonaMem 1M：HeLa-Mem 复现

本目录存放 **HeLa-Mem 在 PersonaMem 1M 设置下的适配源码、共享前缀缓存构建、问答评测脚本与测试**。本仓库不包含原始数据、模型权重和逐题问答内容。

## 实验状态

| 指标 | 数值 |
|---|---:|
| 预期 / 数据集 / 结果记录 | 2674 / 2674 / 2674 |
| 已评分记录 | 2674 |
| 缺失 / 重复 ID / 错误行 | 0 / 0 / 0 |
| 准确率 | **待独立核实** |

这些数字来自本地 `audit.json` 的完整性审计。**已评分不等于答对**；没有核实的正确率不应填入数值。见 [`metrics.json`](metrics.json) 与 [`STATUS.md`](STATUS.md)。

## 代码入口

| 目录 / 文件 | 功能 |
|---|---|
| [`hela_mem/`](hela_mem/) | Hebbian 情景记忆图、知识库、检索、结构化语义适配及本地 Qwen 接口 |
| [`scripts/build_personamem_prefix_cache.py`](scripts/build_personamem_prefix_cache.py) | 共享前缀编码与 Cutoff 快照 |
| [`scripts/eval_personamem_cached.py`](scripts/eval_personamem_cached.py) | 缓存加载与问答评测 |
| [`scripts/eval_personamem_local.py`](scripts/eval_personamem_local.py) | PersonaMem 数据加载与原始工具函数适配 |
| [`scripts/eval_personamem_local_full.py`](scripts/eval_personamem_local_full.py) | 完整知识记忆与本地推理适配 |
| [`tests/`](tests/) | 结构化语义与缓存等价性相关测试 |

## 开始复现

**请阅读 [`REPRODUCE.md`](REPRODUCE.md)**，其中包含必要的外部数据/工具、环境变量、运行路径、已核实的命令行参数和限制。

使用的模型配置包括 Qwen3-8B-Base 与 all-MiniLM-L6-v2；模型版本、CUDA / vLLM 兼容性及缓存与独立编码的严格等价性仍需进一步验证。本目录不代表官方实验结果。
