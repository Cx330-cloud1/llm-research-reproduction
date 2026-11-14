# HeLa-Mem 本地 Qwen 适配 · LongMemEval

> **状态：`full_v2` 历史结果已归档，后续修复版本结果尚未统一核验。**

| 指标 | 数值 |
|---|---:|
| 题目数 | 500 |
| 正确 / 不正确 | 364 / 136 |
| 已记录准确率 | **72.80%** |

来源：[`metrics.json`](metrics.json)，记录的实验标签为 `longmemeval_mem_qwen_full_v2`。

## 已记录的关键参数

| 参数 | 数值 |
|---|---:|
| `top_k` | 15 |
| `semantic_top_k` | 5 |
| `use_consolidation` | false |
| `max_flipped` | 3 |
| `learning_rate` | 0.02 |
| `activation_alpha` | 0.1 |
| `spreading_threshold` | 0.4 |
| `decay_rate` | 0.995 |
| `keyword_weight` | 0.7 |
| `tau` | 1e7 |

源码与具体模型权重的对应关系、依赖版本及完整运行配置仍需核实。72.80% 仅属于 `full_v2` 的记录，不应视为 v21/v22/v23 等后续修复版的最终结果。各规模历史指标不可直接进行公平性比较。口径见 [`../METRICS_FORMAT.md`](../METRICS_FORMAT.md)。
