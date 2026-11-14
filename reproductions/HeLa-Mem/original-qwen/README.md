# HeLa-Mem 本地 Qwen 适配：LongMemEval 归档

本目录记录 HeLa-Mem 的本地 Qwen 适配及 LongMemEval 历史实验汇总。有关源码、指标与版本的具体对应关系仍需进一步核验。

## 已记录结果：full_v2

| 指标 | 数值 |
|---|---:|
| 总题数 | 500 |
| 正确 | 364 |
| 不正确 | 136 |
| 准确率 | 72.80% |

来源：[`metrics.json`](metrics.json)，实验名为 `longmemeval_mem_qwen_full_v2`。

## 记录的配置

- `top_k = 15`
- `semantic_top_k = 5`
- `use_consolidation = false`
- `max_flipped = 3`
- `learning_rate = 0.02`
- `activation_alpha = 0.1`
- `spreading_threshold = 0.4`
- `decay_rate = 0.995`
- `keyword_weight = 0.7`
- `tau = 1e7`

## 注意

- 72.80% 对应 **full_v2** 的归档结果，不应当视为 v21、v22、v23 等后续修复版本的最终表现。
- 运行模型权重、环境版本与完整复现配置尚待独立确认。
- 各个实验规模所用的数据集、模型及代码版本可能不同，不能直接以这些归档指标进行严格横向比较。
