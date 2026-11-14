# PersonaMem 32K（候选实验）

> **状态：历史指标已归档；32K 数据集映射与完整运行配置尚未独立核实。**

| 指标 | 数值 |
|---|---:|
| 总记录 | 589 |
| 正确 / 不正确 | 279 / 310 |
| 已记录准确率 | **47.37%** |

指标来源：[`metrics.json`](metrics.json)；其中 `source_result_filename = results_hela_personamem_cached.json` 仅为原始结果文件名，并非已公开的逐题数据。

## 如何使用本归档

- 这是候选结果记录，不能当作已严格验证的 32K 最终复现指标。
- 尚无足够证据证明该结果与其他规模实验采用完全一致的代码、模型和数据口径。
- 1M 实验的实现与运行方法可参考 [`../1m/REPRODUCE.md`](../1m/REPRODUCE.md)，**但不能据此推断**该 32K 历史运行使用了同一版本。
- 指标字段解释见 [`../METRICS_FORMAT.md`](../METRICS_FORMAT.md)。
