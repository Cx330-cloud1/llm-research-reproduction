# HeLa-Mem 论文复现归档

## 一、目录说明

- `original/`：官方源码的筛选副本。
- `original-qwen/`：Qwen 修改版源码和 LongMemEval 汇总指标。
- `32k/`：PersonaMem 扩展评测共享源码及 32K 候选指标。
- `128k/`：PersonaMem 128K 汇总指标。
- `1m/`：目前未找到完整结果，保留状态说明。

本目录不包含完整模型权重、数据集、记忆缓存和运行环境。

## 二、已核实结果

| 实验 | 正确数 | 总数 | 准确率 |
|---|---:|---:|---:|
| PersonaMem 32K 候选 | 279 | 589 | 47.37% |
| PersonaMem 128K | 1267 | 2727 | 46.46% |
| PersonaMem 1M | — | — | 未核实 |
| Qwen LongMemEval full_v2 | 364 | 500 | 72.80% |

32K 候选来自 `results_hela_personamem_cached.json`，
与 32K 数据集的完整运行映射仍待确认。

LongMemEval 的 72.80% 来自 `full_v2` 汇总。
后续存在 v21、v22、v23 修复记录，不能将其视为所有修复后的最终指标。

## 三、LongMemEval 配置

评测汇总记录：

- top_k：15
- semantic_top_k：5
- use_consolidation：False

其他参数见 `original-qwen/metrics.json`。
上述配置来自评测记录，实际模型权重版本仍需独立核实。

## 四、复现与公开限制

本归档仅保留筛选后的源码及汇总指标。
未上传逐题问答、原始对话、Hebbian 记忆图、缓存或 API 凭据。

源码尚需进一步核对运行依赖和完整执行命令。
三组 PersonaMem 结果不能在缺少配置一致性验证时，
直接解释为严格公平的跨方法对比。

本归档不代表全部实验已完成独立复现。
