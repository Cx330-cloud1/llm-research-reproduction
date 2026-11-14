# HeLa-Mem 论文复现归档

本目录记录 HeLa-Mem 原始代码归档、本地 Qwen 适配及 PersonaMem 不同上下文规模下的实验。**各实验数据集、配置和验证程度不同，不应将准确率直接作为严格公平的跨实验对比。**

## 目录索引

| 目录 | 内容 | 当前状态 |
|---|---|---|
| [`original/`](original/) | 官方代码的筛选归档 | 需依据上游许可证及提交版本独立核对 |
| [`original-qwen/`](original-qwen/) | 本地 Qwen 修改版与 LongMemEval 汇总 | full_v2 结果有记录，后续修复未统一核验 |
| [`32k/`](32k/) | PersonaMem 32K 候选评测归档 | 589 条结果与 32K 映射尚待核实 |
| [`128k/`](128k/) | PersonaMem 128K 指标归档 | 仅核实汇总指标，配置待独立核对 |
| [`1m/`](1m/) | PersonaMem 1M 源码、缓存构建与评测 | 2674 条结果的完整性审计通过，准确率待核验 |

## 已存档的指标

| 实验 | 正确数 | 总数 | 准确率 | 说明 |
|---|---:|---:|---:|---|
| PersonaMem 32K 候选 | 279 | 589 | 47.37% | 结果到 32K 的映射未独立确认 |
| PersonaMem 128K | 1267 | 2727 | 46.46% | 运行配置未独立确认 |
| PersonaMem 1M | — | 2674 | 暂不报告 | 已核实记录完整性，尚未独立复核正确率 |
| Qwen LongMemEval full_v2 | 364 | 500 | 72.80% | 该版本指标，不代表后续修复版最终表现 |

上述比例对应各目录中记录的历史汇总，**不是重新运行所得**。请参见 [`32k/metrics.json`](32k/metrics.json)、[`128k/metrics.json`](128k/metrics.json)、[`original-qwen/metrics.json`](original-qwen/metrics.json) 和 [`1m/STATUS.md`](1m/STATUS.md)。

## 复现说明与限制

- 1M 的源码实现、模型配置和缓存流程见 [`1m/README.md`](1m/README.md)。
- 1M 完整性审计：预期 / 数据集 / 结果 / 已评分均为 **2674**；缺失、重复 ID、错误行均为 **0**。
- 结果完整不代表预测正确，也不能证明共享前缀缓存与独立编码严格等价。
- 各目录不包含完整模型权重、原始 PersonaMem 对话、逐题回答、私有记忆缓存或 API 凭据。
- 公开源码的依赖兼容性和端到端运行方式仍需进一步审计。

> 本仓库是研究复现档案，不代表 HeLa-Mem 作者的官方 PersonaMem 实验报告。
