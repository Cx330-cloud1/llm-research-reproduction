# HeLa-Mem 论文复现归档

本目录按**方法版本**与**评测数据设置**组织 HeLa-Mem 复现材料。每个子目录都有独立说明；指标来自历史运行记录，不能直接作为控制变量后的横向对比。

## 快速导航

| 目录 | 内容 | 代码与指标状态 |
|---|---|---|
| [`original/`](original/) | HeLa-Mem 原始源码的筛选归档 | 上游版本与许可证需进一步核验 |
| [`original-qwen/`](original-qwen/) | 本地 Qwen 适配与 LongMemEval 历史归档 | [`README`](original-qwen/README.md) · [`metrics`](original-qwen/metrics.json) |
| [`32k/`](32k/) | PersonaMem 32K 候选实验 | [`README`](32k/README.md) · [`metrics`](32k/metrics.json) |
| [`128k/`](128k/) | PersonaMem 128K 指标归档 | [`README`](128k/README.md) · [`metrics`](128k/metrics.json) |
| [`1m/`](1m/) | PersonaMem 1M 适配源码与缓存评测 | [`README`](1m/README.md) · [`metrics`](1m/metrics.json) · [`复现指南`](1m/REPRODUCE.md) |

## 历史评测摘要

| 实验 | 已记录正确数 | 题目数 | 已记录准确率 | 核验级别 |
|---|---:|---:|---:|---|
| PersonaMem 32K **候选** | 279 | 589 | 47.37% | 32K 对应关系待独立核实 |
| PersonaMem 128K | 1267 | 2727 | 46.46% | 完整运行配置待独立核实 |
| PersonaMem 1M | 未独立核实 | 2674 | **暂不报告** | 结果完整性检查通过，准确率待复核 |
| Qwen LongMemEval `full_v2` | 364 | 500 | 72.80% | 历史 full_v2 版本，非后续修复版最终指标 |

**指标口径：** JSON 中的 `accuracy` 以 **0–1** 小数表示；上表显示百分比。`null` 表示未独立核实，**不是零分**。字段口径见 [`METRICS_FORMAT.md`](METRICS_FORMAT.md)。

**证据边界：** 1M 的 2674 条结果、0 缺失、0 重复、0 错误行来自 2026-11-14 的本地完整性审计；这不证明 1M 正确率，也不证明缓存算法与独立编码严格等价。

## 复现入口

需要运行 PersonaMem 1M 的读者请按顺序阅读：

1. [`1m/README.md`](1m/README.md)：范围、结构和状态。
2. [`1m/REPRODUCE.md`](1m/REPRODUCE.md)：依赖、外部数据/工具、模型环境变量和经过源码核对的 CLI。
3. [`1m/STATUS.md`](1m/STATUS.md)：已完成与待验证事项。

不同规模历史实验的代码/模型/数据设置未全部锁定，不应假定 1M 代码可直接重现 32K、128K 或 LongMemEval 的原始数值。本仓库不公开模型权重、原始对话、逐题预测与记忆缓存。请遵守原论文、模型和数据集的许可要求。
