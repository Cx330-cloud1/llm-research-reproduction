# SCM4LLMs：论文复现与长历史记忆实验

## 一、项目简介

本目录整理 SCM4LLMs 原论文相关代码、本地复现实验，以及基于 PersonaMem-32K、128K、1M 的扩展实验。

原论文复现与 PersonaMem 扩展实验属于不同实验任务，不应直接混同。

## 二、目录结构

| 目录 | 内容 |
|---|---|
| `original/` | SCM4LLMs 原项目代码及本地复现修改 |
| `original/book/` | Book Summarization 本地复现 |
| `original/dialogue/` | Dialogue Generation 本地复现 |
| `original/meeting/` | Meeting Summarization 本地复现 |
| `32k/` | PersonaMem-32K 代码、指标与审计 |
| `128k/` | PersonaMem-128K 指标与审计 |
| `1m/` | PersonaMem-1M 指标与审计 |

## 三、PersonaMem 实验结果

| 数据集 | 问题数 | 正确数 | 准确率 | 无法解析 | Controller 触发次数 |
|---|---:|---:|---:|---:|---:|
| PersonaMem-32K | 589 | 301 | 51.10% | 25 | 2 |
| PersonaMem-128K | 2727 | 1178 | 43.20% | 228 | 0 |
| PersonaMem-1M | 2674 | 934 | 34.93% | 234 | 7 |

上述结果来自本地实验的 `personamem_scm_metrics_qwen7b.json`，不是原论文官方报告的指标。

不同数据集版本的题目构成存在差异，准确率不应被直接解释为严格控制变量下的长度退化曲线。

## 四、实验配置

- 实验设置：compatibility
- 文本模型：Qwen2.5-7B-Instruct
- 向量检索模型：Contriever
- 检索候选数：Top-6
- 最大历史记忆 Token 数：2500
- 上一轮对话最大 Token 数：500
- 128K / 1M 最大新生成 Token 数：512
- 128K / 1M 模型上下文上限：32768 tokens

32K 结果文件未单独记录 `model_context_limit` 和 `max_new_tokens`，不在此推定其数值。

### 长历史长度说明

32K、128K 和 1M 是 PersonaMem 数据集的历史长度版本，不代表模型在单次推理时直接读取相应长度的完整历史。

128K 实验 Prompt Token 统计：

- 中位数：1522
- P95：2694
- 最大值：3217

1M 实验 Prompt Token 统计：

- 中位数：1353
- P95：2470
- 最大值：3252

## 五、代码复用关系

PersonaMem-128K 和 PersonaMem-1M 没有各自独立的 Python 执行脚本。

相关实验复用 `32k/` 中的多数据集通用脚本，包括：

- `build_scm_personamem_dedup.py`
- `audit_scm_dedup_multi.py`
- `generate_scm_summaries_qwen_multi.py`
- `build_scm_contriever_cache_multi.py`
- `run_scm_retrieval_multi.py`
- `run_scm_controller_multi.py`
- `run_scm_personamem_qa_multi.py`

这些脚本最初位于本地 `adapters/scm_personamem/` 目录。

归档后运行时需要根据实际数据路径调整参数。当前仓库不保证在全新环境中直接一键运行。

## 六、实验审计说明

### 数据隔离

长历史版本使用独立数据目录。128K 使用去重后的适配器目录。

数据构建脚本包含依据问题位置截取可见历史的逻辑，以避免使用问题发生之后的对话内容。

上述内容属于代码设计与已有审计记录，不能替代对完整运行过程的独立验证。

### 检索控制器

三组实验的 Controller 触发次数分别为 2、0、7。

尤其是 128K 实验没有记录到 Controller 触发，因此不能将其准确率解释为动态控制器带来的性能提升。

### 输出解析

无法解析的模型输出数量分别为 25、228、234。

这些输出影响评测结果，应在后续分析中单独讨论。

## 七、数据与模型依赖

为控制仓库体积，不上传：

- PersonaMem 大型数据文件及中间结果
- Contriever 缓存
- 模型权重
- 大规模 JSONL 中间文件
- 临时日志、PID 文件
- API 密钥和私有配置

部分 JSON 指标记录了原始 AutoDL 绝对路径，这些路径仅用于实验溯源，不具备跨机器可移植性。

## 八、原论文与本地复现的区别

`original/` 包含 SCM4LLMs 原项目代码以及本地修改，不是未经修改的官方源码快照。

Book、Dialogue、Meeting 复现实验与 PersonaMem 长历史问答实验分别记录。

本仓库不将本地扩展实验的结果声明为原论文官方复现精度。

## 九、当前状态

- [x] 整理 SCM4LLMs 原论文相关代码
- [x] 整理 Book、Dialogue、Meeting 本地复现实验
- [x] 整理 PersonaMem-32K 实验
- [x] 整理 PersonaMem-128K 实验指标与审计
- [x] 整理 PersonaMem-1M 实验指标与审计
- [x] 核实三个版本的指标 JSON
- [x] 确认 128K / 1M 复用通用脚本
- [ ] 完成敏感信息及版权内容检查
- [ ] 完成 GitHub 提交与上传
