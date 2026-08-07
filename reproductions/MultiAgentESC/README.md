# MultiAgentESC 论文复现

本项目用于复现论文：

**MultiAgentESC: Multi-Agent Framework for Emotional Support Conversation**

研究方向：

- 情感支持对话（Emotional Support Conversation）
- 大语言模型智能体（LLM Agents）
- 多智能体协作（Multi-Agent Collaboration）


本项目基于公开代码实现，在本地环境中复现 MultiAgentESC 的核心流程，并使用本地大语言模型替代论文中的大规模模型，验证方法的可运行性。


---

## 项目简介

情感支持对话要求模型不仅生成自然语言回复，还需要理解用户情绪状态，并选择合适的支持策略。

MultiAgentESC 将任务拆分为多个协作智能体，通过角色分工提升情感理解和回复质量。


整体流程：

```
用户输入
    ↓
Dialogue Analysis Agent
    ↓
Emotion & Need Understanding
    ↓
Strategy Planning Agent
    ↓
Experience Retrieval
    ↓
Multi-Agent Discussion
    ↓
Response Generation
```


相比传统单模型生成方式，多智能体框架能够：

- 分析用户情绪与需求
- 选择合适的支持策略
- 利用历史经验辅助生成
- 通过多个角色协作优化回复


---

## 复现目标

本项目主要目标：

- 理解 MultiAgentESC 方法设计
- 搭建完整本地运行环境
- 复现论文核心实验流程
- 验证多智能体情感支持生成框架
- 分析不同组件对生成效果的影响


---

## 论文信息

| 项目 | 内容 |
| --- | --- |
| 任务 | 情感支持对话生成 |
| 数据集 | ESConv |
| 方法 | Multi-Agent Emotional Support Conversation |
| 核心思想 | 多智能体协作生成支持性回复 |
| 原始模型 | 大规模 LLM |
| 本复现模型 | Qwen2.5-7B |


---

## 当前复现环境

| 项目 | 配置 |
| --- | --- |
| 操作系统 | Windows |
| Python | 3.14 |
| 推理框架 | Ollama |
| Backbone | Qwen2.5-7B |
| 数据集 | ESConv |
| Retrieval | Sentence Transformer |
| Embedding模型 | all-MiniLM-L6-v2 |


---

## 项目结构

```
MultiAgentESC/

├── data/
│   └── ESConv 数据集

├── scripts/
│   ├── run_generations.py      # 生成实验入口
│   ├── methods.py              # 不同方法实现
│   ├── dataset_utils.py        # 数据处理与检索
│   ├── metrics.py              # 指标计算
│   ├── build_table1.py         # Table 1 构建
│   ├── build_table4.py         # 消融实验结果
│   └── run_table3.py           # 自动评价

├── tests/
│   └── 离线测试

├── docs/
│   └── 复现说明

├── config.full.json             # 原论文完整配置
├── config.smoke.json            # 快速测试配置
└── config.qwen7b.json           # 本地复现配置
```


---

# 环境安装

## 创建虚拟环境

```bash
python -m venv .venv
```

激活：

Windows:

```bash
.venv\Scripts\activate
```


## 安装依赖

基础环境：

```bash
pip install -r requirements.txt
```

完整实验环境：

```bash
pip install -r requirements-full.txt
```


---

# 模型准备

本项目使用 Ollama 运行本地模型。


安装 Qwen2.5-7B：

```bash
ollama pull qwen2.5:7b
```


检查模型：

```bash
ollama list
```


---

# 快速验证

## 离线测试

验证代码结构：

```bash
python -m tests.test_offline
```


预期：

```
Ran 7 tests

OK
```


---

# 实验运行

## 小规模测试

运行 5 条样本：

```bash
python -m scripts.run_generations \
--config config.qwen7b.json \
--methods "MultiAgentESC (Ours)" \
--max-targets 5
```


用于确认：

- 模型调用正常
- MultiAgentESC 流程正常
- 输出格式正确


---

## 小规模正式实验

运行 100 条样本：

```bash
python -m scripts.run_generations \
--config config.qwen7b.json \
--methods "MultiAgentESC (Ours)" \
--max-targets 100
```


---

## 完整实验

运行全部 ESConv 数据：

```bash
python -m scripts.run_generations \
--config config.qwen7b.json \
--methods "MultiAgentESC (Ours)"
```


---

# 实验结果

生成结果默认保存：

```
outputs/

├── generations/
│   └── qwen2-5-7b/
│       └── multiagentesc-ours.jsonl

├── logs/
│
└── table1_manifest.json
```


---

# 复现说明

由于原论文实验依赖：

- Qwen2.5-32B
- LLaMA3-70B

等大规模模型，本项目采用：

- Qwen2.5-7B
- 本地 Ollama 推理环境

进行资源友好的复现。


因此，本项目重点验证：

- MultiAgentESC 方法流程
- 多智能体协作机制
- 本地 LLM 可运行性


而不是严格复制论文原始模型性能。


---

# 当前进展

## 已完成

- [x] 项目环境搭建
- [x] ESConv 数据读取
- [x] MultiAgentESC 方法实现
- [x] 本地 Qwen2.5-7B 接入
- [x] Retrieval 模块运行
- [x] 多智能体生成流程测试


## 进行中

- [ ] Table 1 指标复现
- [ ] BLEU / Distinct 等指标计算
- [ ] Table 2 人工评价
- [ ] Table 3 自动评价
- [ ] Table 4 消融实验


---

# 后续计划

- 完成主要实验指标复现
- 对比论文结果差异
- 分析本地模型替代带来的影响
- 完善实验记录与复现报告


---

# Reference

Paper:

**MultiAgentESC: Multi-Agent Framework for Emotional Support Conversation**

Dataset:

**ESConv: A Dataset for Emotional Support Conversation**
