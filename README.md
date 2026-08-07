# LLM Research Reproduction

一个用于复现与分析大语言模型（LLM）、智能体（Agent）以及人机交互相关论文的个人研究实验库。

本仓库记录论文阅读、代码复现、实验验证以及方法分析过程，目标是建立可复现、可扩展的 LLM Research Workflow。


---

# Research Areas

当前关注方向：

- Large Language Model Agents
- Multi-Agent Systems
- Emotional Support Dialogue
- Memory-Augmented LLM
- Human-centered AI


仓库内容包括：

- 论文复现实验
- 实验配置文件
- 代码实现
- 结果记录
- 论文分析笔记
- Research Ideas


---

# Repository Structure

```
llm-research-reproduction/

├── reproductions/
│
│   ├── MultiAgentESC/
│   │   ├── code
│   │   ├── configs
│   │   ├── experiments
│   │   └── README.md
│   │
│   ├── EmoDynamiX/
│   │   ├── reproduction notes
│   │   ├── validation results
│   │   └── README.md
│
├── papers/
│   └── Literature collection
│
├── notes/
│   └── Paper reading notes
│
└── experiments/
    └── Experimental records
```


---

# Reproduction Projects

| Project | Area | Status |
| --- | --- | --- |
| MultiAgentESC | Emotional Support Dialogue / Multi-Agent LLM | In Progress |
| EmoDynamiX | Emotional Strategy Prediction / Emotional Dialogue | Completed Validation |


---

# Current Projects


## MultiAgentESC

**Multi-Agent Framework for Emotional Support Conversation**

方向：

- Emotional Support Dialogue
- LLM Agent
- Multi-Agent Collaboration


当前进展：

已完成：

- [x] 本地实验环境搭建
- [x] ESConv 数据集加载
- [x] MultiAgentESC 方法代码分析
- [x] 本地 Qwen2.5-7B 接入
- [x] Retrieval 模块运行
- [x] MultiAgentESC 生成流程验证


进行中：

- [ ] Table 1 指标复现
- [ ] Table 2 人工评价
- [ ] Table 3 自动评价
- [ ] Table 4 消融实验


复现说明：

由于论文原实验依赖大规模模型，本项目采用本地 LLM 进行资源友好的复现，重点验证方法流程与实验可运行性。


详细信息：

```
reproductions/MultiAgentESC
```


---

## EmoDynamiX

**EmoDynamiX: Emotional Support Dialogue Strategy Prediction by Modelling MiXed Emotions and Discourse Dynamics**

方向：

- Emotional Support Dialogue
- Emotion Modeling
- Dialogue Strategy Prediction


当前进展：

已完成：

- [x] 官方代码环境分析
- [x] 官方 checkpoint 配置验证
- [x] Table 1 主实验复现验证
- [x] 六项主要指标与论文结果核对
- [x] Table 2 消融实验可复现性审计


复现结果：

- Table 1 官方 checkpoint 在本地环境完成推理
- 主要指标与论文报告结果一致（四舍五入后完全匹配）


Table 2 说明：

论文中的消融实验需要对应训练配置与 checkpoint。

由于官方仓库未提供完整消融 checkpoint，本项目完成：

- 实验设置分析
- 官方代码能力检查
- 结果可复现性评估

未虚构不存在的重训练结果。


详细信息：

```
reproductions/EmoDynamiX
```


---

# Reproduction Workflow

每个论文项目遵循：

```
Paper Reading

      ↓

Method Understanding

      ↓

Environment Setup

      ↓

Code Reproduction

      ↓

Experiment Validation

      ↓

Result Analysis
```


---

# Research Focus


## LLM Agent

关注：

- Agent collaboration
- Planning and reasoning
- Tool usage
- Evaluation


## Emotional Dialogue

关注：

- Emotional understanding
- Support strategy prediction
- Empathetic response generation


## Memory Systems

关注：

- Long-term memory
- Memory retrieval
- Adaptive memory management


---

# Environment

主要实验环境：

- Python
- PyTorch
- Ollama
- HuggingFace Ecosystem
- Local LLM Inference


不同项目维护独立配置文件。


---

# Notes

本仓库中的实验会根据个人计算资源进行调整。

部分论文原始实验依赖：

- 大规模闭源模型
- 大规模 GPU 集群
- 特定训练环境


因此复现重点包括：

- 方法流程验证
- 实验环境重建
- 结果差异分析
- 可复现性评估


而不是简单复制论文数字。


---

# Future Plans

计划持续增加：

- LLM Agent 论文复现
- Memory Agent 研究
- 情感计算与对话系统实验
- Agent Evaluation 方法分析


---

# Author

Personal Research Reproduction Repository

Focus:

**LLM · Agents · Human-centered AI**
