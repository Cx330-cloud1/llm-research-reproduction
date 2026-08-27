# EmoDynamiX 复现

本项目用于复现论文 **EmoDynamiX: Emotional Dynamics Modeling for Empathetic Dialogue** 的实验流程，并补充官方代码中缺失的 ERC（Emotion Recognition in Conversation）训练模块。

## 项目结构

```
EmoDynamiX
│
├── table_reproduction
│   ├── results
│   ├── tables
│   ├── README.md
│   └── verify_table1.py
│
└── erc_training
    └── train_erc.py
```

## 复现进展

### 1. Table 实验复现

路径：

```
table_reproduction/
```

已完成：

- 配置 EmoDynamiX 复现实验环境。
- 基于官方代码完成 Table 1 复现流程。
- 验证实验结果生成过程。
- 整理实验结果与表格文件。

包含：

- 实验结果文件
- 表格生成文件
- 验证脚本

---

### 2. ERC 模块训练补充

路径：

```
erc_training/
```

官方仓库未提供完整 ERC checkpoint，导致完整实验流程无法直接运行。

针对该问题，实现了 ERC 模块训练流程。

已完成：

- DailyDialog 情绪数据加载。
- RoBERTa 编码器初始化。
- SequentialERC 模型训练流程实现。
- ERC checkpoint 本地生成。

训练脚本：

```
erc_training/train_erc.py
```

生成模型：

```
sequential_erc_model.pth
```

由于模型文件较大，checkpoint 未上传至 GitHub，仅保存在本地实验环境。

---

## 当前状态

已完成：

- RTX4090 实验环境配置。
- SequentialERC 模块加载验证。
- RoBERTa 前向推理验证。
- ERC 训练流程实现。
- ERC checkpoint 生成。
- EmoDynamiX 复现工程结构整理。

---

## 后续计划

- 根据论文设置优化 ERC 数据预处理流程。
- 使用论文参数重新训练 ERC 模型。
- 将 ERC 模块接入完整 EmoDynamiX 评估流程。
- 完成更多实验表格复现。
