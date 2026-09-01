# EmoDynamiX Reproduction

本目录记录论文 **EmoDynamiX: Emotional Support Dialogue Strategy Prediction by Modelling MiXed Emotions and Discourse Dynamics**（NAACL 2025）的复现、环境校准、消融实验、缺失模块恢复与可复现性分析。

- Paper: https://aclanthology.org/2025.naacl-long.81/
- Official Repository: https://github.com/cw-wan/EmoDynamiX-v2
- 原始项目说明：[UPSTREAM_README.md](UPSTREAM_README.md)

> 本项目用于科研复现与可复现性分析。原论文、代码、模型和数据版权归原作者所有。

## 1. Reproduction Status

| Component | Status | Notes |
|---|---|---|
| ESConv Full Model | Completed | Metric-level reproduction |
| AnnoMI Full Model | Completed | Metric-level reproduction |
| w/o Mixed Emotion | Completed | Official code-supported ablation |
| SDDP | Completed | Metric-level reproduction, STAC F1 = 60.46 |
| ERC | Functional Reconstruction | Training and inference pipeline restored |
| w/o Graph Learning | Reconstructed | Paper-description-based |
| w/o Discourse Parser | Reconstructed | Paper-description-based |
| w/o Dummy Node | Reconstructed | Paper-description-based |
| Flat Context Diagnostic | Completed | Additional local diagnostic |

## 2. Calibrated Environment

主模型最终实验使用统一校准环境：

```text
GPU: NVIDIA GeForce RTX 4090
Python: 3.10.20
PyTorch: 2.2.1+cu121
Transformers: 4.37.2
Datasets: 2.12.0
PyArrow: 12.0.1
NumPy: 1.23.5
Pandas: 1.5.3
Scikit-learn: 1.3.0
```

主模型主要训练配置：

```text
Learning rate: 4e-6
Batch size: 16
Weight decay: 1e-3
ERC temperature: 0.5
Graph hidden dimension: 512

ESConv training steps: 3000
AnnoMI training steps: 1200
```

该环境主要依据官方 `requirements.txt` 进行校准。由于官方环境文件与依赖说明之间存在一定版本差异，因此这里记录的是本次实际完成复现的稳定环境，而不是声称与作者原始运行环境完全一致。

## 3. Main Model Results

### Macro-F1

| Dataset | Paper | Initial Reproduction | Calibrated Reproduction | Difference |
|---|---:|---:|---:|---:|
| ESConv | 27.70 | 26.63 | **28.09** | +0.39 |
| AnnoMI | 27.92 | 26.64 | **28.07** | +0.15 |

### Full Metrics

| Dataset | Paper M-F1 | Local M-F1 | Paper W-F1 | Local W-F1 | Paper Bias | Local Bias |
|---|---:|---:|---:|---:|---:|---:|
| ESConv | 27.70 | **28.09** | 32.71 | **32.04** | 0.45 | **0.4256** |
| AnnoMI | 27.92 | **28.07** | 35.33 | **34.38** | 0.50 | **0.4793** |

最终结果：

```text
ESConv
Paper: 27.70
Local: 28.09
Difference: +0.39

AnnoMI
Paper: 27.92
Local: 28.07
Difference: +0.15
```

两个数据集的 Macro-F1 均与论文结果高度接近，因此本项目认为：

> **The main EmoDynamiX results are successfully reproduced under the calibrated environment.**

## 4. Mixed Emotion Ablation

`erc_mixed` 是官方代码直接提供的控制参数，因此 `w/o Mixed Emotion` 是本项目中可信度最高的消融实验。

### Macro-F1

| Dataset | Full | w/o Mixed | Local Change | Paper Change |
|---|---:|---:|---:|---:|
| ESConv | 28.09 | 27.13 | **-0.96** | -1.80 |
| AnnoMI | 28.07 | 27.00 | **-1.07** | -3.21 |

### Weighted-F1

| Dataset | Full | w/o Mixed | Change |
|---|---:|---:|---:|
| ESConv | 32.04 | 31.55 | -0.49 |
| AnnoMI | 34.38 | 33.49 | -0.89 |

### Preference Bias

| Dataset | Paper Full | Paper w/o Mixed | Local Full | Local w/o Mixed |
|---|---:|---:|---:|---:|
| ESConv | 0.45 | 0.66 | 0.4256 | 0.4125 |
| AnnoMI | 0.50 | 0.70 | 0.4793 | 0.4057 |

移除 Mixed Emotion 后，ESConv 与 AnnoMI 的 Macro-F1 均下降，因此论文关于 Mixed Emotion representation 对策略预测具有正向贡献的主要趋势得到复现。

但需要注意：

- 本地 Macro-F1 的下降幅度小于论文报告值；
- Preference Bias 未复现论文中的变化趋势；
- 论文中移除 Mixed Emotion 后 Bias 上升，而本地两个数据集的 Bias 均下降。

因此更准确的结论是：

> **The F1 degradation trend of the Mixed Emotion ablation is reproduced, while the exact degradation magnitude and Preference Bias behavior are not reproduced.**

## 5. SDDP Reproduction

EmoDynamiX 使用 Structured Dialogue Discourse Parsing（SDDP）模块构建 discourse structure。

在初期恢复过程中，由于原始 `spanningtrees` 依赖无法正常加载，曾临时绕过 spanning-tree decoder，导致独立 SDDP evaluation 的预测树为空并得到 F1 = 0。

后续恢复了：

- 官方 `draw_tree` tree decoding 逻辑；
- `arsenal` 依赖；
- `spanningtrees` 依赖；
- `Graph` / `MST` / `KBest` 解码模块；
- SDDP checkpoint 加载；
- STAC training / evaluation pipeline。

恢复真实 decoder 后，已有 3-epoch checkpoint 得到：

```text
F1 = 57.29
```

随后按照公开训练脚本的主要配置完成 seed=1、6-epoch 实验：

```text
Dataset: STAC
Seed: 1
Learning rate: 2e-5
Epochs: 6
Train batch size: 2
Warmup ratio: 0.1
Max train contexts: 20
Max dev contexts: 37
Max test contexts: 37
Decoder: KBest spanning-tree decoder
```

最终结果：

| Experiment | F1 |
|---|---:|
| 3-epoch recovery checkpoint | 57.29 |
| 6-epoch final checkpoint | **60.46** |
| Independent checkpoint reload | **60.46** |

最终测试：

```text
F1 = 0.6045929018789143
```

保存模型后重新独立加载 checkpoint，得到完全一致的 **60.46 F1**，说明训练、保存、加载与 tree decoding 流程均已恢复。

因此 SDDP 当前状态更新为：

> **Metric-Level Reproduction Completed — STAC F1: 60.46**

需要说明的是，主模型 calibrated experiments 仍使用官方预处理数据中的 `parsed_dialogue`，并不是使用本地重新训练的 SDDP 从原始对话重新生成 discourse graph。

因此：

> SDDP 的指标级复现已经完成，但不能据此将主模型实验描述为完整的本地端到端重新训练。

详细记录见：

[`results/sddp/summary.md`](results/sddp/summary.md)

## 6. ERC Reconstruction

官方项目依赖：

```text
pre_trained_models/sequential_erc_model.pth
```

但该 checkpoint 未作为完整可直接运行资源提供。

本项目恢复了：

- SequentialERC 训练流程；
- DailyDialog emotion 数据训练；
- checkpoint 生成；
- checkpoint 加载；
- forward inference；
- logits 与 embedding 输出接口。

当前状态：

> **Functional Reconstruction**

该模块尚未完成与论文 ERC 指标的严格 metric-level reproduction。

同时，主模型 calibrated experiments 使用的是官方预处理数据中已经生成的 ERC 信息，因此主模型结果不能声称来自本地重新训练 ERC 的完整端到端流程。

## 7. Structural Ablations

论文还报告：

- w/o Graph Learning
- w/o Discourse Parser
- w/o Dummy Node

但当前官方仓库没有提供这三项实验对应的完整实现或直接控制参数。

因此本项目依据论文方法描述进行了结构重建，统一标记为：

> **Paper-description-based reconstruction**

这些实验的主要目的不是声称精确复现论文 Table 2，而是分析不同结构变化对本地模型的影响。

### ESConv

| Variant | Macro-F1 | Weighted-F1 | Bias |
|---|---:|---:|---:|
| Full baseline | 26.63 | 31.12 | 0.4352 |
| w/o Discourse Parser | 26.11 | 30.46 | 0.3734 |
| w/o Dummy Node | 26.51 | 31.44 | 0.5008 |
| w/o Graph Learning | 27.97 | 31.73 | 0.3603 |

### AnnoMI

| Variant | Macro-F1 | Weighted-F1 | Bias |
|---|---:|---:|---:|
| Full baseline | 26.64 | 33.87 | 0.5134 |
| w/o Discourse Parser | 28.02 | 36.15 | 0.3120 |
| w/o Dummy Node | 28.29 | 35.68 | 0.6183 |
| w/o Graph Learning | 28.89 | 35.10 | 0.3901 |

这些结构消融没有稳定复现论文中报告的性能下降。

特别是 reconstructed `w/o Graph Learning` 在两个数据集上均表现较强，因此不能将其视为对论文 graph ablation 的精确重复。

可能影响结果的关键因素包括：

- 论文未公开完整结构消融实现；
- graph removal 后输入信息的编码方式不同；
- emotion / strategy information 的显式表示可能影响模型性能；
- 不同实现可能改变信息容量，而不仅仅是移除 graph computation。

因此这些结果仅用于本地诊断和可复现性审计。

## 8. Flat Context Diagnostic

为进一步分析 reconstructed `w/o Graph Learning` 的异常高结果，本项目增加了普通 Flat Context 对照。

### ESConv

| Variant | Macro-F1 | Weighted-F1 | Bias |
|---|---:|---:|---:|
| Flat Context | 26.35 | 30.05 | 0.4275 |
| Full baseline | 26.63 | 31.12 | 0.4352 |
| w/o Graph + explicit information | 27.97 | 31.73 | 0.3603 |

普通 flattened context 本身没有得到异常高的结果。

较高的 reconstructed no-graph performance 与显式保留 emotion / historical strategy information 的表示方式相关。

因此，本实验说明：

> reconstructed graph ablation 对具体实现和输入信息编码方式高度敏感。

该结果不能用于证明 graph structure 本身无效。

## 9. Result Organization

```text
results/
├── calibrated/
│   ├── esconv_full.txt
│   ├── esconv_wo_mixed.txt
│   ├── annomi_full_paper1200.txt
│   ├── annomi_wo_mixed.txt
│   └── summary.md
│
├── sddp/
│   └── summary.md
│
├── table1/
├── table2/
├── diagnostics/
├── paper_reference/
└── legacy/
```

其中：

- `calibrated/`：最终主模型与 Mixed Emotion 正式结果；
- `sddp/`：SDDP 指标级复现记录；
- `table1/`：早期主模型实验；
- `table2/`：结构消融与早期实验；
- `diagnostics/`：额外诊断实验；
- `paper_reference/`：论文指标参考；
- `legacy/`：历史实验与审计记录。

## 10. Reproduction Reliability

| Experiment | Reproduction Type | Reliability |
|---|---|---|
| ESConv Full | Calibrated official pipeline | High |
| AnnoMI Full | Calibrated official pipeline | High |
| w/o Mixed Emotion | Official code-supported ablation | High |
| SDDP | Metric-level reproduction | High |
| ERC | Functional reconstruction | Medium |
| w/o Graph Learning | Paper-description-based reconstruction | Limited |
| w/o Discourse Parser | Paper-description-based reconstruction | Limited |
| w/o Dummy Node | Paper-description-based reconstruction | Limited |
| Flat Context | Local diagnostic experiment | High for local analysis |

## 11. Final Conclusion

本次 EmoDynamiX 复现得到以下结论：

1. **主模型完成指标级复现。**

```text
ESConv
Paper 27.70 → Local 28.09

AnnoMI
Paper 27.92 → Local 28.07
```

两个数据集的 Macro-F1 均与论文结果高度接近。

2. **Mixed Emotion 的主要 F1 消融趋势得到复现。**

```text
ESConv
28.09 → 27.13

AnnoMI
28.07 → 27.00
```

移除 Mixed Emotion 后两个数据集的性能均下降。

但本地下降幅度小于论文结果，且 Preference Bias 的变化方向与论文报告不同，因此不能声称该消融被数值级精确复现。

3. **SDDP 已完成指标级复现。**

```text
3 epochs: 57.29 F1
6 epochs: 60.46 F1
checkpoint reload: 60.46 F1
```

真实 spanning-tree decoder、训练、checkpoint 保存与独立加载评估均已恢复。

4. **ERC 已完成训练和推理流程恢复，但仍属于功能级重建。**

当前尚未完成 ERC 独立指标的严格论文级复现。

5. **未公开的结构消融仅作为 reconstruction。**

Graph Learning、Discourse Parser 和 Dummy Node 的精确消融实现未完整公开，因此本项目依据论文描述进行了实验性重建，不能视为官方 Table 2 的严格复现。

6. **额外诊断表明结构消融对输入信息编码方式高度敏感。**

因此 graph ablation 的本地异常结果更适合作为后续研究问题，而不是作为对原论文结论的否定。

综上，本项目当前可以定义为：

> **Main-model reproduction completed; official Mixed Emotion ablation validated at the trend level; SDDP reproduced at metric level; ERC functionally reconstructed; unreleased structural ablations audited through paper-description-based reconstruction.**

## 12. Reproduction Scope and Security

本仓库公开：

- 实验配置；
- 聚合指标；
- 必要的复现代码；
- 脱敏后的实验记录；
- 环境和实现差异说明。

本仓库不提交：

- `.pth`, `.pt`, `.bin`, `.ckpt`, `.safetensors` 模型权重；
- 原始服务器训练日志；
- 本地或服务器绝对路径；
- AutoDL 容器信息；
- API Key、Token、密码或其他凭证；
- `.env` 文件；
- 私人信息；
- cache 文件；
- 未经许可重新分发的数据集；
- 受许可限制的模型文件。

数据集、预训练模型和官方 checkpoint 请按照原项目及相应数据源许可获取。

## Citation

```bibtex
@inproceedings{wan-etal-2025-emodynamix,
    title = "EmoDynamiX: Emotional Support Dialogue Strategy Prediction by Modelling MiXed Emotions and Discourse Dynamics",
    author = "Wan, Chenwei and Labeau, Matthieu and Clavel, Chlo{\\'e}",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies",
    year = "2025",
    pages = "1678--1695"
}
```
