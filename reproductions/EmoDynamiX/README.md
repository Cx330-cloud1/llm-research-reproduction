# EmoDynamiX Reproduction

本目录记录论文 **EmoDynamiX: Emotional Support Dialogue Strategy Prediction by Modelling MiXed Emotions and Discourse Dynamics**（NAACL 2025）的复现、环境校准、消融实验与缺失模块恢复过程。

- Paper: https://aclanthology.org/2025.naacl-long.81/
- Official Repository: https://github.com/cw-wan/EmoDynamiX-v2
- 原始项目说明：[UPSTREAM_README.md](UPSTREAM_README.md)

> 本项目用于科研复现与可复现性分析。原论文、代码及相关资源版权归原作者所有。

---

## 1. Reproduction Status

| Component | Status | Notes |
|---|---|---|
| ESConv Full Model | Completed | Environment-calibrated |
| AnnoMI Full Model | Completed | Environment-calibrated |
| w/o Mixed Emotion | Completed | Official code-supported ablation |
| w/o Graph Learning | Reconstructed | Paper-description-based |
| w/o Discourse Parser | Reconstructed | Paper-description-based |
| w/o Dummy Node | Reconstructed | Paper-description-based |
| ERC Module | Functional Reconstruction | Missing checkpoint reconstructed |
| SDDP Module | Functional Reconstruction | Functional recovery, metric-level reproduction incomplete |
| Flat Context Diagnostic | Completed | Additional diagnostic experiment |

---

## 2. Environment-Calibrated Setup

第一轮实验与论文结果已经较为接近，但进一步检查发现本地环境与官方依赖存在版本差异。

因此重新建立统一校准环境，对主模型与官方可执行的 Mixed Emotion 消融进行验证。

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

训练参数保持与论文 / 官方配置一致的主要设置：

```text
Learning rate: 4e-6
Batch size: 16
Weight decay: 1e-3
ERC temperature: 0.5
Graph hidden dimension: 512
```

训练步数：

```text
ESConv: 3000
AnnoMI: 1200
```

---

## 3. Main Results

### Environment-Calibrated Full Model

| Dataset | Paper M-F1 | Initial Reproduction | Calibrated Reproduction | Difference |
|---|---:|---:|---:|---:|
| ESConv | 27.70 | 26.63 | **28.09** | +0.39 |
| AnnoMI | 27.92 | 26.64 | **28.07** | +0.15 |

更完整的结果：

| Dataset | Paper M-F1 | Local M-F1 | Paper W-F1 | Local W-F1 | Paper Bias | Local Bias |
|---|---:|---:|---:|---:|---:|---:|
| ESConv | 27.70 | **28.09** | 32.71 | **32.04** | 0.45 | **0.4256** |
| AnnoMI | 27.92 | **28.07** | 35.33 | **34.38** | 0.50 | **0.4793** |

### Conclusion

环境校准后：

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

两个数据集的 Macro-F1 均与论文结果非常接近。

因此，本项目认为：

> The main EmoDynamiX results are successfully reproduced under the environment-calibrated setup.

---

## 4. Mixed Emotion Ablation

`erc_mixed` 是官方模型代码中直接提供的消融控制参数，因此该实验属于本项目中可信度最高的消融复现。

### Environment-Calibrated Results

| Dataset | Full M-F1 | w/o Mixed M-F1 | Change |
|---|---:|---:|---:|
| ESConv | **28.09** | **27.13** | **-0.96** |
| AnnoMI | **28.07** | **27.00** | **-1.07** |

Weighted-F1：

| Dataset | Full W-F1 | w/o Mixed W-F1 | Change |
|---|---:|---:|---:|
| ESConv | 32.04 | 31.55 | -0.49 |
| AnnoMI | 34.38 | 33.49 | -0.89 |

两个数据集在关闭 Mixed Emotion representation 后均出现 F1 下降。

因此，论文关于 Mixed Emotion 对策略预测具有正向贡献的主要趋势得到复现。

需要注意：

- 本地消融幅度小于论文报告值；
- Preference Bias 的变化方向未完全复现论文结果。

因此该部分应表述为：

> The main F1 degradation trend of the Mixed Emotion ablation is reproduced, while the exact degradation magnitude and Preference Bias behavior differ from the paper.

---

## 5. Structural Ablations

论文还包含以下三个结构消融：

- w/o Graph Learning
- w/o Discourse Parser
- w/o Dummy Node

但当前官方仓库未提供这三项实验对应的完整实现或直接控制参数。

因此，本项目依据论文方法描述进行了重建。

这些实验统一标记为：

> **Paper-description-based reconstruction**

而不是官方精确复现。

### ESConv

| Variant | Paper M-F1 | Local M-F1 |
|---|---:|---:|
| Full | 27.70 | 26.63 |
| w/o Graph Learning | 25.72 | 27.97 |
| w/o Mixed Emotion | 25.90 | 26.34 |
| w/o Discourse Parser | 26.64 | 26.11 |
| w/o Dummy Node | 25.46 | 26.51 |

### AnnoMI

| Variant | Paper M-F1 | Local M-F1 |
|---|---:|---:|
| Full | 27.92 | 26.64 |
| w/o Graph Learning | 26.95 | 28.89 |
| w/o Mixed Emotion | 24.71 | 25.36 |
| w/o Discourse Parser | 27.04 | 28.02 |
| w/o Dummy Node | 24.73 | 28.29 |

其中：

- ESConv `w/o Discourse Parser` 出现下降；
- 其余部分结构消融没有稳定复现论文报告的下降趋势；
- `w/o Graph Learning` 在两个数据集上均出现较强结果。

由于缺少作者原始结构消融代码，这些结果不能用于声称论文对应消融结论被精确复现。

---

## 6. Flat Context Diagnostic

为了进一步分析 reconstructed `w/o Graph Learning` 的异常结果，本项目额外进行了 Flat Context 诊断实验。

### ESConv

| Variant | Macro-F1 | Weighted-F1 | Bias |
|---|---:|---:|---:|
| Flat Context | 26.35 | 30.05 | 0.4275 |
| Full Model | 26.63 | 31.12 | 0.4352 |
| w/o Graph + Emotion / Strategy Tags | 27.97 | 31.73 | 0.3603 |

结果表现为：

```text
Flat Context
26.35

+ explicit emotion / strategy information

w/o Graph
27.97
```

普通 Flattened Context 本身并没有表现出异常强的结果。

较高的 no-graph performance 与显式注入 emotion / historical strategy 信息的文本表示有关。

由于论文没有公开该消融中 tag 的精确编码方式，因此无法进一步确认本地实现是否与作者内部实现完全一致。

---

## 7. Missing Module Reconstruction

### ERC

官方项目依赖：

```text
pre_trained_models/sequential_erc_model.pth
```

但原仓库中未直接提供对应本地权重。

本项目完成：

- SequentialERC 训练流程恢复；
- DailyDialog emotion 数据训练；
- checkpoint 生成；
- 模型加载验证；
- forward 接口验证。

状态：

> **Functional Reconstruction**

需要注意：当前主模型 calibrated experiments 使用官方预处理数据中的 ERC 信息，因此不能声称这些主模型结果由本地重新训练的 ERC 端到端生成。

---

### SDDP

EmoDynamiX 使用 Structured Dialogue Discourse Parser 构建 discourse structure。

本项目完成：

- SDDP 子项目恢复；
- 依赖适配；
- checkpoint 训练流程恢复；
- EmoDynamiX parser 接口接入；
- Full-mode 功能调用恢复。

由于原依赖兼容性和 discourse parsing 训练问题，本地独立 SDDP 指标尚未达到论文报告水平。

因此目前状态为：

> **Functional Reconstruction, not Metric-Level Reproduction**

同样，主模型 calibrated experiments 使用官方预处理的 `parsed_dialogue`，并不是本地重训 SDDP 重新生成的 discourse graph。

---

## 8. Result Organization

```text
results/
├── calibrated/
│   ├── esconv_full.txt
│   ├── esconv_wo_mixed.txt
│   ├── annomi_full_paper1200.txt
│   ├── annomi_wo_mixed.txt
│   └── summary.md
│
├── table1/
│   ├── esconv_full.txt
│   └── annomi_full.txt
│
├── table2/
│   ├── summary.md
│   ├── esconv_wo_mixed.txt
│   ├── annomi_wo_mixed.txt
│   ├── esconv_no_discourse.txt
│   ├── annomi_no_discourse_preliminary.txt
│   ├── esconv_no_dummy.txt
│   ├── annomi_no_dummy_preliminary.txt
│   ├── esconv_no_graph_preliminary.txt
│   └── annomi_no_graph_preliminary.txt
│
├── diagnostics/
│   └── esconv_flat_context.txt
│
├── paper_reference/
└── legacy/
```

说明：

- `calibrated/`：最终优先采用的正式复现结果；
- `table1/`：第一轮主模型复现记录；
- `table2/`：第一轮消融实验；
- `diagnostics/`：额外诊断实验；
- `paper_reference/`：论文原始指标记录；
- `legacy/`：早期实验与审计文件。

---

## 9. Reproduction Reliability

| Experiment | Reproduction Type | Reliability |
|---|---|---|
| ESConv Full | Environment-calibrated official pipeline | High |
| AnnoMI Full | Environment-calibrated official pipeline | High |
| w/o Mixed Emotion | Official code-supported ablation | High |
| ERC | Functional reconstruction | Medium |
| SDDP | Functional reconstruction | Medium |
| w/o Discourse Parser | Paper-description-based reconstruction | Limited |
| w/o Dummy Node | Paper-description-based reconstruction | Limited |
| w/o Graph Learning | Paper-description-based reconstruction | Limited |
| Flat Context | Diagnostic experiment | High for local analysis |

---

## 10. Final Conclusion

本次 EmoDynamiX 复现得到以下主要结论：

1. **主模型成功复现。**

   环境校准后：

   ```text
   ESConv: Paper 27.70 → Local 28.09
   AnnoMI: Paper 27.92 → Local 28.07
   ```

   两个数据集的 Macro-F1 均与论文结果高度接近。

2. **Mixed Emotion 的主要消融趋势得到复现。**

   ```text
   ESConv: 28.09 → 27.13
   AnnoMI: 28.07 → 27.00
   ```

   去除 mixed-emotion representation 后两个数据集的 F1 均下降。

3. **结构消融无法视为官方精确复现。**

   Graph Learning、Discourse Parser 与 Dummy Node 的消融实现未在官方仓库中完整公开，因此本项目只能依据论文描述重建。

4. **ERC 与 SDDP 的缺失部分完成了功能级恢复。**

   其中 SDDP 尚未完成论文指标级复现。

5. **额外诊断实验揭示了结构消融对实现细节的敏感性。**

综上，本项目当前可以定义为：

> **Main-model reproduction completed; official executable ablation validated; missing modules functionally reconstructed; unreleased structural ablations audited through paper-description-based reconstruction.**

---

## 11. Reproduction Scope

本仓库不提交：

- Large model checkpoints
- `.pth`, `.pt`, `.bin` model weights
- Private credentials or API keys
- Unnecessary cache files
- Restricted datasets

数据集、预训练模型与官方 checkpoint 请按照原项目说明与相应许可获取。

---

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
