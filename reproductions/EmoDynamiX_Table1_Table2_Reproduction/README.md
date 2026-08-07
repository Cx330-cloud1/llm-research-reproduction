# EmoDynamiX Table 1 与 Table 2 复现记录

复现日期：2026-08-05  
论文：[EmoDynamiX: Emotional Support Dialogue Strategy Prediction by Modelling MiXed Emotions and Discourse Dynamics](https://aclanthology.org/2025.naacl-long.81/)  
官方代码：[cw-wan/EmoDynamiX-v2](https://github.com/cw-wan/EmoDynamiX-v2)  
核对提交：`c9213d718a9684a5e05ce5daa947f9cbbfb7b927`

## 结论

- Table 1 中 EmoDynamiX 的两套官方 checkpoint 已在 CPU 环境完整推理；六个主指标均与论文四舍五入后的数值完全一致。
- Table 1 中其他 baseline 的数值只按论文录入，没有重新训练或调用外部商业模型，因此不能称为本次实测。
- Table 2 是消融实验。官方仓库没有发布四组消融 checkpoint，也没有发布完整的四组消融实现。仓库只暴露 `--erc_mixed` 开关，且要得到论文中的 `w/o Mixed Emotion` 数值仍需用该设置重新训练，不能在完整模型 checkpoint 上临时切换后冒充消融结果。
- 因此，本包对 Table 2 做了数值复核和可复现性审计，但没有虚构本机重训结果。

## Table 1：主实验

### 本次真实复现的 EmoDynamiX 行

| 数据集 | 指标 | 论文值 | 本次实测 | 未四舍五入实测 | 差值（实测 - 论文） |
| --- | ---: | ---: | ---: | ---: | ---: |
| ESConv | Macro-F1 | 27.70 | 27.70 | 27.7040 | 0.00 |
| ESConv | Weighted-F1 | 32.71 | 32.71 | 32.7087 | 0.00 |
| ESConv | Preference Bias | 0.45 | 0.45 | 0.4497 | 0.00 |
| AnnoMI | Macro-F1 | 27.92 | 27.92 | 27.9164 | 0.00 |
| AnnoMI | Weighted-F1 | 35.33 | 35.33 | 35.3281 | 0.00 |
| AnnoMI | Preference Bias | 0.50 | 0.50 | 0.5004 | 0.00 |

这里的 F1 以百分数显示；`result.json` 中保存的是 0 到 1 的比例。论文值只有两位小数，因此差值按相同精度计算。

### Table 1 完整论文值

完整表见 `tables/table1_paper_values.csv`。其中只有 `EmoDynamiX` 两行是本次使用官方 checkpoint 重新推理得到的；其余模型均标记为 `paper_only`。

## Table 2：消融实验

| 模型 | ESConv M-F1 | ESConv W-F1 | ESConv Bias | AnnoMI M-F1 | AnnoMI W-F1 | AnnoMI Bias | 本次状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| EmoDynamiX | 27.70 | 32.71 | 0.45 | 27.92 | 35.33 | 0.50 | 官方 checkpoint 已实测 |
| w/o Graph Learning | 25.72 | 29.31 | 0.78 | 26.95 | 29.46 | 0.73 | 实现与 checkpoint 未发布 |
| w/o Mixed Emotion | 25.90 | 29.45 | 0.66 | 24.71 | 30.25 | 0.70 | 有前向开关，但独立训练配置/checkpoint 未发布 |
| w/o Discourse Parser | 26.64 | 30.12 | 0.59 | 27.04 | 31.59 | 0.60 | 实现与 checkpoint 未发布 |
| w/o Dummy Node | 25.46 | 29.80 | 0.73 | 24.73 | 29.00 | 0.72 | 实现与 checkpoint 未发布 |

Table 2 的完整机器可读版本见 `tables/table2_paper_values.csv` 和 `tables/table2_reproducibility_audit.csv`。

## 实际运行设置

- 数据：官方仓库中预处理后的 ESConv 与 AnnoMI。
- 测试样本：ESConv 2,895 条；AnnoMI 476 条。
- checkpoint：ESConv `checkpoint-2600.pth`；AnnoMI `checkpoint-1800.pth`。
- 模型：RoBERTa-base + 3 层 RGAT；图嵌入 512；混合情绪初始温度 0.5。
- 指标：Macro-F1、Weighted-F1，以及从混淆矩阵迭代 20 次得到的 Preference Bias。
- 运行硬件：CPU。AnnoMI 约 45 秒；ESConv 约 5 分钟。
- 随机种子：官方默认值 `114514`。本次为固定 checkpoint 推理，且模型处于 eval 模式。

本地兼容环境使用 Python 3.12、PyTorch 2.2.2 CPU、Transformers 4.37.2、PyG 2.5.3、NumPy 1.26.4。作者 `requirements.txt` 中的 NumPy 1.23.5 不支持 Python 3.12，因此替换为兼容版本；六个主指标仍与论文完全一致。

## 一键核验结果

在本目录执行：

```bash
python verify_table1.py
```

脚本仅使用 Python 标准库，会从两个混淆矩阵重新计算 Macro-F1、Weighted-F1 与 Preference Bias，并检查两位小数是否与论文一致。

## 从官方仓库重新跑 Table 1

1. 克隆官方仓库。
2. 按官方 README 下载两套 EmoDynamiX checkpoint，并解压到项目根目录。
3. 安装依赖并下载 `roberta-base`。
4. 执行：

```bash
./test_roberta_hg_esconv.sh
./test_roberta_hg_annomi.sh
```

官方脚本会分别生成：

```text
roberta-hg-esconv-preprocessed-logs/result.json
roberta-hg-annomi-preprocessed-logs/result.json
```

## 重要的可复现性缺口

1. 论文附录称 AnnoMI 训练 1,200 steps，但当前官方 `train_roberta_hg_annomi.sh` 使用 1,800 steps，发布的测试 checkpoint 也是 1,800；这两处不一致需要记录。
2. ESConv 论文附录称训练 3,000 steps，训练脚本也设为 3,000，但发布的最佳测试 checkpoint 是 step 2,600。这可以理解为按验证集选择最佳 checkpoint，但 README 没有专门解释。
3. Table 1 的完整表包含 LLaMA3-70B、ChatGPT、多个微调模型和专用 ESC 系统；官方仓库没有提供所有 baseline 的统一运行脚本与输出，因此本次不能声称完整重跑了整张表。
4. Table 2 缺少三个消融变体的代码与全部四个变体的 checkpoint。仅凭论文描述自行改写，最多属于“近似再实现”，不能作为论文结果的严格复现。

## 推荐的后续顺序

1. 把本次 Table 1 结果作为“官方 checkpoint 推理复现”写入科研日志。
2. 给作者发邮件索要 Table 2 的消融分支或 checkpoint。
3. 若作者无法提供，再独立实现四个消融，并明确写成 `reimplementation`；每个设置至少运行 3 个随机种子，报告均值和标准差，不追求机械地对齐单次论文数值。

