# MultiAgentESC Reproduction

对论文 **MultiAgentESC: A LLM-based Multi-Agent Collaboration Framework for Emotional Support Conversation**（EMNLP 2025）的复现、重构与评价记录。

本项目的目标不是追求“所有数字与论文完全一致”，而是明确回答：

1. 哪些结果可以直接由作者公开代码复现；
2. 哪些实验因实现细节缺失只能重构；
3. 哪些评价因原始评价器或人工标注条件不可获得，只能进行代理验证；
4. 当结果不一致时，差异发生在哪里，以及当前证据能够支持什么结论。

---

## 1. 证据等级

本项目统一使用以下四种标签：

| 标签 | 含义 |
| --- | --- |
| **Official reproduction** | 直接依据作者公开代码与公开配置运行 |
| **Reconstruction** | 论文报告了实验，但公开仓库未提供完整实现；依据论文与源码线索重新实现 |
| **Proxy evaluation** | 原始评价器不可获得，使用替代评价器进行验证 |
| **Not reproduced** | 缺少必要条件，不能声称已经复现 |

所有结论均按上述边界报告，不将 Reconstruction 或 Proxy evaluation 表述为作者原始实验的严格复现。

---

## 2. 论文与官方代码

**Paper**

> MultiAgentESC: A LLM-based Multi-Agent Collaboration Framework for Emotional Support Conversation  
> Yangyang Xu, Jinpeng Hu, Zhuoer Zhao, Zhangling Duan, Xiao Sun, Xun Yang  
> EMNLP 2025

- ACL Anthology: https://aclanthology.org/2025.emnlp-main.232/
- Official repository: https://github.com/MindIntLab-HFUT/MultiAgentESC

本项目固定使用的官方代码 commit：

`631b7f1961fc7502e547fd9258e847230dbcb973`

---

## 3. 正式实验环境

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA A800 80GB PCIe |
| Python | 3.10.21 |
| Backbone | Qwen2.5-32B |
| Inference | Ollama 0.5.11 |
| AutoGen | 0.2.31 |
| PyTorch | 2.4.1+cu121 |
| OpenAI SDK | 1.47.1 |
| sentence-transformers | 3.3.1 |
| Retrieval embedding | all-roberta-large-v1 |
| Dataset | ESConv |
| Seed | 42 |

本地 Ollama 模型为 `qwen2.5:32b`，对应 Q4_K_M。

论文未公开其 Qwen2.5-32B 的具体 quantization，因此 **quantization exactness 尚不能确认**。

---

## 4. Paper–Code Inconsistency

复现前首先发现了一处明确的论文描述与 released code 差异。

论文描述测试集为随机选择 100 个 conversations；而 released `main.py` 实际使用：

`dataset[:100]`

即直接使用 ESConv 前 100 个 conversations。

因此，本项目正式实验选择：

> **优先遵循 released code，同时记录 paper–code inconsistency。**

另一个重要代码细节是 generation unit。

前 100 个 conversations 中共有：

- 1544 个 raw supporter utterances；
- 但 released `main.py` 会合并连续 supporter utterances；
- 最终实际 generation positions 为 **1210**。

因此正式实验规模统一为：

**100 conversations → 1210 response-level generation cases**

---

## 5. 复现状态总览

| 实验 | 当前状态 | 证据等级 |
| --- | --- | --- |
| MultiAgentESC 核心生成流程 | 1210 / 1210 完成 | **Official reproduction** |
| Table 1 自动指标 | 与论文整体接近 | **Official generation + Proxy evaluation** |
| Figure 3 strategy distribution | 未完全恢复论文趋势 | **Negative result** |
| Zero-shot | 1210 / 1210 完成 | Published-prompt implementation |
| Few-shot CoT | 1210 / 1210 完成 | **Reconstruction** |
| Self-consistency | 1210 / 1210 完成 | **Reconstruction** |
| Self-Refine | 1210 / 1210 完成 | **Reconstruction** |
| Table 3 GPT-4o judgement | 原始 GPT-4o 环境不可恢复 | **Not strictly reproduced** |
| Table 3 Mistral Judge | 1210 / 1210 完成 | **Proxy evaluation** |
| Table 2 原始 human evaluation | 未复现 | **Not reproduced** |
| Table 2 simulated evaluation | 300 / 300 calls 完成 | **LLM-simulated evaluation** |
| Table 4 w/o Experience | 已生成并完成指标汇总 | **Reconstruction** |
| Table 4 w/o Group Discussion | 已生成并完成指标汇总 | **Reconstruction** |
| Table 4 w/o Dialogue Analysis | 1210 条输出已生成；Table 4 指标尚未汇总 | **Reconstruction / pending evaluation** |

---

## 6. Official Generation

正式实验使用：

**Qwen2.5-32B + released MultiAgentESC pipeline**

最终结果：

- 100 / 100 conversations completed
- 1210 / 1210 response-level generations completed

结果文件：

`final/results/main/multiagentesc_qwen25_32b_100.json`

这是本项目中最接近严格 official reproduction 的部分。

---

## 7. Table 1：自动指标

MultiAgentESC 官方仓库没有发布完整的 Table 1 evaluation implementation。

因此本项目使用 ESConv 官方项目中的 metric implementation 作为 proxy evaluator。

| Metric | Paper | Local | Difference |
| --- | ---: | ---: | ---: |
| D-1 | 6.78 | 6.9867 | +0.2067 |
| D-2 | 35.15 | 35.6841 | +0.5341 |
| B-1 | 17.66 | 17.2166 | -0.4434 |
| B-2 | 5.38 | 4.9609 | -0.4191 |
| B-3 | 2.35 | 2.2018 | -0.1482 |
| F1 | 18.30 | 18.0298 | -0.2702 |
| ROUGE-L | 14.66 | 14.4627 | -0.1973 |

### 结论

在 **official generation + ESConv proxy evaluator** 条件下，自动指标整体与论文较接近。

这支持“核心生成流程具有较好的指标层可复现性”，但不能表述为：

> Table 1 evaluator-level exact reproduction。

---

## 8. Figure 3：Strategy Distribution

正式生成结果的 strategy distribution 没有完全恢复论文 Figure 3 的分布。

目前可能相关的因素包括：

- 论文随机测试集与 released code `dataset[:100]` 的差异；
- Qwen2.5-32B checkpoint / quantization；
- Ollama 与底层推理行为；
- strategy normalization；
- 未公开实验细节。

现有实验无法区分这些因素，因此：

> **Figure 3 strategy distribution was not reproduced under the released-code setting; the exact cause remains unresolved.**

该结果作为 negative result 保留，不通过调参强行对齐论文。

---

## 9. Baseline Reconstruction

### 9.1 Zero-shot

论文 Prompt 较明确，因此直接实现。

结果：

**1210 / 1210**

脚本：

`final/scripts/baselines/run_zero_shot.py`

---

### 9.2 Few-shot CoT

论文未公开：

- exact few-shot examples；
- example selection rule；
- example 数量；
- 完整 reasoning implementation。

根据 released code 中 `get_cases()` 的线索，本项目构建 response-level case pool。

- 原始 case pool：267
- 排除与 test 前 100 conversations 来源重叠后：223 usable cases
- Retrieval：all-roberta-large-v1
- 每个 target：Top-3 examples

结果：

- 1210 / 1210
- unique responses = 1128
- exact example copies = 1

因此该方法统一标记为：

**Few-shot CoT reconstruction**

---

### 9.3 Self-consistency

论文未公开 path 数量和 consistency selection rule。

本项目采用：

**3 reasoning paths → strategy majority vote → semantic medoid response**

结果：

- 1210 / 1210
- unique final responses = 1125
- 358 / 1210 cases 出现多个不同 path responses

该实现属于 **Reconstruction**，不代表作者原始 Self-consistency 实现。

---

### 9.4 Self-Refine

论文未公开：

- exact feedback prompt；
- refinement prompt；
- iteration 数；
- stopping criterion。

本项目采用：

**frozen Zero-shot output → feedback → one refinement round**

并在需要时执行长度修正。

结果：

- 1210 / 1210
- unique responses = 1202
- 1210 / 1210 outputs changed
- 127 cases required length correction

该实现统一标记为：

**Self-Refine reconstruction**

---

## 10. Table 3：Mistral Proxy Judge

论文 Table 3 使用 GPT-4o，但未公开：

- exact GPT-4o snapshot；
- deployment information；
- raw judge outputs；
- 完整 position / randomization protocol。

因此本项目不声称严格复现 GPT-4o Table 3，而使用：

**Mistral-Small-24B**

作为 proxy judge。

### Position Control

前期 fixed-order / reverse-order smoke test 中观察到 position bias。

正式评价采用 balanced cyclic assignment：

- 1210 samples
- 5 methods
- 每种方法在 A / B / C / D / E 各出现 242 次

这是本项目加入的 evaluation control，并非论文公开协议。

### Proxy Result

| Method | Fluency | Identification | Comforting | Suggestion | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot | 3.788 | 3.260 | 3.156 | 2.704 | 3.157 |
| Few-shot CoT | 3.793 | 3.544 | 3.340 | 1.457 | 3.092 |
| Self-consistency | 3.595 | 3.493 | 3.235 | 1.444 | 2.989 |
| **Self-Refine** | **4.191** | **4.050** | **4.026** | **3.458** | **4.093** |
| MultiAgentESC | 4.026 | 3.430 | 3.357 | 2.551 | 3.339 |

论文 Overall：

**MultiAgentESC > Zero-shot > Self-Refine > Few-shot CoT > Self-consistency**

本地 Mistral Proxy：

**Self-Refine > MultiAgentESC > Zero-shot > Few-shot CoT > Self-consistency**

### 结论

MultiAgentESC 相对 Zero-shot、Few-shot CoT、Self-consistency 的优势部分保留，但论文中的：

**MultiAgentESC > Self-Refine**

没有在当前 proxy setting 下复现。

这不能直接说明论文结论错误，因为至少两个关键变量发生了变化：

1. Judge Model：GPT-4o → Mistral-Small-24B；
2. Self-Refine：作者实现未知 → 本地 reconstruction。

更严谨的结论是：

> **方法的主观相对排序可能对 Judge Model、Baseline implementation 和 Evaluation Protocol 较为敏感。**

---

## 11. Table 2：Human Evaluation Boundary

论文 Table 2 使用 3 名具有心理学背景的研究生标注者。

本项目没有满足该真实人工评价条件，因此：

**Original Table 2 = Not reproduced**

为了验证 pairwise evaluation protocol，本项目额外进行了：

**LLM-simulated evaluation**

设置：

- 100 response-level cases
- random seed = 42
- 3 simulated evaluator profiles
- backbone judge = Mistral-Small-24B
- 300 / 300 evaluation calls completed

三个 simulated profiles 的 unanimous rate：

**50.1%**

该数值不能解释为真实 human inter-annotator agreement。

### Overall Majority Vote

| Comparison | Win | Lose | Tie |
| --- | ---: | ---: | ---: |
| MultiAgentESC vs Zero-shot | 38 | 33 | 29 |
| MultiAgentESC vs Few-shot CoT | 16 | 77 | 7 |
| MultiAgentESC vs Self-consistency | 38 | 57 | 5 |
| MultiAgentESC vs Self-Refine | 9 | 90 | 1 |

模拟评价也明显偏好本地 Self-Refine reconstruction。

由于 Table 2 simulation 与 Table 3 proxy 共用 Mistral judge family，这只能说明：

> **同一 judge family 下观察到一致倾向。**

不能推出真实人类偏好。

---

## 12. Table 4：Ablation Reconstruction

论文 Table 4 报告三组消融：

- w/o Dialogue Analysis
- w/o Experience
- w/o Group Discussion

论文的定义为：

- **w/o Dialogue Analysis**：移除 dialogue analysis module；
- **w/o Experience**：strategy deliberation 阶段不使用 retrieved experience；
- **w/o Group Discussion**：strategy selection 与 response generation 由单一 Agent 完成，而不是多 Agent 协作。

官方仓库没有提供三组 ablation 的完整可直接运行实现，因此本项目均标记为：

**Reconstructed ablation**

### 12.1 Paper vs Local

为保证直接可比，下表仅保留论文 Table 4 同样报告的五项指标。

| Setting | Source | D-1 | D-2 | B-1 | B-2 | R-L |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MultiAgentESC | Paper | 6.78 | 35.15 | 17.66 | 5.38 | 14.66 |
| MultiAgentESC | Local baseline | 6.9867 | 35.6841 | 17.2166 | 4.9609 | 14.4627 |
| w/o Experience | Paper | 6.50 | 33.31 | 17.64 | 5.32 | 14.54 |
| w/o Experience | Local reconstruction | 6.4806 | 32.3320 | 16.7915 | 4.6398 | 13.9446 |
| w/o Group Discussion | Paper | 6.57 | 34.44 | 17.50 | 5.31 | 14.58 |
| w/o Group Discussion | Local reconstruction | 7.1034 | 35.6754 | 17.3426 | 5.0062 | 14.5285 |
| w/o Dialogue Analysis | Paper | 6.70 | 33.98 | 17.55 | 5.28 | 14.38 |
| w/o Dialogue Analysis | Local reconstruction | **尚未完成 Table 4 指标汇总** | — | — | — | — |

### 12.2 Relative Change vs Local Baseline

| Ablation | D-1 | D-2 | B-1 | B-2 | F1 | R-L | 当前判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| w/o Experience | -0.5062 | -3.3520 | -0.4251 | -0.3211 | -0.5669 | -0.5181 | 各项均下降，**退化方向与论文一致** |
| w/o Group Discussion | +0.1166 | -0.0087 | +0.1260 | +0.0453 | +0.0129 | +0.0659 | 未出现论文报告的整体退化 |
| w/o Dialogue Analysis | — | — | — | — | — | — | 1210 条输出已生成，但指标汇总仍 pending |

因此目前最严谨的 Table 4 结论是：

> **Table 4 仅得到部分一致的 reconstruction evidence。**

具体而言：

- **w/o Experience**：本地重构中所有已汇总自动指标均下降，退化方向与论文一致；但数值幅度并不等同于论文。
- **w/o Group Discussion**：本地重构没有出现论文所报告的整体性能下降，因此该趋势没有复现。
- **w/o Dialogue Analysis**：仓库中已有 1210 条正式生成结果，但当前 Table 4 CSV / summary 尚未完成其自动指标汇总，因此不能把这一行写成“已完成 Table 4 复现”。

### 12.3 Strategy Distribution Check

从当前上传的 raw outputs 直接统计 `pred_strategy`，主要分布如下：

| Setting | None | Affirmation & Reassurance | Reflection of feelings | Providing Suggestions |
| --- | ---: | ---: | ---: | ---: |
| Local baseline | 36.94% | 32.73% | 5.54% | 13.72% |
| w/o Experience | 32.98% | 20.17% | 43.72% | 0.99% |
| w/o Group Discussion | 34.38% | 32.40% | 8.10% | 12.98% |
| w/o Dialogue Analysis | 36.69% | 36.36% | 14.21% | 7.27% |

可观察到：

- w/o Experience 出现明显的 strategy concentration，尤其集中到 Reflection of feelings；
- w/o Group Discussion 的主要分布整体更接近 baseline；
- w/o Dialogue Analysis 虽然 `None` 与 Affirmation 的比例接近 baseline，但 Reflection of feelings 与 Providing Suggestions 出现较明显变化。

论文报告 w/o Dialogue Analysis 的 strategy distribution 与 MultiAgentESC 较接近；当前 reconstruction 的 raw strategy output 并不能完全恢复这一行为趋势。

因此，该部分同样应保留为 **reconstruction discrepancy**，具体原因尚不能确认。

### 12.4 Table 4 当前结论

不建议表述：

> “Table 4 已完整复现。”

建议表述：

> **三组 ablation reconstruction 均已有生成输出；其中 w/o Experience 与 w/o Group Discussion 已完成自动指标汇总，w/o Dialogue Analysis 尚待相同 evaluator 的指标汇总。当前仅 w/o Experience 的退化方向与论文一致，w/o Group Discussion 未恢复论文趋势。**

---

## 13. 主要结论

本次复现得到三个层次的结果。

### 1. Generation level

MultiAgentESC released generation pipeline 可以稳定运行：

**1210 / 1210**

### 2. Automatic metric level

在 ESConv proxy evaluator 下，Table 1 多项自动指标与论文较接近。

### 3. Behavior / subjective evaluation level

复现稳定性明显下降：

- Figure 3 strategy distribution 未完全恢复；
- reconstructed Table 4 只有部分趋势与论文一致；
- 更换 LLM Judge 后方法排序发生变化；
- Self-Refine reconstruction 在 Mistral proxy 下表现突出。

因此，本项目最重要的观察不是简单的“复现成功 / 失败”，而是：

> **自动指标接近论文，并不意味着策略行为、消融趋势和主观方法排序同样稳定。**

---

## 14. 当前复现边界

本项目没有证明：

- 所有论文数字都能够 exact reproduce；
- ESConv proxy evaluator 与作者内部 evaluator 完全一致；
- Mistral-Small-24B 与 GPT-4o 的评价等价；
- reconstructed baselines 与作者内部实现一致；
- simulated evaluators 可以替代真实心理学背景标注者；
- Table 4 三组 reconstruction 已全部完成相同指标评估；
- Figure 3 或 Table 4 差异由某一个确定因素造成。

以上问题当前均应表述为：

**尚不能确认。**

---

## 15. 后续工作

当前最直接的复现工作包括：

1. 使用与 Table 1 / Table 4 已汇总结果完全相同的 evaluator，对 `no_dialogue_analysis_100.json` 补齐自动指标；
2. 统一整理 Table 4 三组 reconstruction 的 metric、strategy distribution 和 response-level diagnostics；
3. 使用多个不同 LLM Judge 检验 Table 3 排名是否稳定；
4. 对 Self-Refine 的 prompt、refinement rounds 与 stopping rule 做敏感性实验；
5. 若条件允许，引入小规模真实人工评价，作为 LLM Judge 的外部参照。

由当前结果进一步产生的研究问题是：

> **情感支持对话方法的相对表现，在多大程度上受到 Judge Model、Baseline Implementation 与 Evaluation Protocol 的共同影响？**

---

## 16. 项目结构

    MultiAgentESC/
    ├── README.md
    ├── SOURCE.md
    ├── VALIDATION.md
    ├── data/
    │   └── ESConv.json
    ├── final/
    │   ├── scripts/
    │   │   ├── baselines/
    │   │   │   ├── run_zero_shot.py
    │   │   │   ├── run_fewshot_cot_retrieval.py
    │   │   │   ├── run_self_consistency.py
    │   │   │   └── run_self_refine.py
    │   │   └── evaluation/
    │   │       ├── run_mistral_proxy_judge_balanced_json.py
    │   │       └── run_simulated_human_eval_v2.py
    │   └── results/
    │       ├── audit/
    │       ├── main/
    │       ├── table2_simulated/
    │       ├── table3_proxy/
    │       │   └── baselines/
    │       └── table4_reconstructed/
    ├── scripts/
    └── tests/

---

## 17. AI-assisted Workflow Disclosure

本项目在代码阅读、报错定位、脚本草拟和实验文档整理过程中使用了生成式 AI 辅助。

但实验事实与结论均优先依据以下材料确认：

1. 原论文；
2. 作者公开代码；
3. 实际运行日志；
4. 正式输出文件；
5. 可重复统计结果。

AI 生成内容不直接作为实验事实或论文实现依据。对于公开材料未说明的部分，统一标记为 Reconstruction、Proxy evaluation 或尚不能确认。

---

## 18. Data and Reproducibility Note

ESConv 数据集来自：

https://github.com/thu-coai/Emotional-Support-Conversation

数据及代码的使用应遵循原项目的学术研究许可与相关条款。

正式实验输出及校验信息保存在：

`final/results/`

其中：

- `audit/FINAL_RESULTS_SHA256.txt`：关键结果文件校验记录
- `main/`：MultiAgentESC 正式 generation
- `table2_simulated/`：LLM-simulated evaluation
- `table3_proxy/`：Mistral proxy judgement 与 baseline outputs
- `table4_reconstructed/`：ablation reconstruction outputs 与分析文件

---

## Citation

```bibtex
@inproceedings{xu-etal-2025-multiagentesc,
    title = {MultiAgentESC: A LLM-based Multi-Agent Collaboration Framework for Emotional Support Conversation},
    author = {Xu, Yangyang and Hu, Jinpeng and Zhao, Zhuoer and Duan, Zhangling and Sun, Xiao and Yang, Xun},
    booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
    year = {2025},
    pages = {4665--4681},
    publisher = {Association for Computational Linguistics}
}
```
