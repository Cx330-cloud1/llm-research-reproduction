# MultiAgentESC Reproduction

对论文 **MultiAgentESC: A LLM-based Multi-Agent Collaboration Framework for Emotional Support Conversation**（EMNLP 2025）的复现、重构与评价记录。

本项目不以“把所有论文数字做成一致”为目标，而是尽量回答以下问题：

1. 哪些结果可以直接由作者公开代码复现；
2. 哪些实验因实现细节缺失，只能进行 reconstruction；
3. 哪些评价因原始 evaluator / human annotation 条件不可获得，只能进行 proxy evaluation；
4. 当结果与论文不一致时，差异发生在哪一层，现有证据能够支持什么结论。

> **核心结论：** MultiAgentESC 的 released generation pipeline 可以稳定运行，自动指标在统一 proxy evaluator 下整体接近论文；但 strategy behavior、ablation trend 与 subjective ranking 的稳定性明显更弱。

---

## 1. Reproduction Labels

本项目统一使用以下四类标签，避免把不同证据等级混为一谈。

| 标签 | 含义 |
| --- | --- |
| **Official reproduction** | 直接依据作者公开代码与公开配置运行 |
| **Reconstruction** | 论文报告了实验，但公开仓库未提供完整实现；依据论文与源码线索重新实现 |
| **Proxy evaluation** | 原始评价器不可获得，使用替代评价器进行验证 |
| **Not reproduced** | 缺少必要条件，不能声称已经复现 |

所有实验结论均按照上述边界报告，不将 reconstruction 或 proxy evaluation 表述为作者原始实验的严格复现。

---

## 2. Paper and Upstream Code

**Paper**

> MultiAgentESC: A LLM-based Multi-Agent Collaboration Framework for Emotional Support Conversation  
> EMNLP 2025

- ACL Anthology: https://aclanthology.org/2025.emnlp-main.232/
- Official repository: https://github.com/MindIntLab-HFUT/MultiAgentESC

本项目固定使用的官方代码 commit：

`631b7f1961fc7502e547fd9258e847230dbcb973`

主要官方文件 SHA：

| File | SHA |
| --- | --- |
| `main.py` | `f7a1bfab462b0323b53a27c2891404fe19050598` |
| `multiagent.py` | `5135f59dd76659e8939fb3029493abbd281cac30` |
| `prompt.py` | `074af872d6b9c9aceb0f33900fe2e7e389a882cc` |
| `strategy.py` | `b7a417a39e83e69738772309c35fc2572b86b8c2` |

---

## 3. Formal Experimental Environment

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

官方 requirements：

- `openai==1.47.1`
- `autogen==0.2.31`
- `sentence-transformers==3.3.1`
- `tokenizers==0.19.1`
- `torch==2.4.1`
- `tqdm==4.66.4`

本地 Ollama 使用：

`qwen2.5:32b / Q4_K_M`

论文未公开其 Qwen2.5-32B 的具体 quantization，因此：

> **quantization exactness 尚不能确认。**

---

## 4. Paper–Code Inconsistency

复现前首先确认了一处明确的论文描述与 released code 差异。

论文描述测试集为：

> 随机选择 100 个 conversations。

而 released `main.py` 实际使用：

`dataset[:100]`

即直接使用 ESConv 前 100 个 conversations。

因此，本项目正式实验选择：

> **优先遵循 released code，同时记录 paper–code inconsistency。**

另一个重要代码细节是 generation unit。

前 100 个 conversations 中：

- raw supporter utterances = **1544**
- released `main.py` 会合并连续 supporter utterances
- official response-level generation positions = **1210**

因此正式实验规模统一为：

**100 conversations → 1210 response-level generation cases**

---

## 5. Reproduction Status

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
| Table 4 三组 ablation | 生成与自动指标汇总均完成 | **Reconstruction** |

---

## 6. Official Generation

正式实验使用：

**Qwen2.5-32B + released MultiAgentESC pipeline**

最终结果：

- 100 / 100 conversations completed
- 1210 / 1210 response-level generations completed

正式结果文件：

`final/results/main/multiagentesc_qwen25_32b_100.json`

该文件字段包括：

- `strategy`
- `reference`
- `context`
- `response`
- `pred_strategy`

这是本项目中最接近严格 official reproduction 的部分。

---

## 7. Formal Evaluator for Table 1 / Table 4

MultiAgentESC 官方仓库没有发布完整的 Table 1 / Table 4 evaluation implementation。

本项目正式结果使用 ESConv 官方项目中的 metric implementation 作为 proxy evaluator：

- Upstream: `thu-coai/Emotional-Support-Conversation`
- Commit: `f262d062`
- File: `codes_zcj/metric/myMetrics.py`

该 evaluator 计算：

- Distinct-1 / Distinct-2 / Distinct-3
- BLEU-1 / BLEU-2 / BLEU-3 / BLEU-4
- unigram F1
- ROUGE-L

### Important

仓库中早期的：

`scripts/metrics.py`

属于本地早期实现，tokenization 与 Distinct 计算细节和 ESConv 官方 `myMetrics.py` 不完全相同。

因此：

> **README 中 Table 1 / Table 4 的正式数字均以 ESConv 官方 proxy evaluator 为准，不使用 legacy `scripts/metrics.py` 重新计算。**

---

## 8. Table 1: Automatic Metrics

| Metric | Paper | Local | Difference |
| --- | ---: | ---: | ---: |
| D-1 | 6.78 | 6.9867 | +0.2067 |
| D-2 | 35.15 | 35.6841 | +0.5341 |
| B-1 | 17.66 | 17.2166 | -0.4434 |
| B-2 | 5.38 | 4.9609 | -0.4191 |
| B-3 | 2.35 | 2.2018 | -0.1482 |
| F1 | 18.30 | 18.0298 | -0.2702 |
| ROUGE-L | 14.66 | 14.4627 | -0.1973 |

### Conclusion

在：

**official generation + ESConv proxy evaluator**

条件下，自动指标整体与论文较接近。

可以支持：

> 核心生成流程在自动指标层具有较好的可复现性。

但不能表述为：

> Table 1 evaluator-level exact reproduction。

---

## 9. Figure 3: Strategy Distribution

正式 generation output 的 strategy distribution 没有完全恢复论文 Figure 3 的趋势。

当前可能相关的因素包括：

- paper 中的随机 test split 与 released code `dataset[:100]` 的差异；
- Qwen2.5-32B checkpoint / quantization；
- Ollama / inference behavior；
- strategy normalization；
- 论文未公开的实验细节。

现有实验无法区分这些因素，因此：

> **Figure 3 strategy distribution was not reproduced under the released-code setting; the exact cause remains unresolved.**

该结果作为 negative result 保留，不通过调参强行对齐论文。

---

## 10. Baseline Reconstruction

### 10.1 Zero-shot

论文中的 Zero-shot Prompt 相对明确，因此直接实现。

结果：

**1210 / 1210**

脚本：

`final/scripts/baselines/run_zero_shot.py`

---

### 10.2 Few-shot CoT

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

脚本：

`final/scripts/baselines/run_fewshot_cot_retrieval.py`

---

### 10.3 Self-consistency

论文未公开：

- reasoning path 数量；
- sampling protocol；
- consistency selection rule。

本项目采用：

**3 reasoning paths → strategy majority vote → semantic medoid response**

结果：

- 1210 / 1210
- unique final responses = 1125
- 358 / 1210 cases 出现多个不同 path responses

该实现属于 **Reconstruction**，不代表作者原始 Self-consistency implementation。

脚本：

`final/scripts/baselines/run_self_consistency.py`

---

### 10.4 Self-Refine

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

脚本：

`final/scripts/baselines/run_self_refine.py`

---

## 11. Table 3: Mistral Proxy Judge

论文 Table 3 使用 GPT-4o，但未公开：

- exact GPT-4o snapshot；
- deployment information；
- raw judge outputs；
- exact sample protocol；
- 完整 position / randomization protocol。

因此本项目不声称严格复现 GPT-4o Table 3，而使用：

**Mistral-Small-24B**

作为 proxy judge。

### 11.1 Position Control

前期 fixed-order / reverse-order smoke test 中观察到 position bias。

正式评价采用 balanced cyclic assignment：

- 1210 samples
- 5 methods
- 每种方法在 A / B / C / D / E 各出现 242 次

这是本项目加入的 evaluation control，并非论文公开协议。

### 11.2 Proxy Result

| Method | Fluency | Identification | Comforting | Suggestion | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot | 3.788 | 3.260 | 3.156 | 2.704 | 3.157 |
| Few-shot CoT | 3.793 | 3.544 | 3.340 | 1.457 | 3.092 |
| Self-consistency | 3.595 | 3.493 | 3.235 | 1.444 | 2.989 |
| **Self-Refine** | **4.191** | **4.050** | **4.026** | **3.458** | **4.093** |
| MultiAgentESC | 4.026 | 3.430 | 3.357 | 2.551 | 3.339 |

论文 Overall 排名：

**MultiAgentESC > Zero-shot > Self-Refine > Few-shot CoT > Self-consistency**

本地 Mistral Proxy 排名：

**Self-Refine > MultiAgentESC > Zero-shot > Few-shot CoT > Self-consistency**

### 11.3 Interpretation

MultiAgentESC 相对：

- Zero-shot
- Few-shot CoT
- Self-consistency

的优势部分保留。

但论文中的：

**MultiAgentESC > Self-Refine**

没有在当前 proxy setting 下复现。

这不能直接说明论文结论错误，因为至少两个关键变量发生了变化：

1. Judge Model：GPT-4o → Mistral-Small-24B；
2. Self-Refine：作者实现未知 → 本地 reconstruction。

更严谨的结论是：

> **方法的主观相对排序可能对 Judge Model、Baseline implementation 和 Evaluation Protocol 较为敏感。**

---

## 12. Table 2: Human Evaluation Boundary

论文 Table 2 使用：

**3 名具有心理学背景的研究生标注者**

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

### 12.1 Overall Majority Vote

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

## 13. Table 4: Ablation Reconstruction

论文 Table 4 报告三组消融：

- w/o Dialogue Analysis
- w/o Experience
- w/o Group Discussion

官方仓库没有提供三组 ablation 的完整可直接运行实现，因此本项目均标记为：

**Reconstructed ablation**

三组实验均已完成：

- 1210 条正式 generation；
- 与 baseline 相同的 ESConv proxy evaluator；
- 自动指标汇总。

---

### 13.1 Paper vs Local

为保证可比，下表仅列论文 Table 4 同样报告的 5 项指标。

| Setting | Source | D-1 | D-2 | B-1 | B-2 | R-L |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MultiAgentESC | Paper | 6.78 | 35.15 | 17.66 | 5.38 | 14.66 |
| MultiAgentESC | Local baseline | 6.9867 | 35.6841 | 17.2166 | 4.9609 | 14.4627 |
| w/o Dialogue Analysis | Paper | 6.70 | 33.98 | 17.55 | 5.28 | 14.38 |
| w/o Dialogue Analysis | Local reconstruction | **6.8090** | **34.5238** | **17.3521** | **4.9053** | **14.2709** |
| w/o Experience | Paper | 6.50 | 33.31 | 17.64 | 5.32 | 14.54 |
| w/o Experience | Local reconstruction | **6.4806** | **32.3320** | **16.7915** | **4.6398** | **13.9446** |
| w/o Group Discussion | Paper | 6.57 | 34.44 | 17.50 | 5.31 | 14.58 |
| w/o Group Discussion | Local reconstruction | **7.1034** | **35.6754** | **17.3426** | **5.0062** | **14.5285** |

---

### 13.2 Relative Change vs Local Baseline

Local baseline：

| Metric | Score |
| --- | ---: |
| D-1 | 6.9867 |
| D-2 | 35.6841 |
| B-1 | 17.2166 |
| B-2 | 4.9609 |
| B-3 | 2.2018 |
| F1 | 18.0298 |
| R-L | 14.4627 |

三组 ablation 相对 local baseline 的变化：

| Ablation | ΔD-1 | ΔD-2 | ΔB-1 | ΔB-2 | ΔB-3 | ΔF1 | ΔR-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| w/o Dialogue Analysis | -0.1777 | -1.1603 | +0.1355 | -0.0556 | -0.0655 | -0.0280 | -0.1918 |
| w/o Experience | -0.5062 | -3.3520 | -0.4251 | -0.3211 | — | -0.5669 | -0.5181 |
| w/o Group Discussion | +0.1166 | -0.0087 | +0.1260 | +0.0453 | — | +0.0129 | +0.0659 |

> 注：现有正式汇总文件未记录 w/o Experience / w/o Group Discussion 的 B-3 delta，因此此处不补写未验证值。

---

### 13.3 Interpretation

#### w/o Experience

本地 reconstruction 中：

- D-1 ↓
- D-2 ↓
- B-1 ↓
- B-2 ↓
- F1 ↓
- R-L ↓

即所有已汇总自动指标均下降。

因此可以严谨地说：

> **w/o Experience 的性能退化方向与论文一致。**

但不能说：

> 数值与论文完全复现。

因为退化幅度明显不同。

---

#### w/o Dialogue Analysis

本地 reconstruction 中：

- D-1 ↓
- D-2 ↓
- B-2 ↓
- B-3 ↓
- F1 ↓
- R-L ↓
- B-1 小幅 ↑

在论文 Table 4 同样报告的 5 项指标中：

- 4 项方向与论文一致；
- 1 项（B-1）方向不一致。

尤其是 D-2：

- Paper delta: -1.17
- Local delta: -1.1603

幅度非常接近。

因此最合适的表述是：

> **w/o Dialogue Analysis 在多数指标上恢复了论文的下降方向，但并非所有指标一致，因此属于 partial trend consistency，而不是严格复现。**

---

#### w/o Group Discussion

本地 reconstruction 中：

- D-2 仅下降 -0.0087
- D-1、B-1、B-2、F1、R-L 均略有上升

因此：

> **w/o Group Discussion 未恢复论文报告的整体性能退化趋势。**

当前 evidence 不支持在该 reconstruction setting 下确认 Group Discussion 的边际贡献。

可能原因包括：

- 未公开 prompt / agent interaction details；
- reconstructed removal 与作者内部实现不完全一致；
- model / inference configuration 差异；
- 其他未公开实验细节。

以上均为 hypothesis，当前尚不能确认具体原因。

---

### 13.4 Strategy Distribution

从正式 raw outputs 的 `pred_strategy` 直接统计主要策略：

| Setting | None | Affirmation & Reassurance | Reflection of feelings | Providing Suggestions |
| --- | ---: | ---: | ---: | ---: |
| Local baseline | 36.94% | 32.73% | 5.54% | 13.72% |
| w/o Dialogue Analysis | 36.69% | 36.36% | 14.21% | 7.27% |
| w/o Experience | 32.98% | 20.17% | 43.72% | 0.99% |
| w/o Group Discussion | 34.38% | 32.40% | 8.10% | 12.98% |

可观察到：

- **w/o Experience** 出现明显 strategy concentration，Reflection of feelings 大幅增加，而 Providing Suggestions 显著减少；
- **w/o Group Discussion** 的主要策略分布整体更接近 baseline；
- **w/o Dialogue Analysis** 的 None / Affirmation 比例接近 baseline，但 Reflection of feelings 上升、Providing Suggestions 下降。

因此行为层面的趋势也没有完全恢复论文 Figure 3。

---

### 13.5 Table 4 Conclusion

当前最严谨的总结是：

> **三组 reconstructed ablations 均已完成相同 evaluator 下的生成与指标汇总。w/o Experience 的退化方向与论文一致；w/o Dialogue Analysis 在多数指标上呈现相同下降方向，但并非全部一致；w/o Group Discussion 未恢复论文报告的整体退化趋势。**

因此 Table 4 应标记为：

**Reconstructed ablation — evaluation complete**

而不是：

**Official Table 4 reproduced**

---

## 14. Audit Notes

正式 generation 中存在少量字符串 `"None"` response：

| Setting | `"None"` responses |
| --- | ---: |
| MultiAgentESC baseline | 4 |
| w/o Dialogue Analysis | 2 |
| w/o Experience | 13 |
| w/o Group Discussion | 1 |

为保持正式评价口径一致，这些记录没有被静默删除。

关键结果校验信息保存在：

`final/results/audit/FINAL_RESULTS_SHA256.txt`

---

## 15. Main Findings

本次复现得到三个层次的结果。

### Level 1 — Generation

MultiAgentESC released generation pipeline 可以稳定运行：

**1210 / 1210**

### Level 2 — Automatic Metrics

在 ESConv proxy evaluator 下，Table 1 多项自动指标与论文较接近。

### Level 3 — Behavior / Subjective Evaluation

复现稳定性明显下降：

- Figure 3 strategy distribution 未完全恢复；
- reconstructed Table 4 仅部分趋势与论文一致；
- 更换 LLM Judge 后方法排序发生变化；
- Self-Refine reconstruction 在 Mistral proxy 下表现突出。

因此，本项目最重要的观察不是简单的“复现成功 / 失败”，而是：

> **自动指标接近论文，并不意味着策略行为、消融趋势和主观方法排序同样稳定。**

---

## 16. Reproduction Boundary

本项目没有证明：

- 所有论文数字都能够 exact reproduce；
- ESConv proxy evaluator 与作者内部 evaluator 完全一致；
- Mistral-Small-24B 与 GPT-4o 的评价等价；
- reconstructed baselines 与作者内部 implementation 一致；
- simulated evaluators 可以替代真实心理学背景标注者；
- Figure 3 或 Table 4 差异由某一个确定因素造成；
- w/o Group Discussion 在所有实现条件下都没有贡献。

以上问题当前均应表述为：

**尚不能确认。**

---

## 17. Project Structure

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
    │       │   └── FINAL_RESULTS_SHA256.txt
    │       ├── main/
    │       │   └── multiagentesc_qwen25_32b_100.json
    │       ├── table2_simulated/
    │       │   └── simulated_human_eval_v2_100.jsonl
    │       ├── table3_proxy/
    │       │   ├── mistral_proxy_judge_balanced_1210.jsonl
    │       │   └── baselines/
    │       │       ├── zero_shot_1210.jsonl
    │       │       ├── fewshot_cot_retrieval_1210.jsonl
    │       │       ├── self_consistency_1210.jsonl
    │       │       └── self_refine_1210.jsonl
    │       └── table4_reconstructed/
    │           ├── TABLE4_ANALYSIS.md
    │           ├── TABLE4_RECONSTRUCTION.csv
    │           ├── TABLE4_RESULT.md
    │           ├── no_dialogue_analysis_100.json
    │           ├── no_experience_100.json
    │           └── no_group_discussion_100.json
    ├── scripts/
    └── tests/

其中：

- `final/scripts/`：正式 baseline / evaluation scripts；
- `final/results/`：正式实验输出；
- 原有 `scripts/`：早期本地 smoke / engineering implementation，不应与正式 evaluator 口径混淆。

---

## 18. AI-assisted Workflow Disclosure

本项目在以下环节使用了生成式 AI 辅助：

- 代码阅读；
- 报错定位；
- 实验脚本草拟；
- 实验记录整理；
- 文档结构整理。

但实验事实与结论均优先依据以下材料确认：

1. 原论文；
2. 作者公开代码；
3. 固定 commit；
4. 实际运行日志；
5. 正式输出文件；
6. 可重复统计结果。

AI 生成内容不直接作为实验事实或论文实现依据。

对于公开材料未说明的部分，统一标记为：

- Reconstruction
- Proxy evaluation
- Not reproduced
- 尚不能确认

---

## 19. Current Research Questions

本次复现进一步产生了几个值得验证的问题：

1. 不同 LLM Judge 是否会显著改变 ESC 方法排序？
2. Self-Refine 的表现是否高度依赖 feedback prompt、refinement rounds 与 stopping rule？
3. MultiAgentESC 的 strategy distribution 是否对 test split、model version 与 quantization 敏感？
4. candidate order / presentation format 是否会影响 LLM-as-a-Judge？
5. BLEU、ROUGE、Distinct 等自动指标与真实 emotional-support preference 的相关性有多高？
6. multi-agent collaboration 的增益是否对 backbone capability 与 interaction protocol 敏感？

---

## 20. Next Step

下一阶段不再以“继续追论文数字”为主要目标，而优先验证：

**Judge Model × Baseline Implementation × Evaluation Protocol**

是否会系统性改变情感支持对话方法的相对结论。

进一步希望从：

**How to respond?**

推进到：

**Whether / when / how proactively to respond?**

关注：

- proactive agents
- mixed-initiative interaction
- intervention timing
- response autonomy
- silence as an explicit action

---

## 21. Data and Reproducibility Note

ESConv 数据集来自：

https://github.com/thu-coai/Emotional-Support-Conversation

数据、代码及派生结果的使用应遵循原项目的许可与学术研究条款。

本仓库保留正式实验输出与校验信息，用于实验审计与结果复查。

---

## Citation

```bibtex
@inproceedings{xu-etal-2025-multiagentesc,
  title     = {MultiAgentESC: A LLM-based Multi-Agent Collaboration Framework for Emotional Support Conversation},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year      = {2025},
  publisher = {Association for Computational Linguistics}
}
```
