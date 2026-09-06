# MultiAgentESC 论文复现

本项目复现论文：

**MultiAgentESC: A LLM-based Multi-Agent Collaboration Framework for Emotional Support Conversation**

会议：**EMNLP 2025**

本项目围绕 MultiAgentESC 的官方生成流程、自动评价、Baseline、LLM Judge、人工评价流程与消融实验进行复现与重构。

由于论文并未公开全部 Baseline、评价器和消融实现，本项目对实验结果进行分级记录：

- **Official reproduction**：依据作者公开代码直接运行
- **Proxy evaluation**：原始评价器不可用，使用替代评价器
- **Reconstruction**：依据论文描述和源码线索重新实现
- **Not reproduced**：缺少必要条件，无法进行严格复现

---

## 1. 实验环境

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA A800 80GB PCIe |
| Python | 3.10.21 |
| 推理框架 | Ollama 0.5.11 |
| 主模型 | Qwen2.5-32B |
| Embedding | all-roberta-large-v1 |
| Proxy Judge | Mistral-Small-24B |
| 数据集 | ESConv |
| Seed | 42 |

官方代码固定版本：

`631b7f1961fc7502e547fd9258e847230dbcb973`

主要依赖：

- openai==1.47.1
- autogen==0.2.31
- sentence-transformers==3.3.1
- tokenizers==0.19.1
- torch==2.4.1
- tqdm==4.66.4

论文未公开 Qwen2.5-32B 的具体 Ollama quantization。

本地正式实验使用：

`qwen2.5:32b / Q4_K_M`

因此无法确认量化版本与论文实验完全一致。

---

## 2. 复现状态

| 实验 | 状态 | 类型 |
| --- | --- | --- |
| MultiAgentESC 官方生成 | 完成 | Official reproduction |
| Table 1 自动指标 | 接近论文 | Proxy evaluator |
| Figure 3 策略分布 | 未完全复现 | Official output analysis |
| Zero-shot | 完成 | Published-prompt implementation |
| Few-shot CoT | 完成 | Reconstruction |
| Self-consistency | 完成 | Reconstruction |
| Self-Refine | 完成 | Reconstruction |
| Table 3 | 完成 | Mistral proxy evaluation |
| Table 2 真人评价 | 未严格复现 | Not reproduced |
| Table 2 模拟评价 | 完成 | LLM-simulated evaluation |
| Table 4 消融 | 完成 | Reconstruction |

---

## 3. 官方生成流程

官方 `main.py` 实际使用：

`samples = dataset[:100]`

论文文本描述为随机选择 100 个 conversations，但 released code 使用数据集前 100 个 conversations。

本项目正式实验优先遵循 released code。

前 100 个 ESConv conversations 中共有：

- 1544 个 supporter utterances
- 1210 个官方实际生成位置

这是因为官方代码会将连续的 supporter utterances 合并为一个 response-level generation position。

最终正式生成结果：

**1210 / 1210 完成**

结果文件：

`final/results/main/multiagentesc_qwen25_32b_100.json`

---

## 4. Table 1 自动指标

MultiAgentESC 官方仓库没有公开论文 Table 1 的完整评价实现。

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

整体结果与论文较接近。

因此 Table 1 是本次复现中最稳定的部分。

需要注意：

该结果属于 **官方生成流程 + ESConv proxy evaluator**，不能表述为评价器层面的完全一致复现。

---

## 5. Figure 3 策略分布

正式生成结果中的策略分布没有完全恢复论文 Figure 3。

目前可能影响结果的因素包括：

- 论文随机测试集与 released code `dataset[:100]` 的差异
- Qwen2.5-32B checkpoint 或 quantization 差异
- Ollama 版本差异
- strategy normalization
- 论文未公开的实验细节

目前尚不能确认具体原因。

因此 Figure 3 记录为：

**未完全复现**

---

## 6. Baseline Reconstruction

### Zero-shot

论文公开了 Zero-shot Prompt，因此该方法可以较直接实现。

正式生成：

**1210 / 1210**

脚本：

`final/scripts/baselines/run_zero_shot.py`

---

### Few-shot CoT

论文没有公开：

- 实际 few-shot examples
- example reasoning
- example 数量
- example selection rule

官方 `get_cases()` 使用 `samples[400:420]` 构建 case pool。

本地基于该线索采用：

`official case pool → all-roberta-large-v1 retrieval → top-3 examples → Few-shot CoT generation`

正式结果：

- 1210 / 1210
- unique responses = 1128
- exact example copies = 1

该方法标记为：

**Few-shot CoT reconstruction**

脚本：

`final/scripts/baselines/run_fewshot_cot_retrieval.py`

---

### Self-consistency

论文没有公开 reasoning path 数量和 consistency selection rule。

本项目采用：

`3 reasoning paths → strategy majority vote → semantic medoid response`

正式结果：

- 1210 / 1210
- unique final responses = 1125
- 358 / 1210 样本出现多个不同 reasoning path response

该方法标记为：

**Self-consistency reconstruction**

脚本：

`final/scripts/baselines/run_self_consistency.py`

---

### Self-Refine

论文没有公开具体 Prompt、iteration 数和 stopping criterion。

本项目采用：

`Zero-shot response → feedback → one refinement round`

正式结果：

- 1210 / 1210
- unique responses = 1202
- 1210 / 1210 outputs changed
- 127 个样本进行了长度修正

该方法标记为：

**Self-Refine reconstruction**

脚本：

`final/scripts/baselines/run_self_refine.py`

---

## 7. Table 3：LLM Judge

论文使用 GPT-4o 作为 Judge。

由于无法获得论文使用的具体 GPT-4o snapshot 和完整评价环境，本项目使用：

**Mistral-Small-24B**

作为 proxy judge。

为降低 position bias，正式评价采用 balanced cyclic assignment。

每一种方法在 A、B、C、D、E 五个位置均出现：

**242 次**

最终：

- 1210 / 1210 evaluated
- missing = 0

### Proxy Judge Result

| Method | Fluency | Identification | Comforting | Suggestion | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot | 3.788 | 3.260 | 3.156 | 2.704 | 3.157 |
| Few-shot CoT | 3.793 | 3.544 | 3.340 | 1.457 | 3.092 |
| Self-consistency | 3.595 | 3.493 | 3.235 | 1.444 | 2.989 |
| **Self-Refine** | **4.191** | **4.050** | **4.026** | **3.458** | **4.093** |
| MultiAgentESC | 4.026 | 3.430 | 3.357 | 2.551 | 3.339 |

论文 Overall 排名：

**MultiAgentESC > Zero-shot > Self-Refine > Few-shot CoT > Self-consistency**

本地 Proxy Overall 排名：

**Self-Refine > MultiAgentESC > Zero-shot > Few-shot CoT > Self-consistency**

MultiAgentESC 相对 Zero-shot、Few-shot CoT 和 Self-consistency 的优势仍部分保留。

但论文中的：

**MultiAgentESC > Self-Refine**

没有在本地 proxy judge 中复现。

因此 Table 3 应理解为：

**代理评价结果，而不是 GPT-4o 原始评价的严格复现。**

---

## 8. Table 2：模拟人工评价

论文 Table 2 使用三名具有心理学背景的研究生进行真人评价。

本项目没有真实标注者，因此论文原始 Table 2：

**Not reproduced**

为了验证评价流程，本项目额外进行了：

**LLM-simulated human evaluation**

使用 Mistral-Small-24B 的三种 evaluator profiles，分别偏重：

- 问题识别与建议
- 共情与情绪支持
- 综合支持质量

固定抽样规则：

`random.Random(42).sample(range(1210), 100)`

最终完成：

- 100 cases
- 3 simulated evaluator profiles
- 300 evaluation calls
- 300 / 300 successful

三种模拟评价视角完全一致率：

**50.1%**

注意：

该值不能解释为真正的人类 inter-annotator agreement。

### Overall 多数票结果

| Comparison | Win | Lose | Tie |
| --- | ---: | ---: | ---: |
| MultiAgentESC vs Zero-shot | 38 | 33 | 29 |
| MultiAgentESC vs Few-shot CoT | 16 | 77 | 7 |
| MultiAgentESC vs Self-consistency | 38 | 57 | 5 |
| MultiAgentESC vs Self-Refine | 9 | 90 | 1 |

模拟评价同样对本地 Self-Refine reconstruction 表现出较强偏好。

该实验只用于分析模型评价趋势，不能替代真实 human evaluation。

---

## 9. Table 4：Ablation

论文报告三组消融：

- w/o Dialogue Analysis
- w/o Experience
- w/o Group Discussion

官方仓库没有提供完整的 Table 4 ablation implementation。

因此本项目根据论文描述和源码结构重新实现三组实验。

该部分统一标记为：

**Reconstructed ablation**

正式结果保存在：

`final/results/table4_reconstructed/`

包括：

- `TABLE4_ANALYSIS.md`
- `TABLE4_RECONSTRUCTION.csv`
- `TABLE4_RESULT.md`
- `no_dialogue_analysis_100.json`
- `no_experience_100.json`
- `no_group_discussion_100.json`

---

## 10. 项目结构

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

其中：

- `final/scripts/`：本次 A800 + Qwen2.5-32B 正式复现实验脚本
- `final/results/`：正式实验输出
- 原有 `scripts/`：早期本地 smoke 和小模型验证阶段代码

---

## 11. 主要结论

本次复现得到三个主要结果。

第一，MultiAgentESC released generation pipeline 可以稳定运行，Table 1 自动指标与论文结果整体接近。

第二，Figure 3 的策略分布没有完全复现，说明策略行为可能比传统自动生成指标更加敏感。

第三，主观评价结果对 Judge Model 和 Baseline implementation 表现出明显敏感性。

其中最明显的差异是：

**Paper**

MultiAgentESC > Self-Refine

**Local Proxy**

Self-Refine > MultiAgentESC

因此本项目不支持“整篇论文已经完全复现”的表述。

更准确的结论是：

> **MultiAgentESC 的核心生成流程与自动指标具有较好的可复现性，但策略行为、Baseline 相对表现和主观评价结果对实现细节与评价模型较为敏感。**

---

## 12. 进一步研究问题

本次复现进一步产生了几个值得验证的问题：

1. 不同 LLM Judge 是否会显著改变情感支持对话方法的排名？
2. Self-Refine 的表现是否高度依赖 Prompt 和 refinement rounds？
3. MultiAgentESC 的 strategy distribution 是否对 test split、seed 和 quantization 敏感？
4. BLEU、ROUGE、Distinct 等自动指标与真实 emotional-support preference 的相关性有多高？

---

## 13. 复现边界

本项目没有证明：

- 所有论文数字都可以 exact reproduce
- Mistral-Small-24B 与 GPT-4o 的评价完全等价
- 模拟 evaluator 可以替代真实心理学背景标注者
- Reconstruction baseline 与作者内部实现完全一致
- Figure 3 的差异由某一个确定因素造成

这些问题目前均：

**尚不能确认。**

---

## Reference

**MultiAgentESC: A LLM-based Multi-Agent Collaboration Framework for Emotional Support Conversation**

**ESConv: A Dataset for Emotional Support Conversation**
