MultiAgentESC 论文复现
本项目复现论文：
MultiAgentESC: A LLM-based Multi-Agent Collaboration Framework for Emotional Support Conversation
会议：EMNLP 2025
本项目围绕 MultiAgentESC 的官方生成流程、自动评价、baseline、LLM Judge、人工评价流程与消融实验进行复现与重构。
由于论文并未公开全部 baseline、评价器和消融实现，本项目明确区分：
Official reproduction：依据作者公开代码直接运行
Proxy evaluation：使用替代评价器进行评价
Reconstruction：依据论文和源码线索重新实现
Not reproduced：缺少必要条件，无法严格复现
---
1. 实验环境
项目	配置
GPU	NVIDIA A800 80GB PCIe
Python	3.10.21
推理框架	Ollama 0.5.11
主模型	Qwen2.5-32B
Embedding	all-roberta-large-v1
Proxy Judge	Mistral-Small-24B
数据集	ESConv
Seed	42
官方代码固定版本：
```text
631b7f1961fc7502e547fd9258e847230dbcb973
```
主要依赖：
```text
openai==1.47.1
autogen==0.2.31
sentence-transformers==3.3.1
tokenizers==0.19.1
torch==2.4.1
tqdm==4.66.4
```
论文未公开 Qwen2.5-32B 的具体 Ollama quantization。
本地使用：
```text
qwen2.5:32b
Q4_K_M
```
因此无法确认量化版本与论文实验完全一致。
---
2. 复现状态
实验	状态	类型
MultiAgentESC 官方生成	完成	Official reproduction
Table 1 自动指标	接近论文	Proxy evaluator
Figure 3 策略分布	未完全复现	Official output analysis
Zero-shot	完成	Published-prompt implementation
Few-shot CoT	完成	Reconstruction
Self-consistency	完成	Reconstruction
Self-Refine	完成	Reconstruction
Table 3	完成	Mistral proxy evaluation
Table 2 真人评价	未严格复现	Not reproduced
Table 2 模拟评价	完成	LLM-simulated evaluation
Table 4 消融	完成	Reconstruction
---
3. 官方生成流程
官方 `main.py` 实际使用：
```python
samples = dataset[:100]
```
论文文本描述为随机选择 100 个 conversations，但 released code 使用前 100 个 conversations。
本项目正式实验优先遵循 released code。
前 100 个 ESConv conversations 中共有：
```text
1544 supporter utterances
```
官方代码会将连续 supporter utterances 合并，因此最终得到：
```text
100 conversations
1210 response-level generation positions
```
正式生成：
```text
1210 / 1210
```
---
4. Table 1
MultiAgentESC 官方仓库没有公开 Table 1 的完整评价代码。
本项目使用 ESConv 官方 metric implementation 作为 proxy evaluator。
Metric	Paper	Local	Difference
D-1	6.78	6.9867	+0.2067
D-2	35.15	35.6841	+0.5341
B-1	17.66	17.2166	-0.4434
B-2	5.38	4.9609	-0.4191
B-3	2.35	2.2018	-0.1482
F1	18.30	18.0298	-0.2702
ROUGE-L	14.66	14.4627	-0.1973
整体结果与论文较接近。
因此 Table 1 是本次复现中最稳定的部分。
---
5. Figure 3
正式生成结果中的策略分布没有完全恢复论文 Figure 3。
目前可能影响因素包括：
论文随机测试集与 released code `dataset[:100]` 的差异
Qwen2.5-32B checkpoint / quantization
Ollama 版本
strategy normalization
未公开实验细节
目前尚不能确认具体原因。
---
6. Baseline Reconstruction
Zero-shot
论文公开了 Zero-shot Prompt。
正式生成：
```text
1210 / 1210
```
Few-shot CoT
论文没有公开：
实际 few-shot examples
reasoning
example 数量
example selection rule
本项目基于官方 `get_cases()` 构建 case pool，并使用：
```text
all-roberta-large-v1
→ similarity retrieval
→ top-3 examples
→ Few-shot CoT generation
```
正式结果：
```text
1210 / 1210
unique responses = 1128
exact example copies = 1
```
该方法标记为：
Few-shot CoT reconstruction
Self-consistency
论文未公开 reasoning path 数量和 consistency selection rule。
本项目采用：
```text
3 reasoning paths
→ strategy majority vote
→ semantic medoid response
```
正式结果：
```text
1210 / 1210
unique final responses = 1125
358 / 1210 samples contained multiple unique paths
```
该方法标记为：
Self-consistency reconstruction
Self-Refine
论文未公开具体 Prompt、iteration 数和 stopping criterion。
本项目采用：
```text
Zero-shot response
→ feedback
→ one refinement round
```
正式结果：
```text
1210 / 1210
unique responses = 1202
1210 / 1210 outputs changed
127 outputs required length correction
```
该方法标记为：
Self-Refine reconstruction
---
7. Table 3：LLM Judge
论文使用 GPT-4o Judge。
由于具体 GPT-4o snapshot 和原始评价输出未公开，本项目使用：
```text
Mistral-Small-24B
```
作为 proxy judge。
为减少 position bias，正式评价使用 balanced cyclic assignment。
每种方法在 A / B / C / D / E 五个位置均出现 242 次。
最终：
```text
1210 / 1210
missing = 0
```
Proxy Result
Method	Fluency	Identification	Comforting	Suggestion	Overall
Zero-shot	3.788	3.260	3.156	2.704	3.157
Few-shot CoT	3.793	3.544	3.340	1.457	3.092
Self-consistency	3.595	3.493	3.235	1.444	2.989
Self-Refine	4.191	4.050	4.026	3.458	4.093
MultiAgentESC	4.026	3.430	3.357	2.551	3.339
论文 Overall 排名：
```text
MultiAgentESC
> Zero-shot
> Self-Refine
> Few-shot CoT
> Self-consistency
```
本地 Proxy 排名：
```text
Self-Refine
> MultiAgentESC
> Zero-shot
> Few-shot CoT
> Self-consistency
```
MultiAgentESC 相对 Zero-shot、Few-shot CoT 和 Self-consistency 的优势部分保留，但论文中 `MultiAgentESC > Self-Refine` 没有在本地 proxy judge 中复现。
---
8. Table 2：模拟人工评价
论文 Table 2 使用 3 名 psychology-background postgraduate annotators。
本项目没有真实标注者，因此原始 human evaluation：
```text
Not reproduced
```
为了验证评价流程，本项目额外进行：
LLM-simulated human evaluation
使用 Mistral-Small-24B 的三种 evaluator profiles。
固定抽样：
```python
random.Random(42).sample(range(1210), 100)
```
最终：
```text
100 cases
3 simulated evaluators
300 evaluation calls
300 / 300 successful
```
三种评价视角完全一致率：
```text
50.1%
```
Overall 多数票：
Comparison	Win	Lose	Tie
MultiAgentESC vs Zero-shot	38	33	29
MultiAgentESC vs Few-shot CoT	16	77	7
MultiAgentESC vs Self-consistency	38	57	5
MultiAgentESC vs Self-Refine	9	90	1
该结果只用于分析模型评价偏好，不能替代真实 human evaluation。
---
9. Table 4：Ablation
论文报告：
```text
w/o Dialogue Analysis
w/o Experience
w/o Group Discussion
```
官方仓库没有公开完整 Table 4 implementation。
因此本项目依据论文和源码结构完成三组：
Reconstructed ablation
不能描述为作者官方消融代码的严格复现。
---
10. 项目结构
```text
MultiAgentESC/
│
├── README.md
├── SOURCE.md
├── VALIDATION.md
│
├── data/
│   └── ESConv.json
│
├── final/
│   └── scripts/
│       ├── baselines/
│       │   ├── run_zero_shot.py
│       │   ├── run_fewshot_cot_retrieval.py
│       │   ├── run_self_consistency.py
│       │   └── run_self_refine.py
│       │
│       └── evaluation/
│           ├── run_mistral_proxy_judge_balanced_json.py
│           └── run_simulated_human_eval_v2.py
│
├── scripts/
└── tests/
```
`final/scripts/` 为本次 A800 + Qwen2.5-32B 正式复现实验使用的最终脚本。
原来的 `scripts/` 和相关配置为早期本地 smoke / 小模型验证阶段的实现。
---
11. 主要结论
本次复现得到三个主要结果。
第一，MultiAgentESC released generation pipeline 可以稳定运行，Table 1 自动指标与论文结果整体接近。
第二，Figure 3 的策略分布没有完全复现，说明策略行为可能比自动生成指标更加敏感。
第三，主观评价结果对 Judge Model 和 baseline implementation 表现出明显敏感性。
特别是：
```text
Paper:
MultiAgentESC > Self-Refine

Local proxy:
Self-Refine > MultiAgentESC
```
因此本项目不支持“整篇论文完全复现”的表述。
更准确的结论是：
> **MultiAgentESC 的核心生成流程与自动指标具有较好的可复现性，但策略行为、baseline 相对表现和主观评价结果对实现细节与评价模型较为敏感。**
---
12. 后续研究问题
本次复现进一步产生了以下问题：
不同 LLM Judge 是否会显著改变 ESC 方法排名？
Self-Refine 的表现是否高度依赖 Prompt 与 refinement round？
Strategy distribution 是否对 test split、seed 和 quantization 敏感？
BLEU、ROUGE、Distinct 等自动指标与真实 emotional-support preference 的相关性有多高？
---
Reference
MultiAgentESC: A LLM-based Multi-Agent Collaboration Framework for Emotional Support Conversation
ESConv: A Dataset for Emotional Support Conversation
