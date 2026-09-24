# Rapid State + Adaptive-k 实验系统设计

日期：2026-09-23  
状态：已确认，待实施

## 1. 目标与边界

本项目构建一个独立实验控制层，将 EmoDynamiX 的策略预测与 Qwen2.5-7B 的回复生成连接起来，快速回答两个问题：

1. Predicted State Card 是否能提高情感支持回复质量？
2. Adaptive Top-1/Top-2 是否能在不明显降低质量的情况下减少候选生成成本？

本轮不重新训练 EmoDynamiX，不修改两篇论文的官方核心代码，不开展 Graph/Discourse 重训、多智能体辩论、Top-3+、多 selector 比较、多 seed 或完整大规模人工评价。

## 2. 实现方案

采用独立实验层：所有新增代码位于 `experiments/rapid_state_adaptive_k/`，仅调用现有项目接口。

不采用以下方案：

- 直接修改 EmoDynamiX trainer 或 MultiAgentESC generation pipeline：会污染原复现代码和已有证据。
- Notebook 或临时脚本串联：不利于测试、断点恢复、哈希验证和跨机器交接。

## 3. 系统架构与数据流

```text
ESConv.json
  ↓ 官方 seed=13 对话级划分重建
valid dialogues → dev_cal 全部目标轮
test dialogues  → 每对话最后目标轮 → pilot100 → smoke20
  ↓
4090：EmoDynamiX checkpoint-2600
  ↓
raw strategy logits + label order + predicted state
  ↓
本地：Temperature Scaling + calibrated probabilities
  ↓
K1 / K2 / KA / KR 分配清单
  ↓
A800：Qwen2.5-7B 回复生成与固定 selector
  ↓
本地：自动评价、人工 pilot、错误分析
```

计算分工：

- 本地 Windows：代码、manifest、校准、Adaptive-k、评价和报告。
- RTX 4090：加载已有 EmoDynamiX checkpoint 并导出真实 logits；不重训。
- A800：运行 Qwen2.5-7B 候选生成、固定 selector 和辅助 judge。
- 两张租赁卡不同时开启。

## 4. 上下文定义

每条样本保留两种输入：

- `model_context`：目标回复前最近 5 个话轮，严格匹配 EmoDynamiX 的官方预处理窗口。
- `generation_context`：目标回复前全部可见历史，供所有回复生成条件统一使用。

两种上下文均不得包含目标回复或未来话轮。F0–F3 的 `generation_context` 完全相同，唯一允许变化的是 State 区域。

## 5. 官方划分重建与抽样

严格复现 `data/esconv/make.py`：

- 使用 Python `random.seed(13)`；
- 对 1300 个完整对话打乱；
- valid 为前 15%，共 195 个对话；
- test 为随后 15%，共 195 个对话；
- train 为剩余 70%，共 910 个对话；
- 每个带策略标注的 supporter 轮单独成为目标，不合并连续 supporter 轮。

实验集合：

- `dev_cal`：官方 valid 的全部目标轮，用于拟合温度与选择 Adaptive-k 阈值。
- `pilot100`：官方 test 中每个对话的最后一个带策略标签的 supporter 轮；从 195 个候选对话中按自然末轮策略比例、seed=42 固定抽取 100 个不同对话。
- `smoke20`：从 pilot100 按相同比例、seed=42 固定抽取 20 条。

完整策略分类指标可在官方 test 的全部目标轮上报告；pilot100 仅用于快速回复实验。

## 6. Manifest 与评价侧文件

### 6.1 生成清单

`generation_manifest.jsonl` 仅包含模型运行允许看到的信息：

```text
schema_version
sample_id
split
dialogue_id
target_turn_id
visible_turn_ids
model_context
generation_context
context_hash
predicted_state
state_card
strategy_logits
label_order
raw_probabilities
calibrated_probabilities
top1_strategy
top2_strategy
condition_id
```

### 6.2 评价侧文件

`evaluation_sidecar.jsonl` 只在本地评价阶段使用：

```text
sample_id
gold_strategy
reference_response
original_dialogue_index
target_turn_index
```

gold strategy 与 reference response 不进入 A800 生成输入。

## 7. 状态实验

| ID | State input | 目的 |
|---|---|---|
| F0 | 无状态 | 核心 baseline |
| F1 | Emotion + Cause + Intention + Support Need 原始字段 | 检验丰富状态是否有用 |
| F2 | 压缩 State Card | 检验稳定压缩接口 |
| F3 | 来自其他样本的 State Card | 排除 Prompt 变长效应 |

F3 在 pilot100 内使用固定 seed 做无自配对乱序，并尽量保持长度分布接近。

## 8. Temperature Scaling

4090 输出分类器未经 softmax 的 strategy logits。仅使用 dev_cal 拟合标量温度：

\[
p_i = \operatorname{softmax}(z_i/T)
\]

优化目标为最小化 dev_cal NLL，并报告校准前后 NLL、Brier Score 和 ECE。pilot100 只应用冻结的温度，不参与拟合。

若校准未改善 NLL，则保留 raw distribution，并明确报告校准无收益。

## 9. Candidate 与 Adaptive-k 实验

| ID | 方法 |
|---|---|
| K1 | 固定 Top-1，只使用一个候选 |
| K2 | 固定 Top-2，生成两个候选并使用 selector |
| KA | 按校准置信度选择 Top-1 或 Top-2 |
| KR | 与 KA 保持相同平均候选数，随机分配 Top-2 |

在 dev_cal 上搜索有限阈值：

- top-1 probability threshold：0.50–0.80；
- top-1/top-2 margin threshold：0.10–0.35。

选择优先级：保持 Top-2 recall、降低平均候选数，并以相对 K2 至少节省 20% 候选为目标。阈值冻结后不得根据 pilot100 调整。

A800 为 pilot100 统一生成 Top-1 与 Top-2 strategy candidates；K1、K2、KA、KR 均复用同一候选池，不重复调用模型。

## 10. 生成条件与 selector

Qwen2.5-7B 固定参数：

```text
temperature = 0
max_tokens = 400
response_word_limit = 30
seed = 42
```

四个状态条件只允许改变 State 区域。Top-1 与 Top-2 候选只改变指定 strategy。

Selector 只能看到对话上下文、当前 State Card 和匿名候选 A/B；不能看到 strategy 概率、候选来源、gold response 或自动评价分数。候选顺序按固定 seed 随机化。

## 11. 缓存、恢复与失败记录

缓存键至少包含：

```text
data_hash
sample_id
model_revision
prompt_hash
decoding_hash
condition_id
stage
```

要求：

- 每完成一条样本立即追加 JSONL；
- 已成功样本自动跳过；
- 失败样本最多重试 3 次；
- 保存每次错误、时间、attempt 与 fallback；
- 不同 condition 使用独立路径；
- 正式结果禁止 `--overwrite`；
- smoke20 未全部通过时禁止启动 pilot100；
- 冷运行与缓存运行分开统计。

## 12. 自动验证

manifest 生成后必须自动检查：

1. dev_cal 与 pilot100 的 dialogue ID 无交集；
2. pilot100 恰好 100 条且来自 100 个不同对话；
3. smoke20 恰好 20 条且完全属于 pilot100；
4. context 的最后一轮严格早于 target；
5. 生成文件不包含目标回复和未来话轮；
6. sample_id、context_hash 唯一且可重复生成；
7. gold/reference 仅存在于 evaluation sidecar；
8. 重复运行产生完全相同的 SHA256；
9. F3 不发生 State Card 自配对；
10. logits 长度与 label order 长度一致。

## 13. 评价

### 13.1 策略层

- Top-1 accuracy 与 macro recall；
- Top-2 recall；
- NLL、Brier Score、ECE；
- 校准温度；
- KA 的 k=1/k=2 比例、平均候选数和候选节省率；
- KA 与 KR 的预算一致性。

### 13.2 回复层

核心比较：F0 vs F2、K1 vs KA。解释性对照：F1、F3、K2、KR。

指标：Empathy、Relevance、Helpfulness、Premature advice、Safety、Semantic repetition、token 数、latency、失败与重试、selector 选择 Top-2 的比例。

LLM judge 只作为辅助趋势证据。

### 13.3 人工 pilot

- 固定 30 条；
- F0 vs F2、K1 vs KA；
- 匿名 A/B，固定 seed 随机左右顺序；
- 允许 tie；
- 理想状态每条 3 人。

若仅由一人完成，标记为 `single-rater formative pilot`，不作为正式人类评价结论。

## 14. 文件结构

```text
experiments/rapid_state_adaptive_k/
├── README.md
├── configs/
│   ├── experiment.json
│   ├── calibration.json
│   └── generation.json
├── manifests/
├── scripts/
│   ├── build_manifests.py
│   ├── validate_manifests.py
│   ├── export_emodynamix_logits.py
│   ├── calibrate_and_allocate.py
│   └── build_generation_tasks.py
├── tests/
│   ├── test_build_manifests.py
│   ├── test_validate_manifests.py
│   └── test_calibration.py
└── reports/
```

真实大体积输出、checkpoint 和模型权重不提交 Git；固定 manifest、配置、脚本、测试和轻量汇总可以提交。

## 15. Go/No-go

继续扩大实验需要同时观察到：

- F2 相对 F0 出现稳定正向趋势；
- F2 明显优于 F3；
- KA 相对 K2 节省至少 20% 候选；
- KA 相对 K1 没有明显质量下降；
- KA 优于相同预算的 KR；
- premature advice 与安全问题不增加。

停止或回退规则：

- F2 ≈ F3：状态未被真正利用；
- F2 弱于 F0：预测状态可能误导生成；
- K2 ≈ K1：停止多候选研究；
- KA ≈ KR：不确定性不能有效分配计算；
- 候选高度重复：保留 Top-1；
- 校准不改善：保留 raw distribution 并报告。

## 16. 实施顺序

```text
本地：manifest 与测试
→ 4090：资产核对与 logits 导出
→ 本地：校准与 K 分配
→ A800：smoke20
→ A800：pilot100
→ 本地：评价、人工 pilot 与报告
```
