# Rapid State + Adaptive-k Experiment

## 1. Goal

本实验是 3–5 天快速预实验，只回答两个问题：

1. Predicted State Card 是否能提高支持性回复质量？
2. Adaptive Top-1/Top-2 是否能在不明显降低质量的情况下节省生成成本？

本轮不训练新模型，不做完整论文实验。

## 2. Compute Allocation

### 本地 Windows

负责：

- 维护 Git 仓库与实验配置
- 固定 calibration、smoke20、pilot100 样本
- 构造 F0–F3 输入
- Temperature Scaling
- Adaptive-k 与随机预算对照
- 汇总指标和错误分析

### RTX 4090 租赁实例

只负责加载已有 EmoDynamiX checkpoint，并导出固定样本的 8 类原始 strategy logits。EmoDynamiX 不生成 State Card。

### A800 租赁实例

负责使用同一 Qwen2.5-7B revision 依次提取 Emotion、Cause、Intention、Support Need，构造 State Card，生成 Top-1/Top-2 候选，并运行固定 selector 与辅助 judge。

4090 与 A800 不需要同时开机。所有脚本先在本地准备并提交，再分别拉取到对应实例运行。

## 3. Fixed Conditions

- Dataset: ESConv
- Random seed: 42
- Generation model: Qwen2.5-7B
- Temperature: 0
- Maximum generated tokens: 400
- Response word limit: 30
- 相同的对话上下文、模板和解码参数
- 不向生成模型提供未来话轮、reference response 或 gold strategy
- 所有条件使用相同的固定样本 ID
- smoke20 是 pilot100 的固定子集
- calibration 集与 pilot100 按 dialogue ID 隔离

## 4. State Experiment

| ID | State input | Purpose |
|---|---|---|
| F0 | 无状态 | 核心 baseline |
| F1 | 原始 Emotion + Cause + Intention + Support Need | 检验丰富状态是否有用 |
| F2 | 压缩 State Card | 检验稳定压缩接口 |
| F3 | 打乱的 State Card | 排除模型只是从更长 Prompt 获益 |

F3 只能在 pilot100 内使用固定 seed 打乱，并且不能把 State Card 分配回原样本。

如果 F2 与 F0 没有趋势差异，停止状态字段拆解。

如果 F2 与 F3 接近，不能声称模型真正利用了正确状态。

## 5. Candidate Experiment

使用状态实验中表现最好的状态格式：

| ID | Method |
|---|---|
| K1 | 固定 Top-1 strategy，只生成一个回复 |
| K2 | 固定 Top-2 strategies，生成两个回复并使用固定 selector |
| KA | 按校准置信度选择 Top-1 或 Top-2 |
| KR | 保持与 KA 相同平均候选数，随机分配 Top-2 |

初始 Adaptive-k 规则：

```text
if calibrated_top1_probability >= 0.65
and calibrated_top1_minus_top2_margin >= 0.25:
    k = 1
else:
    k = 2
```

阈值只能在 calibration 集上确定。pilot100 只用于最终比较，不能用于调阈值。

KA 和 KR 复用 K2 已生成的候选池，不重新生成候选。

## 6. Prompt Conditions

四个状态条件只允许改变 State 区域，其余 system prompt、dialogue context、strategy instruction、输出限制必须完全一致。

Top-1 和 Top-2 候选必须分别明确指定目标 strategy。

Selector 只能看到：

- 当前对话上下文
- 当前 State Card
- 两个匿名候选 A/B

Selector 不能看到：

- strategy 概率
- 候选由 Top-1 还是 Top-2 产生
- gold response
- 自动评价分数

候选 A/B 顺序必须按固定 seed 随机化。

## 7. Evaluation

### Strategy metrics

- Top-1 accuracy/recall
- Top-2 recall
- NLL
- Brier score
- ECE
- 平均候选数

### Response metrics

- Empathy
- Relevance
- Helpfulness
- Premature advice
- Safety
- Semantic repetition
- Selector 选择第二策略候选的比例
- Token 数
- 延迟
- 失败和重试次数

LLM judge 只作为辅助趋势，不能单独证明用户体验提高。

### Human pilot

- 固定 30 条
- F0 vs F2
- K1 vs KA
- 每条 3 人
- 允许 tie
- 隐藏方法名称并随机化左右顺序

## 8. Decision Gates

继续扩大实验需要同时观察到：

- F2 相对 F0 的 tie-adjusted win rate 约为 55% 或更高
- F2 明显优于 F3
- KA 相对 K2 节省至少 20% 候选生成
- KA 相对 K1 没有明显质量下降
- KA 优于相同平均预算的 KR
- premature advice、安全问题和错误状态没有增加

停止条件：

- F2 ≈ F0：暂不继续状态字段消融
- F2 ≈ F3：状态信息可能没有被正确利用
- K2 ≈ K1：停止多候选研究
- KA ≈ KR：不确定性不能有效分配计算
- predicted state 明显弱于 F0：先处理状态预测误差
- 两个候选高度重复：保留 Top-1

## 9. Required Artifacts

每个样本必须保留：

- sample_id
- dialogue_id
- target_turn_index
- dialogue_context
- gold_strategy（仅评价使用）
- raw strategy logits
- calibrated probabilities
- predicted state fields
- State Card
- F0–F3 prompt version
- Top-1/Top-2 strategy
- candidate responses
- selector input and result
- generation parameters
- token usage
- latency
- failure/retry log

所有中间结果使用 JSONL，并支持按 sample_id 断点续跑。
