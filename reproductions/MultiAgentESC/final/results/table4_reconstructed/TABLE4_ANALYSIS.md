# MultiAgentESC Table 4 Reconstruction Analysis

## Overview

This analysis reconstructs the ablation study of MultiAgentESC.

Experimental setting:

- Dataset: ESConv
- Backbone: Qwen2.5-32B (Q4_K_M)
- Evaluation unit: supporter turns
- Total samples: 1210

The reconstruction successfully reproduces the effect of removing the Experience module, while the effect of removing Group Discussion is not observed under the current implementation.

---

# 1. Automatic Metric Analysis

## Baseline

| Metric | Score |
|---|---:|
| D-1 | 6.9867 |
| D-2 | 35.6841 |
| B-1 | 17.2166 |
| B-2 | 4.9609 |
| F1 | 18.0298 |
| R-L | 14.4627 |

---

## w/o Experience

| Metric | Score | Delta |
|---|---:|---:|
| D-1 | 6.4806 | -0.5062 |
| D-2 | 32.3320 | -3.3520 |
| B-1 | 16.7915 | -0.4251 |
| B-2 | 4.6398 | -0.3211 |
| F1 | 17.4628 | -0.5669 |
| R-L | 13.9446 | -0.5181 |

Observation:

Removing experience examples causes consistent performance degradation.

---

## w/o Group Discussion

| Metric | Score | Delta |
|---|---:|---:|
| D-1 | 7.1034 | +0.1166 |
| D-2 | 35.6754 | -0.0087 |
| B-1 | 17.3426 | +0.1260 |
| B-2 | 5.0062 | +0.0453 |
| F1 | 18.0427 | +0.0129 |
| R-L | 14.5285 | +0.0659 |

Observation:

Removing group discussion does not reproduce the degradation reported in the original paper.

---

# 2. Strategy Distribution Analysis

## Baseline

| Strategy | Ratio |
|---|---:|
| None | 36.94% |
| Affirmation and Reassurance | 32.89% |
| Providing Suggestions | 13.72% |
| Reflection of feelings | 5.62% |

---

## w/o Experience

| Strategy | Ratio |
|---|---:|
| Reflection of feelings | 43.72% |
| None | 32.98% |
| Affirmation and Reassurance | 20.17% |
| Providing Suggestions | 0.99% |

Observation:

Removing experience examples causes strategy concentration.

The model relies much more on emotional reflection while using fewer suggestion and question strategies.

---

## w/o Group Discussion

| Strategy | Ratio |
|---|---:|
| None | 34.38% |
| Affirmation and Reassurance | 32.40% |
| Providing Suggestions | 12.98% |
| Reflection of feelings | 8.10% |

Observation:

Strategy distribution remains close to baseline.

---

# 3. Response Length Analysis

| Setting | Avg Characters | Avg Words |
|---|---:|---:|
| Baseline | 125.15 | 20.84 |
| w/o Experience | 129.88 | 21.62 |
| w/o Group Discussion | 122.63 | 20.45 |

Observation:

Performance degradation in w/o Experience is not caused by shorter responses.

The model generates slightly longer responses but with less diverse strategies.

---

# 4. Lexical Diversity Analysis

Distinct-n analysis:

| Setting | D-1 | D-2 |
|---|---:|---:|
| Baseline | 13.1255 | 47.7174 |
| w/o Experience | 11.5702 | 42.0087 |
| w/o Group Discussion | 13.1841 | 47.6755 |

Observation:

Removing experience significantly reduces lexical diversity.

This is consistent with the automatic metric degradation.

---

# 5. Overall Conclusion

The reconstructed ablation study indicates:

## Experience Module

Removing experience examples leads to:

1. Lower automatic evaluation scores.
2. Reduced strategy diversity.
3. Increased dependence on generic emotional reflection.
4. Lower lexical diversity.

Therefore, the Experience module appears to contribute to both strategic planning and response diversity.

---

## Group Discussion Module

The reconstructed removal of group discussion does not lead to measurable degradation.

Possible explanations:

1. The released code may not contain all original discussion configurations.
2. Prompt templates or interaction protocols may differ from the paper.
3. A stronger backbone model may reduce the marginal contribution of multi-agent discussion.

---

Current conclusion:

The Table 4 reconstruction is partially successful.

- w/o Experience: reproduced.
- w/o Group Discussion: not reproduced.
- w/o Dialogue Analysis: pending.
