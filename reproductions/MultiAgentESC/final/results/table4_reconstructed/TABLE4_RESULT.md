# MultiAgentESC Table 4 Reconstruction Result

## Experimental Setup

- Dataset: ESConv
- Backbone: Qwen2.5-32B (Q4_K_M via Ollama)
- Evaluation: same evaluation pipeline as baseline
- Unit: supporter-turn level
- Records: 1210

---

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

## Ablation 1: w/o Group Discussion

Result:

| Metric | Delta |
|---|---:|
| D-1 | +0.1166 |
| D-2 | -0.0087 |
| B-1 | +0.1260 |
| B-2 | +0.0453 |
| F1 | +0.0129 |
| R-L | +0.0659 |

Observation:

The reconstructed ablation does not reproduce the performance degradation reported in the paper.

Possible reason:

The official repository does not release the exact ablation implementation and prompts.

---

## Ablation 2: w/o Experience

Result:

| Metric | Delta |
|---|---:|
| D-1 | -0.5062 |
| D-2 | -3.3520 |
| B-1 | -0.4251 |
| B-2 | -0.3211 |
| F1 | -0.5669 |
| R-L | -0.5181 |

Observation:

The reconstructed ablation successfully reproduces the overall degradation trend.

Removing experience examples reduces response diversity and generation quality.

---

## Current Conclusion

Table 4 is partially reproduced.

- w/o Experience: reproduced
- w/o Group Discussion: not reproduced

The inconsistency suggests that some details of the original ablation implementation may not be fully available in the released code.

---

## Overall Conclusion

The reconstructed Table 4 shows partial consistency with the original paper.

| Ablation | Result |
|---|---|
| w/o Experience | Performance degradation reproduced |
| w/o Group Discussion | Performance degradation not reproduced |
| w/o Dialogue Analysis | Pending |

The released implementation does not include the exact original ablation settings. Therefore, differences between reconstructed and reported results may come from unavailable prompts, agent interaction settings, or implementation details.

