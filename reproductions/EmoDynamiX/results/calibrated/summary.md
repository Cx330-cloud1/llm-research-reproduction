# Environment-Calibrated Results

## Main Model

| Dataset | Paper M-F1 | Local M-F1 | Paper W-F1 | Local W-F1 | Paper Bias | Local Bias |
|---|---:|---:|---:|---:|---:|---:|
| ESConv | 27.70 | 28.09 | 32.71 | 32.04 | 0.45 | 0.4256 |
| AnnoMI | 27.92 | 28.07 | 35.33 | 34.38 | 0.50 | 0.4793 |

## Mixed Emotion Ablation

| Dataset | Full M-F1 | w/o Mixed M-F1 | Delta |
|---|---:|---:|---:|
| ESConv | 28.09 | 27.13 | -0.96 |
| AnnoMI | 28.07 | 27.00 | -1.07 |

## Summary

The environment-calibrated Full Model closely reproduces the paper's main Macro-F1 results on both ESConv and AnnoMI.

Removing Mixed Emotion decreases Macro-F1 on both datasets, reproducing the main direction of the paper's ablation finding. The exact degradation magnitude and Preference Bias behavior differ from the paper.
