# EmoDynamiX Table 2 Reproduction Summary

## ESConv

| Variant | Paper M-F1 | Local M-F1 | Paper W-F1 | Local W-F1 | Paper Bias | Local Bias |
|---|---:|---:|---:|---:|---:|---:|
| Full | 27.70 | 26.63 | 32.71 | 31.12 | 0.45 | 0.4352 |
| w/o Graph Learning | 25.72 | 27.97 | 29.31 | 31.73 | 0.78 | 0.3603 |
| w/o Mixed Emotion | 25.90 | 26.34 | 29.45 | 30.74 | 0.66 | 0.4886 |
| w/o Discourse Parser | 26.64 | 26.11 | 30.12 | 30.46 | 0.59 | 0.3734 |
| w/o Dummy Node | 25.46 | 26.51 | 29.80 | 31.44 | 0.73 | 0.5008 |

## AnnoMI

| Variant | Paper M-F1 | Local M-F1 | Paper W-F1 | Local W-F1 | Paper Bias | Local Bias |
|---|---:|---:|---:|---:|---:|---:|
| Full | 27.92 | 26.64 | 35.33 | 33.87 | 0.50 | 0.5134 |
| w/o Graph Learning | 26.95 | 28.89 | 29.46 | 35.10 | 0.73 | 0.3901 |
| w/o Mixed Emotion | 24.71 | 25.36 | 30.25 | 31.41 | 0.70 | 0.4362 |
| w/o Discourse Parser | 27.04 | 28.02 | 31.59 | 36.15 | 0.60 | 0.3120 |
| w/o Dummy Node | 24.73 | 28.29 | 29.00 | 35.68 | 0.72 | 0.6183 |

## Diagnostic

ESConv Flattened Context baseline:

- Macro-F1: 26.35
- Weighted-F1: 30.05
- Preference Bias: 0.4275

Comparison:

- Flat Context: 26.35 M-F1
- Full EmoDynamiX: 26.63 M-F1
- Reconstructed w/o Graph Learning: 27.97 M-F1

## Interpretation

- The Full model is reproducible within approximately 1–2 F1 points of the paper.
- The official `erc_mixed` ablation reproduces the expected degradation direction on both datasets.
- The repository does not provide official implementations for w/o Graph Learning, w/o Discourse Parser, or w/o Dummy Node.
- These three variants were reconstructed from the paper's textual descriptions.
- The reconstructed structural ablations do not consistently reproduce the paper's reported degradation.
- The Flat Context diagnostic shows that the strong no-graph result is associated with the tag-enhanced flattened representation rather than plain flattened dialogue alone.
- Therefore, structural-ablation results should be reported as paper-description-based reconstructions rather than exact official reproductions.
