# SDDP Reproduction

**Status:** Metric-Level Reproduction Completed

## Results

| Experiment | F1 |
|---|---:|
| 3-epoch recovery checkpoint | 57.29 |
| 6-epoch final checkpoint | **60.46** |
| Independent checkpoint reload | **60.46** |

Final F1: **0.6045929018789143**

## Configuration

- Dataset: STAC
- Seed: 1
- Learning rate: 2e-5
- Epochs: 6
- Train batch size: 2
- Warmup ratio: 0.1
- Decoder: KBest spanning-tree decoder

The original spanning-tree decoder and required dependencies were restored successfully. The saved checkpoint was independently reloaded and reproduced the same F1 score.

The calibrated EmoDynamiX main-model experiments still use the official preprocessed `parsed_dialogue`, so this result is reported separately from the main-model reproduction.

Model weights, raw server logs, credentials, private paths, cache files, and restricted datasets are not included.
