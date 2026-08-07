# Validation record

Validation date: 2026-08-07

## Checks passed

- Seven offline unit tests passed.
- The bundled ESConv parser extracts 1,210 supporter targets from the first 100 dialogues, matching the audited public-code behavior.
- Mock Table 1 generated all 13 method rows.
- Mock Table 2 generated the expected blind-evaluation form and private mapping.
- Mock Table 3 completed local-judge prompt generation and five-dimension score parsing.
- Mock Table 4 generated the full model and all three ablation rows.
- Python source compilation passed.
- Both configurations point the judge to local Ollama rather than an OpenAI endpoint.

## Not validated in this environment

- Qwen2.5-32b and LLaMA3-70b inference, because the required Ollama model weights and hardware are not present here.
- Similarity retrieval with `all-roberta-large-v1`, because the full model dependency is intended for the user's experiment machine.
- Human ratings for Table 2.
- Numerical agreement with the paper, which can only be measured after real local generation.

Mock outputs were used only for plumbing validation and are intentionally excluded from the deliverable archive.
