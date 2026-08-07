# Reproduction Notes

## Objective

This project reproduces the experimental pipeline of MultiAgentESC in a local environment.

The focus is:

- multi-agent workflow implementation
- local LLM inference
- evaluation pipeline construction

## Method Mapping

Original idea:

```
Dialogue Input
      ↓
Emotion Understanding
      ↓
Empathy Strategy Planning
      ↓
Response Generation
      ↓
Final Supportive Response
```

Local implementation:

```
ESConv Dataset
      ↓
Local LLM Agent Pipeline
      ↓
Generated Responses
      ↓
Evaluation
```

## Model Replacement

This reproduction uses:

```
Qwen2.5-7B + Ollama
```

The purpose is reproducibility with locally available models. Results should be interpreted as methodology reproduction rather than official benchmark reproduction.

## Evaluation Difference

Original experiments may rely on proprietary models, human annotation and specific evaluation settings.

This project provides:

- automatic metric evaluation
- local judge framework
- reproducible generation pipeline

## Current Status

Completed:

- environment setup
- local inference pipeline
- Table 1 evaluation pipeline

In progress:

- large-scale experiments
- baseline comparison
- metric calibration
