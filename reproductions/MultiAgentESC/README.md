# MultiAgentESC Local Reproduction

A local reproduction project for **MultiAgentESC: Multi-Agent Framework for Empathetic Support Conversation**.

This repository implements the paper's experimental workflow using local large language model inference and reproduces the evaluation pipeline of Table 1-4.

The goal is transparent methodology reproduction rather than claiming exact reproduction of original benchmark scores.

---

## Overview

MultiAgentESC improves empathetic dialogue generation by decomposing the task into multiple agent stages:

```
User Input
    ↓
Emotion Understanding Agent
    ↓
Empathy Strategy Agent
    ↓
Response Generation Agent
    ↓
Final Response
```

This project rebuilds this workflow with local LLM deployment.

---

## Environment

Recommended:

```
Python >= 3.11
```

Tested:

```
Windows + Python 3.14
```

Model backend:

```
Ollama
```

Model:

```
Qwen2.5-7B
```

---

## Installation

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For complete evaluation:

```bash
pip install -r requirements-full.txt
```

Install model:

```bash
ollama pull qwen2.5:7b
```

---

## Run Experiments

### Offline Validation

```bash
python tests/test_offline.py
```

### Smoke Test

```bash
python scripts/run_generations.py --config config.smoke.json
```

### Table 1

```bash
python scripts/build_table1.py
```

### Table 3 Judge

```bash
python scripts/run_table3.py
```

### Table 4 Ablation

```bash
python scripts/run_table4.py
```

---

## Structure

```
├── data/
├── scripts/
├── outputs/
├── tests/
├── docs/
├── config.full.json
├── config.smoke.json
└── requirements.txt
```

---

## Status

| Component | Status |
|---|---|
| Environment | Completed |
| Local inference | Completed |
| Multi-agent pipeline | Completed |
| Table 1 pipeline | Completed |
| Table 2-4 experiments | In progress |

---

## Limitations

Differences from the original paper may come from:

- local model capability
- evaluation settings
- unavailable human annotation environment

See `docs/reproduction_notes.md` for details.
