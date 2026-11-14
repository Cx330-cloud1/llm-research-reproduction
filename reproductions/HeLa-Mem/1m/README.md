# HeLa-Mem: PersonaMem 1M Reproduction

This repository contains an experimental reproduction and adaptation of **HeLa-Mem** for the **PersonaMem 1M** benchmark.

The project extends the original HeLa-Mem implementation with local LLM inference, structured PersonaMem data processing, shared-prefix memory construction, persistent inference caching, and cached evaluation.

The primary goal is to investigate the scalability and efficiency of long-term conversational memory construction and retrieval under large-scale personalized dialogue evaluation.

> **Reproduction status:** The 1M evaluation pipeline has produced 2,674 result records, and the dataset/result completeness audit passed. Exact algorithmic equivalence with the original independent execution pipeline and independently verified accuracy remain under investigation.

---

## 1. Project Overview

HeLa-Mem is a Hebbian-learning-based memory architecture designed for long-term conversational memory.

This reproduction adapts the architecture to PersonaMem 1M and introduces several engineering components:

- Local Qwen-based LLM inference.
- Structured PersonaMem data adaptation.
- Episodic memory construction with batch processing.
- Incremental long-term knowledge extraction.
- Shared-prefix memory encoding.
- Independent cutoff snapshots.
- Persistent inference caching.
- Cached evaluation and equivalence testing.

The implementation aims to reduce redundant memory construction across questions that share conversation histories.

## 2. Project Structure

```text
1m/
│
├── README.md
├── STATUS.md
├── requirements.txt
│
├── hela_mem/
│   ├── hebbian_memory.py
│   ├── hebbian_knowledge_memory.py
│   ├── hebbian_retriever.py
│   ├── local_qwen_bridge.py
│   ├── personamem_structured_adapter.py
│   ├── profile_utils.py
│   ├── utils.py
│   ├── encode_locomo.py
│   ├── encode_longmemeval.py
│   ├── eval_locomo.py
│   ├── eval_longmemeval.py
│   └── __init__.py
│
├── scripts/
│   ├── build_personamem_prefix_cache.py
│   └── eval_personamem_cached.py
│
├── tests/
│   ├── test_personamem_cache_equivalence.py
│   ├── test_structured_semantic_adapter.py
│   └── test_structured_semantic_render.py
│
├── docs/
│
└── results/
```

## 3. Main Components

### 3.1 Hebbian Episodic Memory

`hebbian_memory.py`

Implements the episodic memory graph and Hebbian memory operations.

The PersonaMem adaptation includes batch memory insertion to reduce repeated embedding and keyword-processing overhead.

### 3.2 Long-term Knowledge Memory

`hebbian_knowledge_memory.py`

Implements long-term knowledge storage and incremental knowledge extraction.

The adapted pipeline processes conversation turns in fixed-size batches and maintains an incomplete buffer for the remaining turns.

### 3.3 Local LLM Inference

`local_qwen_bridge.py`

Provides a local inference interface for Qwen-based models.

It supports:

- Batched completion.
- Chat generation.
- Persistent response caching.
- Reuse of previously generated outputs.

Model weights are not included in this repository.

### 3.4 Structured PersonaMem Adapter

`personamem_structured_adapter.py`

Provides structured data adaptation for PersonaMem conversations and evaluation inputs.

### 3.5 Shared-prefix Memory Construction

`scripts/build_personamem_prefix_cache.py`

Implements shared-prefix memory construction.

Questions associated with the same shared conversation context are processed in ascending cutoff order.

Instead of independently rebuilding the entire memory state for every question, the pipeline advances a canonical memory state and creates snapshots at required cutoffs.

### 3.6 Independent Cutoff Snapshots

Each cutoff snapshot uses independent copies of the episodic graph and long-term knowledge memory.

The remaining incomplete knowledge buffer is processed on the snapshot copy rather than directly modifying the canonical knowledge state.

This design is intended to preserve cutoff-specific memory states.

### 3.7 Cached Evaluation

`scripts/eval_personamem_cached.py`

Loads previously constructed memory snapshots and performs evaluation without rebuilding the complete conversation memory for every question.

## 4. Experimental Configuration

The experimental implementation uses a local Qwen inference backend.

The execution environment is expected to provide:

- Python.
- PyTorch.
- Sentence Transformers.
- NumPy.
- OpenAI-compatible interfaces where required.
- Locally available model weights.
- PersonaMem dataset files.

The exact model paths, inference settings, and runtime configurations must be configured in the execution environment.

Neither model weights nor original benchmark data are distributed in this repository.

## 5. Installation

Clone the repository and enter this directory.

```bash
cd reproductions/HeLa-Mem/1m
```

Install the listed dependencies:

```bash
pip install -r requirements.txt
```

Additional dependencies may be required depending on the execution environment and the selected inference backend.

## 6. Reproduction Pipeline

The experimental pipeline consists of the following stages.

### Stage 1: Data Preparation

Prepare PersonaMem conversation data and transform it into the structured format required by the memory system.

### Stage 2: Local Model Initialization

Configure the local inference backend and embedding model.

Ensure that the model paths and runtime dependencies are available.

### Stage 3: Shared-prefix Cache Construction

Run the prefix cache builder:

```bash
python scripts/build_personamem_prefix_cache.py
```

The actual dataset paths and runtime arguments should be configured according to the execution environment and script configuration.

### Stage 4: Cached Evaluation

Run the cached evaluation pipeline:

```bash
python scripts/eval_personamem_cached.py
```

Evaluation configuration, dataset paths, and cache locations must be set before execution.

### Stage 5: Validation

Run the available tests:

```bash
python -m pytest tests/ -v
```

Passing individual tests does not by itself establish complete equivalence with the original implementation.

## 7. PersonaMem 1M Data Integrity Audit

The local experiment audit reported the following results:

| Metric | Result |
|---|---:|
| Expected questions | 2,674 |
| Dataset rows | 2,674 |
| Result rows | 2,674 |
| Missing questions | 0 |
| Duplicate question IDs | 0 |
| Error rows | 0 |
| Scored rows | 2,674 |

**Data completeness audit: PASSED.**

These results establish the structural completeness of the generated evaluation records.

They do not independently establish:

- Model accuracy.
- Correctness of every prediction.
- Equivalence with the original HeLa-Mem implementation.
- Reproducibility across different hardware and software environments.

No accuracy claim is made without an independently verified aggregate evaluation report.

## 8. Validation and Reproducibility

The following validation aspects are being investigated:

1. Episodic memory batch processing versus original sequential processing.
2. Shared-prefix encoding versus independent per-question encoding.
3. Cutoff snapshot isolation.
4. Incremental knowledge extraction consistency.
5. Retrieval behavior and memory-state consistency.
6. Evaluation result reproducibility.

The test suite contains preliminary checks for cache equivalence and structured semantic adaptation.

Full algorithmic equivalence remains subject to further validation.

## 9. Data and Model Availability

The following materials are intentionally excluded:

- Original PersonaMem conversation data.
- Per-question evaluation records.
- Reference answers and generated responses.
- Model weights.
- Local inference databases.
- Private memory snapshots.
- Temporary cache files.

This repository focuses on the implementation and reproduction methodology rather than redistributing benchmark data or model checkpoints.

## 10. Limitations

This is an experimental adaptation rather than a claim of a fully verified reproduction.

Important limitations include:

- The original and optimized memory pipelines require further equivalence testing.
- The dependency configuration may need additional environment-specific packages.
- Local inference results may vary with model versions and generation settings.
- Complete reproducibility requires the original benchmark data and model checkpoints.
- Data completeness should not be interpreted as prediction accuracy.

## 11. Acknowledgements

This project builds upon the original HeLa-Mem implementation and adapts it for PersonaMem-based long-term conversational memory experiments.

Please refer to the original project and its license for upstream implementation details and attribution.

---

**Project:** HeLa-Mem PersonaMem 1M Reproduction

**Repository:** `llm-research-reproduction`

**Directory:** `reproductions/HeLa-Mem/1m/`

**Status:** Experimental reproduction; validation ongoing.
