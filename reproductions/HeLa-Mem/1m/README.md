# HeLa-Mem：PersonaMem 1M 复现

本目录提供 HeLa-Mem 在 PersonaMem 1M 数据设置下的适配与复现实现，包括核心记忆模块、本地大语言模型推理接口、PersonaMem 数据适配、共享前缀缓存构建、缓存评测以及相关测试。

本项目基于 HeLa-Mem 原始实现进行扩展，主要目的是研究和验证 HeLa-Mem 在长对话用户记忆场景下的记忆构建、知识提取、记忆检索与个性化问答能力。

---

## 1. 项目简介

### 1.1 研究目标

PersonaMem 关注长对话场景中的用户个性化记忆与问答问题。

本复现工作将 HeLa-Mem 的记忆机制适配到 PersonaMem 数据格式，主要涉及：

- **情景记忆（Episodic Memory）**：利用 Hebbian Memory Graph 存储和关联历史对话信息。
- **知识记忆（Knowledge Memory）**：从对话中提取和维护长期知识。
- **记忆检索（Memory Retrieval）**：通过记忆图和相关检索机制获取与当前问题有关的信息。
- **用户画像（User Profile）**：支持用户长期信息的组织和检索。
- **本地大模型推理**：通过 vLLM 加载本地 Qwen 模型，完成知识处理和问答生成。

### 1.2 本次复现的主要工作

相较于直接运行原始 HeLa-Mem 实现，本项目针对 PersonaMem 的数据结构和实验需求进行了以下适配：

1. 实现 PersonaMem 对话数据的结构化转换。
2. 接入本地 Qwen 模型，减少对远程 LLM API 的依赖。
3. 实现 Qwen 推理结果的 SQLite 持久化缓存。
4. 实现共享对话前缀（Shared Prefix）的统一编码。
5. 为不同问题截止位置（Cutoff）构建独立记忆快照。
6. 实现缓存状态加载与后续问答评测。
7. 提供缓存等价性测试及结构化语义适配测试。

---

## 2. 项目目录

```text
1m/
├── README.md
├── STATUS.md
├── .gitignore
├── requirements.txt
│
├── hela_mem/
│   ├── __init__.py
│   ├── hebbian_memory.py
│   ├── hebbian_knowledge_memory.py
│   ├── hebbian_retriever.py
│   ├── profile_utils.py
│   ├── utils.py
│   ├── local_qwen_bridge.py
│   ├── personamem_structured_adapter.py
│   ├── encode_locomo.py
│   ├── encode_longmemeval.py
│   ├── eval_locomo.py
│   └── eval_longmemeval.py
│
├── scripts/
│   ├── eval_personamem_local.py
│   ├── eval_personamem_local_full.py
│   ├── build_personamem_prefix_cache.py
│   └── eval_personamem_cached.py
│
└── tests/
    ├── test_personamem_cache_equivalence.py
    ├── test_structured_semantic_adapter.py
    └── test_structured_semantic_render.py
```

### 核心文件说明

| 文件 | 功能 |
|---|---|
| `hebbian_memory.py` | Hebbian 情景记忆图及记忆更新机制 |
| `hebbian_knowledge_memory.py` | 长期知识记忆管理 |
| `hebbian_retriever.py` | 记忆检索与关联激活 |
| `profile_utils.py` | 用户画像相关处理 |
| `utils.py` | Embedding、关键词及通用工具 |
| `local_qwen_bridge.py` | 本地 Qwen 推理与持久化缓存 |
| `personamem_structured_adapter.py` | PersonaMem 结构化语义适配 |
| `build_personamem_prefix_cache.py` | 共享前缀记忆缓存构建 |
| `eval_personamem_cached.py` | 基于缓存的 PersonaMem 问答评测 |

---

## 3. 环境配置

### 3.1 基础环境

原项目使用 Python 3.10 及以上版本。

主要依赖：

- Python
- PyTorch
- NumPy
- Sentence Transformers
- vLLM
- Pandas
- OpenAI Python SDK
- tqdm

安装依赖：

```bash
pip install -r requirements.txt
```

注意：vLLM 与 PyTorch、CUDA 版本存在兼容性要求。建议根据实际 GPU、CUDA 和 PyTorch 环境选择兼容版本，不建议直接无条件安装最新版依赖。

### 3.2 模型配置

本次实验使用的模型：

| 模型 | 用途 |
|---|---|
| Qwen3-8B-Base | 本地大语言模型推理 |
| all-MiniLM-L6-v2 | 文本 Embedding |

模型权重不包含在本仓库中，需要自行下载或准备本地模型。

通过环境变量配置模型路径：

```bash
export HEBBIAN_QWEN_MODEL=/path/to/Qwen3-8B-Base

export HEBBIAN_EMBED_MODEL=/path/to/all-MiniLM-L6-v2

export HEBBIAN_LOCAL_QWEN=1
```

其中：

- `HEBBIAN_QWEN_MODEL`：Qwen 模型目录。
- `HEBBIAN_EMBED_MODEL`：Embedding 模型目录。
- `HEBBIAN_LOCAL_QWEN`：启用本地 Qwen 推理模式。

---

## 4. PersonaMem 数据准备

本项目不包含 PersonaMem 原始数据集，需要根据数据集的获取和使用要求自行准备。

预期的数据文件：

```text
personamem_data/
└── data/
    ├── questions_1M.csv
    └── shared_contexts_1M.jsonl
```

其中：

- `questions_1M.csv`：PersonaMem 问题及相关元数据。
- `shared_contexts_1M.jsonl`：共享对话上下文。

默认数据目录为：

```text
1m/data/personamem_data/
```

也可以通过环境变量指定实际数据目录：

```bash
export PERSONAMEM_DATA_ROOT=/path/to/personamem_data
```

如果需要分别指定文件：

```bash
export PERSONAMEM_QUESTION_PATH=/path/to/questions_1M.csv

export PERSONAMEM_CONTEXT_PATH=/path/to/shared_contexts_1M.jsonl
```

---

## 5. 实验参数

PersonaMem 不提供真实的对话时间戳，因此本复现中关闭时间衰减，避免人为引入时间排序信号。

主要实验参数包括：

| 参数 | 默认值 |
|---|---:|
| Learning Rate | 0.02 |
| Decay Rate | 0.995 |
| Activation Alpha | 0.1 |
| Spreading Threshold | 0.4 |
| Max Flipped | 3 |
| Keyword Weight | 0.7 |
| Knowledge Buffer Size | 10 |
| Time Decay | False |

对应环境变量：

```bash
export HEBBIAN_USE_TIME_DECAY=false

export HEBBIAN_USE_KEYWORD_MATCH=true

export HEBBIAN_LEARNING_RATE=0.02

export HEBBIAN_DECAY_RATE=0.995

export HEBBIAN_ACTIVATION_ALPHA=0.1

export HEBBIAN_SPREADING_THRESHOLD=0.4

export HEBBIAN_MAX_FLIPPED=3

export HEBBIAN_KEYWORD_WEIGHT=0.7

export HEBBIAN_KNOWLEDGE_BUFFER_SIZE=10
```

本地 Qwen 推理采用 vLLM，代码中的主要配置包括：

- Tensor Parallel Size：1
- GPU Memory Utilization：0.9
- Maximum Model Length：16384
- Prefix Caching：Enabled
- Guided Decoding Backend：Guidance

---

## 6. 共享前缀缓存机制

为了避免对具有相同历史对话前缀的多个问题重复执行完整记忆编码，本项目实现了 Shared-Prefix Cache。

### 6.1 基本思想

对于共享同一段历史对话的多个问题：

1. 按照对话截止位置对问题进行排序。
2. 找到最长的共享对话上下文。
3. 验证各个截止位置是否构成真实的前缀关系。
4. 按照对话顺序推进统一的记忆状态。
5. 对完整知识批次执行知识提取。
6. 对尚未完成的知识缓冲区，在独立快照副本上进行处理。
7. 为每个截止位置保存独立的记忆状态。

### 6.2 缓存隔离

每个截止位置保存独立的记忆快照，包括：

- `memory_hebbian.json`
- `memory_long_term.json`
- `memory_long_term_kb_graph.json`
- `metadata.json`
- `READY`

采用临时目录构建、JSON 文件验证和原子发布机制，避免不完整的缓存被评测程序读取。

### 6.3 缓存构建

运行前请先查看脚本参数：

```bash
python scripts/build_personamem_prefix_cache.py --help
```

根据实际数据路径、缓存目录及参数配置执行缓存构建。

### 6.4 缓存评测

查看评测参数：

```bash
python scripts/eval_personamem_cached.py --help
```

评测阶段通过对应问题的截止位置加载记忆快照，执行记忆检索和答案生成。

具体参数以脚本实际支持的命令行选项为准。

---

## 7. 测试

本项目提供三组测试。

### 7.1 结构化语义适配测试

```bash
python -m pytest tests/test_structured_semantic_adapter.py -v
```

### 7.2 结构化语义渲染测试

```bash
python -m pytest tests/test_structured_semantic_render.py -v
```

### 7.3 缓存等价性测试

```bash
python -m pytest tests/test_personamem_cache_equivalence.py -v
```

缓存等价性测试涉及记忆状态、对话处理及模型调用，需要提前准备相应的数据和模型环境。

---

## 8. 实验进度与结果审计

本项目已在 AutoDL GPU 环境完成 PersonaMem 1M 的一次完整问答评测运行。

根据本地实验审计记录：

| 审计项目 | 数量 |
|---|---:|
| 预期问题数 | 2674 |
| 数据集问题数 | 2674 |
| 评测结果数 | 2674 |
| 缺失问题数 | 0 |
| 重复问题数 | 0 |
| 错误记录数 | 0 |
| 已评分记录数 | 2674 |

上述数据说明本次运行完成了全部预期问题的结果生成，没有发现缺失、重复或错误记录。

**注意：** 完成率不等同于模型准确率。本 README 不报告未经独立核实的准确率或与论文结果的性能对比。完整结果文件和原始数据未随本次源码提交。

---

## 9. 复现注意事项

1. 本仓库不包含 Qwen 模型权重。
2. 本仓库不包含 Embedding 模型权重。
3. 本仓库不包含 PersonaMem 原始数据集。
4. 模型和数据路径需要根据实际运行环境配置。
5. 不同 CUDA、PyTorch 和 vLLM 版本可能影响运行兼容性。
6. 本地 Qwen 推理需要具备足够的 GPU 显存。
7. 建议先运行结构化语义测试，再执行完整缓存构建和问答评测。
8. 复现实验时应保留模型版本、依赖版本、实验参数及评测日志，以便进行结果核验。

---

## 10. 参考与致谢

本项目基于 HeLa-Mem 的公开实现进行 PersonaMem 适配和实验复现。

感谢原论文作者及开源项目贡献者提供的研究成果。

本目录属于独立的实验复现与适配实现，不代表原论文作者发布的官方 PersonaMem 实验结果。

