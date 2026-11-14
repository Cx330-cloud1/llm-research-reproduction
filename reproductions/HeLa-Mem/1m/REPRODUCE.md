# PersonaMem 1M：复现操作说明

本说明基于当前仓库的 `scripts/` 源码入口整理。它提供**可执行的命令格式**，但不表示本公开副本已经在全新环境完成端到端验证。

## 1. 依赖与模型

- Python 3.10 及以上（源项目声明）。
- Python 依赖见 [`requirements.txt`](requirements.txt)；测试工具见 [`requirements-dev.txt`](requirements-dev.txt)。其中 CUDA、PyTorch 与 vLLM 需要自行选择兼容组合，**现有文件未锁定复现用具体版本**。
- 需要本地模型 **Qwen3-8B-Base** 和 **all-MiniLM-L6-v2**。本仓库不发布权重。

从 `1m/` 目录配置环境变量：

```bash
export HEBBIAN_LOCAL_QWEN=1
export HEBBIAN_QWEN_MODEL=/your/path/Qwen3-8B-Base
export HEBBIAN_EMBED_MODEL=/your/path/all-MiniLM-L6-v2
export HEBBIAN_USE_TIME_DECAY=false
```

## 2. 外部 PersonaMem 数据与工具（必需）

`eval_personamem_local.py` **动态加载** `PERSONAMEM_DATA_ROOT/utils.py` 中的 `load_rows_with_context` 等函数。因此只准备 CSV/JSONL 还不够：必须取得兼容的 PersonaMem 数据转换工具代码，并按原数据集许可要求使用。

预期结构：

```text
personamem_data/
├── utils.py
└── data/
    ├── questions_1M.csv
    └── shared_contexts_1M.jsonl
```

使用自有路径：

```bash
export PERSONAMEM_DATA_ROOT=/your/path/personamem_data
# 如有必要，再分别覆盖：
# export PERSONAMEM_QUESTION_PATH=/your/path/questions_1M.csv
# export PERSONAMEM_CONTEXT_PATH=/your/path/shared_contexts_1M.jsonl
```

**注意：** 缓存构建脚本对当前数据规模有硬性检查：2674 个问题、276 个不同 cutoff、31 个 shared contexts；不同版本的数据可能无法直接运行。不要为通过检查而虚构记录。

## 3. 从项目根目录运行

进入 `reproductions/HeLa-Mem/1m`，确保 `hela_mem` 包和 `scripts` 中的辅助模块都可导入：

```bash
cd reproductions/HeLa-Mem/1m
export PYTHONPATH="$PWD:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
```

首先检查脚本入口和数据：

```bash
python scripts/build_personamem_prefix_cache.py --help
python scripts/eval_personamem_cached.py --help
python scripts/eval_personamem_cached.py --dry_run
```

注意 `--dry_run` 虽不执行问答，但仍需要满足脚本顶层导入的 Python 依赖和 PersonaMem 工具文件。

## 4. 构建共享前缀快照

缓存脚本已核实的参数：`--cache_root`、`--context_limit`、`--force`。`--context_limit 0` 表示处理全部共享上下文。

```bash
python scripts/build_personamem_prefix_cache.py   --cache_root personamem_cache_1M_prefix   --context_limit 0
```

缓存包含三份记忆 JSON、`metadata.json` 和 `READY`。中间结果只应留在私有实验环境，不要提交到公开仓库。`--force` 会重建缓存，**已有缓存不要在未备份的情况下强制覆盖**。

## 5. 读取缓存并评测

缓存评测脚本支持 `--phase {encode,qa,all}`、`--top_k`、`--semantic_top_k`、`--cache_root`、`--results_root`、`--output_prefix`、`--num_items`、`--force_rebuild`、`--dry_run`。

如已由共享前缀构建器生成兼容的缓存，可先尝试 QA 阶段：

```bash
python scripts/eval_personamem_cached.py   --phase qa   --cache_root personamem_cache_1M_prefix   --results_root personamem_results_1M_prefix   --output_prefix results_hela_personamem_1M_prefix   --top_k 15   --semantic_top_k 5
```

**重要限制：** 共享前缀构建器与缓存评测器的快照格式及复用行为尚未经过在全新环境下的端到端兼容性证明。建议先以隔离的小规模配置验证缓存加载、节点/知识量、cutoff 和问答日志，再运行全量。尤其不要直接把 `--num_items` 的小样本结果宣称为 1M 全量结果。

## 6. 测试与结果审计

```bash
python -m pytest tests/test_structured_semantic_adapter.py   tests/test_structured_semantic_render.py -v
```

`tests/test_personamem_cache_equivalence.py` 涉及完整的对话处理和模型依赖；运行前应阅读测试脚本，并确认本地数据、模型与执行方式。**目前没有通过完整等价性验证的公开证据。**

此前本地 `audit.json` 记录：2674 预期/实际/已评分，0 缺失、0 重复、0 错误行。这个审计不包含正确数的独立复核；在核实 `score` 字段评分标准与原始结果后才能填写 1M 的准确率。

## 7. 公开限制与已知风险

- 原始 PersonaMem 数据、RF 适配 `utils.py`、本地 Qwen 模型、Embedding 权重及个人缓存不在仓库中，**不是开箱即用的一键复现**。
- 当前 `requirements.txt` 未锁定经过全流程验证的依赖版本。
- 共享前缀缓存的性能收益不等于算法等价性，需额外的独立编码对照实验。
- `metrics.json` 中的 `null` 代表准确率未独立核实，而不是失败或零分。
