# Rapid State + Adaptive-k 无卡阶段 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不开启 4090/A800 的条件下完成可复现的数据清单、泄漏验证、Temperature Scaling、Adaptive-k 分配、状态/生成/评价脚本及全链路 mock 验证，使上卡阶段只负责真实模型推理。

**Architecture:** 所有新增逻辑保留在 `experiments/rapid_state_adaptive_k/`，不修改 EmoDynamiX 或 MultiAgentESC 核心代码。流水线通过版本化 JSONL 交接：本地构建样本与 sidecar，4090 只追加策略 logits，本地校准和分配预算，A800 只追加状态与回复，最后回到本地评价。

**Tech Stack:** Python 3.10+、标准库、NumPy 1.26+、pytest 8+、OpenAI-compatible API（仅后续 A800）、PyTorch/EmoDynamiX 环境（仅后续 4090）。

**Spec:** `docs/superpowers/specs/2026-09-23-rapid-state-adaptive-k-design.md`

## Global Constraints

- 所有新增代码必须位于 `experiments/rapid_state_adaptive_k/`；不得修改两篇复现项目的核心实现。
- 数据集固定为 `reproductions/MultiAgentESC/data/ESConv.json`，共 1300 个对话。
- 官方划分必须使用 Python `random.seed(13)` 等价逻辑：train 910、valid 195、test 195 个对话。
- 每个带策略标注的 supporter 轮独立成为目标；不得合并连续 supporter 轮。
- `dev_cal` 使用 valid 全部目标；`pilot100` 使用 test 每个对话的最后目标后按策略比例 seed=42 抽取 100 个不同对话；`smoke20` 是 pilot100 的 seed=42 分层子集。
- `model_context` 严格复现 EmoDynamiX 最近 5 个输入节点；`generation_context` 使用目标前全部可见历史。
- `gold_strategy` 与 `reference_response` 只能进入 `evaluation_sidecar.jsonl`，不得进入状态或生成任务。
- EmoDynamiX 只产生 8 类策略 raw logits；State Card 由 A800 上的同一 Qwen2.5-7B revision 产生。
- 校准温度和 Adaptive-k 阈值只能使用 `dev_cal`；`pilot100` 禁止参与拟合或调参。
- Qwen2.5-7B 参数固定：`temperature=0`、`max_tokens=400`、`response_word_limit=30`、`seed=42`。
- 失败最多重试 3 次；成功样本必须断点跳过；正式结果禁止 `--overwrite`。
- 未通过全部本地测试与 smoke20 mock 验证前，不得开启 4090 或 A800。
- 真实大体积输出、checkpoint、模型权重和 `reports/` 不提交 Git；配置、固定 manifest、脚本、测试和轻量汇总可以提交。

## Review Focus

- ESConv 中连续 supporter 轮必须生成两个独立目标；由 Task 2 的连续轮测试固定。
- 早期目标窗口可能包含 `<START>` 且长度不足 5；由 Task 2 的窗口边界测试固定。
- 不同样本可能拥有相同文本上下文；4090 join 必须检测重复 signature 并拒绝静默错配；由 Task 4 测试固定。
- Temperature Scaling 未改善 dev NLL 时必须冻结 `T=1.0`，不能强行使用劣化校准；由 Task 5 测试固定。
- F3 在小策略组或相同长度样本下仍不得自配对；由 Task 7 的极端 derangement 测试固定。

---

## File Map

| Path | Responsibility |
|---|---|
| `requirements-local.txt` | 无卡阶段最小依赖 |
| `.gitignore` | 排除真实输出、缓存和临时报告 |
| `configs/experiment.json` | 数据、划分、集合和标签顺序 |
| `configs/calibration.json` | 温度范围、ECE bins、阈值网格和预算约束 |
| `configs/generation.json` | Qwen revision、解码参数、Prompt 版本和重试 |
| `scripts/io_utils.py` | canonical JSON、SHA256、JSONL 原子写入与断点状态 |
| `scripts/build_manifests.py` | 官方划分、目标构造、分层抽样、base/sidecar 输出 |
| `scripts/validate_manifests.py` | 跨集合、泄漏、哈希、数量和确定性验证 |
| `scripts/export_emodynamix_logits.py` | 4090 资产预检、preprocessed join、raw logits 导出 |
| `scripts/calibrate_and_allocate.py` | Temperature Scaling、指标、阈值选择、K1/K2/KA/KR |
| `scripts/extract_state.py` | 四次冻结 Prompt、字段解析、State Card、断点恢复 |
| `scripts/build_generation_tasks.py` | F0–F3、长度匹配乱序、候选任务与 cache key |
| `scripts/generate_and_select.py` | Top-1/Top-2 候选池、匿名 selector、mock 模式 |
| `scripts/judge_responses.py` | 冻结辅助 judge rubric、严格 JSON 解析、断点恢复 |
| `scripts/evaluate_results.py` | 策略指标、预算、运行统计和人工 pilot 文件 |
| `scripts/run_local_checks.ps1` | Windows 无卡验收入口 |
| `tests/` | 每个纯函数、接口边界和离线链路测试 |

### Task 1: 冻结本地配置与确定性 I/O

**Files:**
- Create: `experiments/rapid_state_adaptive_k/requirements-local.txt`
- Create: `experiments/rapid_state_adaptive_k/.gitignore`
- Create: `experiments/rapid_state_adaptive_k/configs/experiment.json`
- Create: `experiments/rapid_state_adaptive_k/configs/calibration.json`
- Create: `experiments/rapid_state_adaptive_k/configs/generation.json`
- Create: `experiments/rapid_state_adaptive_k/scripts/__init__.py`
- Create: `experiments/rapid_state_adaptive_k/scripts/io_utils.py`
- Create: `experiments/rapid_state_adaptive_k/tests/test_io_utils.py`
- Modify: `experiments/rapid_state_adaptive_k/README.md`

**Interfaces:**
- Consumes: UTF-8 dictionaries and JSONL rows。
- Produces: `canonical_json(value) -> str`、`sha256_json(value) -> str`、`read_jsonl(path) -> list[dict]`、`write_jsonl_atomic(path, rows) -> None`、`append_jsonl(path, row) -> None`、`successful_ids(path) -> set[str]`。

- [ ] **Step 1: Write the failing deterministic-I/O tests**

```python
from pathlib import Path

from scripts.io_utils import read_jsonl, sha256_json, successful_ids, write_jsonl_atomic


def test_sha256_json_ignores_dict_insertion_order():
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def test_jsonl_round_trip_and_success_filter(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    rows = [
        {"sample_id": "s1", "status": "ok", "text": "中文"},
        {"sample_id": "s2", "status": "error", "error": "boom"},
    ]
    write_jsonl_atomic(path, rows)
    assert read_jsonl(path) == rows
    assert successful_ids(path) == {"s1"}
```

- [ ] **Step 2: Run the test and verify the missing module failure**

Run from `experiments/rapid_state_adaptive_k`:

```powershell
python -m pytest tests/test_io_utils.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.io_utils'`.

- [ ] **Step 3: Add the minimal dependency and configuration files**

`requirements-local.txt`:

```text
numpy>=1.26,<3
pytest>=8,<9
openai>=1.47,<2
```

`configs/experiment.json` must contain the exact label order used by EmoDynamiX:

```json
{
  "schema_version": "1.0",
  "dataset": "../../../reproductions/MultiAgentESC/data/ESConv.json",
  "official_split_seed": 13,
  "sampling_seed": 42,
  "pilot_size": 100,
  "smoke_size": 20,
  "label_order": [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others"
  ]
}
```

`configs/calibration.json`:

```json
{
  "temperature_min": 0.05,
  "temperature_max": 10.0,
  "temperature_grid_size": 2001,
  "ece_bins": 10,
  "probability_thresholds": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
  "margin_thresholds": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
  "max_top2_recall_drop": 0.01,
  "minimum_candidate_saving": 0.20,
  "allocation_seed": 42
}
```

`configs/generation.json`:

```json
{
  "model_revision": "qwen2.5:7b",
  "temperature": 0.0,
  "max_tokens": 400,
  "response_word_limit": 30,
  "seed": 42,
  "retries": 3,
  "state_prompt_version": "state-v1",
  "response_prompt_version": "response-v1",
  "selector_prompt_version": "selector-v1"
}
```

`.gitignore`:

```gitignore
reports/
__pycache__/
*.py[cod]
.pytest_cache/
```

- [ ] **Step 4: Implement canonical JSON and atomic JSONL writes**

```python
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(row) + "\n")


def successful_ids(path: Path) -> set[str]:
    return {
        str(row["sample_id"])
        for row in read_jsonl(path)
        if row.get("status") == "ok" and "sample_id" in row
    }
```

- [ ] **Step 5: Correct README compute ownership**

Replace the README statement that the 4090 exports state fields with:

```markdown
### RTX 4090 租赁实例

只负责加载已有 EmoDynamiX checkpoint，并导出固定样本的 8 类原始 strategy logits。EmoDynamiX 不生成 State Card。

### A800 租赁实例

负责使用同一 Qwen2.5-7B revision 依次提取 Emotion、Cause、Intention、Support Need，构造 State Card，生成 Top-1/Top-2 候选并运行固定 selector 与辅助 judge。
```

- [ ] **Step 6: Run tests and commit**

```powershell
python -m pytest tests/test_io_utils.py -v
git add experiments/rapid_state_adaptive_k
git commit -m "chore: freeze adaptive-k local configs"
```

Expected: all Task 1 tests PASS.

### Task 2: 重建官方划分和无泄漏 manifests

**Files:**
- Create: `experiments/rapid_state_adaptive_k/scripts/build_manifests.py`
- Create: `experiments/rapid_state_adaptive_k/tests/test_build_manifests.py`

**Interfaces:**
- Consumes: `ESConv.json` list and `configs/experiment.json`。
- Produces: `official_split_indices(dialogues, seed) -> dict[str, list[int]]`、`extract_strategy_targets(dialogue, dialogue_index, split, label_order) -> list[tuple[dict, dict]]`、`stratified_sample(rows, size, seed) -> list[dict]` and collection files under `manifests/{dev_cal,pilot100,smoke20}/`。

- [ ] **Step 1: Write failing tests for split sizes, consecutive turns, and early windows**

```python
from scripts.build_manifests import extract_strategy_targets, official_split_indices


def _dialogue(index: int):
    return {
        "dialog": [
            {"speaker": "seeker", "content": f"u{index}", "annotation": {}},
            {"speaker": "supporter", "content": "a1", "annotation": {"strategy": "Question"}},
            {"speaker": "supporter", "content": "a2", "annotation": {"strategy": "Information"}},
        ]
    }


def test_official_dialogue_split_is_910_195_195_and_disjoint():
    split = official_split_indices([_dialogue(i) for i in range(1300)], seed=13)
    assert {name: len(ids) for name, ids in split.items()} == {
        "train": 910,
        "valid": 195,
        "test": 195,
    }
    assert not (set(split["train"]) & set(split["valid"]))
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["valid"]) & set(split["test"]))


def test_consecutive_supporter_turns_are_separate_targets():
    pairs = extract_strategy_targets(_dialogue(0), 0, "valid", ["Question", "Information"])
    assert [base["target_turn_id"] for base, _ in pairs] == ["u001", "u002"]
    assert [side["gold_strategy"] for _, side in pairs] == ["Question", "Information"]


def test_early_model_window_contains_start_but_not_target():
    first, _ = extract_strategy_targets(_dialogue(0), 0, "valid", ["Question", "Information"])[0]
    assert first["model_context"]["dialogue_history"] == "<START> </s> u0"
    assert "a1" not in first["model_context"]["dialogue_history"]
```

- [ ] **Step 2: Run tests and confirm failure**

```powershell
python -m pytest tests/test_build_manifests.py -v
```

Expected: FAIL because the manifest builder does not exist.

- [ ] **Step 3: Implement the official split and exact target window**

```python
def official_split_indices(dialogues, seed=13):
    indexed = list(range(len(dialogues)))
    random.Random(seed).shuffle(indexed)
    dev_size = int(0.15 * len(indexed))
    test_size = int(0.15 * len(indexed))
    return {
        "valid": indexed[:dev_size],
        "test": indexed[dev_size:dev_size + test_size],
        "train": indexed[dev_size + test_size:],
    }


def extract_strategy_targets(dialogue, dialogue_index, split, label_order):
    turns = dialogue["dialog"]
    utterances = ["<START>"]
    speakers = ["None"]
    strategies = [-1]
    pairs = []
    for raw_index, turn in enumerate(turns):
        text = normalize_text(turn.get("content", ""))
        annotation = turn.get("annotation") or {}
        strategy = annotation.get("strategy")
        if turn.get("speaker") == "supporter" and strategy:
            if strategy not in label_order:
                raise ValueError(f"unknown strategy: {strategy}")
            model_index = raw_index + 1
            start = max(0, model_index - 5)
            model_context = {
                "dialogue_history": " </s> ".join(utterances[start:model_index]),
                "strategy_history": str(strategies[start:model_index]),
                "speaker_turn": " ".join(speakers[start:model_index]),
            }
            generation_turns = turns[:raw_index]
            generation_context = "\n".join(
                f"{'User' if item['speaker'] == 'seeker' else 'Assistant'}: {normalize_text(item.get('content', ''))}"
                for item in generation_turns
            )
            sample_id = f"{split}-d{dialogue_index:04d}-u{raw_index:03d}"
            base_without_hash = {
                "schema_version": "1.0",
                "sample_id": sample_id,
                "split": split,
                "dialogue_id": f"d{dialogue_index:04d}",
                "target_turn_id": f"u{raw_index:03d}",
                "visible_turn_ids": [f"u{i:03d}" for i in range(raw_index)],
                "model_context": model_context,
                "generation_context": generation_context,
            }
            base = {**base_without_hash, "context_hash": sha256_json(base_without_hash)}
            sidecar = {
                "sample_id": sample_id,
                "gold_strategy": strategy,
                "reference_response": text,
                "original_dialogue_index": dialogue_index,
                "target_turn_index": raw_index,
            }
            pairs.append((base, sidecar))
        utterances.append(text)
        speakers.append(str(turn.get("speaker", "")))
        strategies.append(label_order.index(strategy) if strategy in label_order else -1)
    return pairs
```

- [ ] **Step 4: Implement deterministic proportional sampling**

Use largest remainders for class quotas, then a class-local seeded shuffle; sort the final rows by `sample_id` before writing. `pilot100` candidates are the maximum `target_turn_index` row per test dialogue. `smoke20` is sampled from the completed pilot100 rows, not independently from test.

```python
def proportional_quotas(counts: dict[str, int], size: int) -> dict[str, int]:
    total = sum(counts.values())
    exact = {key: counts[key] * size / total for key in counts}
    quotas = {key: min(counts[key], int(exact[key])) for key in counts}
    remaining = size - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - int(exact[key])), key))
    for key in order:
        if remaining == 0:
            break
        if quotas[key] < counts[key]:
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise ValueError("unable to allocate stratified quotas")
    return quotas
```

- [ ] **Step 5: Write the three collections and verify the real dataset**

Run:

```powershell
python -m scripts.build_manifests `
    --config .\configs\experiment.json `
    --output-root .\manifests

python -m pytest tests/test_build_manifests.py -v
```

Expected summary must include `train_dialogues=910`, `valid_dialogues=195`, `test_dialogues=195`, `pilot100=100`, and `smoke20=20`.

- [ ] **Step 6: Commit**

```powershell
git add experiments/rapid_state_adaptive_k/scripts/build_manifests.py `
        experiments/rapid_state_adaptive_k/tests/test_build_manifests.py `
        experiments/rapid_state_adaptive_k/manifests
git commit -m "feat: build official adaptive-k manifests"
```

### Task 3: 自动验证集合、泄漏和可重复性

**Files:**
- Create: `experiments/rapid_state_adaptive_k/scripts/validate_manifests.py`
- Create: `experiments/rapid_state_adaptive_k/tests/test_validate_manifests.py`

**Interfaces:**
- Consumes: three `base_manifest.jsonl` files and three `evaluation_sidecar.jsonl` files。
- Produces: `validate_collection(base_rows, sidecar_rows) -> list[str]`、`validate_all(root) -> None`; invalid input raises `ValueError` with every detected violation。

- [ ] **Step 1: Write failing leakage and cross-split tests**

```python
import pytest

from scripts.validate_manifests import validate_all_rows, validate_collection


def test_gold_fields_are_rejected_from_base_manifest():
    base = [{
        "sample_id": "s1",
        "dialogue_id": "d1",
        "target_turn_id": "u002",
        "visible_turn_ids": ["u000", "u001"],
        "context_hash": "x",
        "gold_strategy": "Question",
    }]
    with pytest.raises(ValueError, match="forbidden field: gold_strategy"):
        validate_collection(base, [{"sample_id": "s1"}])


def test_pilot_and_dev_dialogues_must_be_disjoint():
    with pytest.raises(ValueError, match="dev_cal/pilot100 overlap"):
        validate_all_rows(
            dev=[{"sample_id": "v1", "dialogue_id": "d1"}],
            pilot=[{"sample_id": "t1", "dialogue_id": "d1"}],
            smoke=[],
        )
```

- [ ] **Step 2: Run tests and confirm failure**

```powershell
python -m pytest tests/test_validate_manifests.py -v
```

- [ ] **Step 3: Implement validation with accumulated errors**

The validator must check the ten invariants in spec section 12. Use numeric parsing of `uNNN` to prove every visible turn precedes the target, compare base/sidecar sample ID sets exactly, recursively reject `gold_strategy`, `reference_response`, and target-text keys from base rows, and reject duplicate `sample_id` or `context_hash`.

```python
FORBIDDEN_BASE_FIELDS = {"gold_strategy", "reference_response", "target_response", "future_turns"}


def _keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _keys(child)


def validate_collection(base_rows, sidecar_rows):
    errors = []
    base_ids = [str(row.get("sample_id")) for row in base_rows]
    side_ids = [str(row.get("sample_id")) for row in sidecar_rows]
    if len(base_ids) != len(set(base_ids)):
        errors.append("duplicate sample_id")
    if set(base_ids) != set(side_ids):
        errors.append("base/sidecar sample_id mismatch")
    hashes = [row.get("context_hash") for row in base_rows]
    if len(hashes) != len(set(hashes)):
        errors.append("duplicate context_hash")
    for row in base_rows:
        for key in sorted(FORBIDDEN_BASE_FIELDS & set(_keys(row))):
            errors.append(f"forbidden field: {key}")
        target = int(str(row["target_turn_id"])[1:])
        if any(int(str(turn_id)[1:]) >= target for turn_id in row["visible_turn_ids"]):
            errors.append(f"future turn visible: {row['sample_id']}")
    if errors:
        raise ValueError("; ".join(errors))
```

- [ ] **Step 4: Add deterministic rebuild verification**

`validate_manifests.py --rebuild-check` must build into two separate temporary directories and compare SHA256 for all six JSONL files. It must never overwrite the checked-in manifests.

- [ ] **Step 5: Run validation and commit**

```powershell
python -m scripts.validate_manifests --root .\manifests --rebuild-check
python -m pytest tests/test_build_manifests.py tests/test_validate_manifests.py -v
git add experiments/rapid_state_adaptive_k
git commit -m "test: enforce manifest isolation and determinism"
```

### Task 4: 准备 4090 logits 导出器，但不启动 GPU

**Files:**
- Create: `experiments/rapid_state_adaptive_k/scripts/export_emodynamix_logits.py`
- Create: `experiments/rapid_state_adaptive_k/tests/test_export_emodynamix_logits.py`

**Interfaces:**
- Consumes: base manifests, server-side `valid.pkl`/`test.pkl`, `strategies.json`, checkpoint 2600。
- Produces: `context_signature(model_context) -> str`、`join_preprocessed(base_rows, preprocessed_rows) -> list[tuple[dict, dict]]` and later `strategy_results.raw.jsonl` rows with exactly eight raw logits and verified `label_order`。

- [ ] **Step 1: Write failing join and duplicate-signature tests**

```python
import pytest

from scripts.export_emodynamix_logits import context_signature, join_preprocessed


MODEL_CONTEXT = {
    "dialogue_history": "<START> </s> hello",
    "strategy_history": "[-1, -1]",
    "speaker_turn": "None seeker",
}


def test_join_uses_all_three_model_inputs():
    base = [{"sample_id": "s1", "model_context": MODEL_CONTEXT}]
    prepared = [{**MODEL_CONTEXT, "parsed_dialogue": [], "erc_logits": [[0.0] * 7]}]
    joined = join_preprocessed(base, prepared)
    assert joined[0][0]["sample_id"] == "s1"


def test_duplicate_preprocessed_signature_fails_closed():
    base = [{"sample_id": "s1", "model_context": MODEL_CONTEXT}]
    prepared = [{**MODEL_CONTEXT}, {**MODEL_CONTEXT}]
    with pytest.raises(ValueError, match="duplicate preprocessed signature"):
        join_preprocessed(base, prepared)
```

- [ ] **Step 2: Run tests and confirm failure**

```powershell
python -m pytest tests/test_export_emodynamix_logits.py -v
```

- [ ] **Step 3: Implement pure join and label-order validation**

```python
def context_signature(model_context):
    return sha256_json({
        "dialogue_history": model_context["dialogue_history"],
        "strategy_history": str(model_context["strategy_history"]),
        "speaker_turn": str(model_context["speaker_turn"]),
    })


def verified_label_order(strategy2id, expected):
    actual = [label for label, _ in sorted(strategy2id.items(), key=lambda item: item[1])]
    if actual != expected:
        raise ValueError(f"label order mismatch: expected={expected}, actual={actual}")
    return actual
```

- [ ] **Step 4: Add lazy GPU-only imports and an asset preflight**

The module top level must import only the standard library and `io_utils`; import `torch`, EmoDynamiX classes, and pickle payloads inside `main()`. Before model construction, verify all exact paths and print a JSON preflight record containing checkpoint SHA256, strategy mapping SHA256, pkl SHA256, manifest SHA256, CUDA availability, and device name. Missing assets or `cuda=False` must exit before model loading.

For every batch, call `outputs = model(batch)` and export `outputs["logits"].detach().cpu().tolist()`. Do not call softmax in the model. Each row must be:

```python
{
    "sample_id": sample_id,
    "strategy_logits": logits,
    "label_order": label_order,
    "status": "ok",
    "checkpoint_sha256": checkpoint_sha256,
}
```

- [ ] **Step 5: Test the CPU-safe module surface and commit**

```powershell
python -m pytest tests/test_export_emodynamix_logits.py -v
python -m scripts.export_emodynamix_logits --help
git add experiments/rapid_state_adaptive_k
git commit -m "feat: prepare emodynamix logits export"
```

Expected: help and unit tests run without importing torch or requiring CUDA.

### Task 5: Temperature Scaling 和 K1/K2/KA/KR 分配

**Files:**
- Create: `experiments/rapid_state_adaptive_k/scripts/calibrate_and_allocate.py`
- Create: `experiments/rapid_state_adaptive_k/tests/test_calibration.py`
- Create: `experiments/rapid_state_adaptive_k/tests/fixtures/strategy_results.jsonl`

**Interfaces:**
- Consumes: dev raw logits + dev sidecar, pilot raw logits, `calibration.json`。
- Produces: `softmax(logits, temperature) -> ndarray`、`fit_temperature(logits, labels, config) -> dict`、`select_thresholds(probabilities, labels, config) -> dict`、`allocate_policies(rows, thresholds, seed) -> list[dict]`, frozen `calibration_report.json` and calibrated strategy JSONL files。

- [ ] **Step 1: Write failing calibration and fallback tests**

```python
import numpy as np

from scripts.calibrate_and_allocate import fit_temperature, policy_allocations


def test_temperature_falls_back_to_one_when_nll_does_not_improve():
    logits = np.array([[10.0, 0.0], [0.0, 10.0]])
    labels = np.array([0, 1])
    result = fit_temperature(logits, labels, 0.05, 10.0, 101)
    assert result["temperature"] == 1.0
    assert result["used_calibration"] is False


def test_random_budget_has_exactly_same_number_of_k2_rows_as_adaptive():
    adaptive_k = [1, 2, 1, 2, 2]
    policies = policy_allocations(adaptive_k, seed=42)
    assert policies["KR"].count(2) == policies["KA"].count(2)
    assert policies["K1"] == [1] * 5
    assert policies["K2"] == [2] * 5
```

- [ ] **Step 2: Run tests and confirm failure**

```powershell
python -m pytest tests/test_calibration.py -v
```

- [ ] **Step 3: Implement numerically stable metrics and temperature search**

```python
def softmax(logits, temperature=1.0):
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def nll(probabilities, labels):
    chosen = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean())


def fit_temperature(logits, labels, lower, upper, grid_size):
    raw = softmax(logits, 1.0)
    raw_nll = nll(raw, labels)
    temperatures = np.exp(np.linspace(np.log(lower), np.log(upper), grid_size))
    scored = [(nll(softmax(logits, value), labels), float(value)) for value in temperatures]
    best_nll, best_temperature = min(scored, key=lambda item: (item[0], abs(item[1] - 1.0)))
    if best_nll >= raw_nll - 1e-12:
        return {"temperature": 1.0, "raw_nll": raw_nll, "calibrated_nll": raw_nll, "used_calibration": False}
    return {"temperature": best_temperature, "raw_nll": raw_nll, "calibrated_nll": best_nll, "used_calibration": True}
```

Also implement multiclass Brier score, fixed-bin ECE, Top-1 accuracy, macro recall, and Top-2 recall using only NumPy.

- [ ] **Step 4: Implement frozen threshold selection and exact-budget KR**

For each probability/margin threshold pair, use `k=1` only when both conditions pass. First retain pairs with recall at least `top2_recall - max_top2_recall_drop` and saving at least `minimum_candidate_saving`; choose the retained pair with lowest mean k, then highest recall, then highest thresholds. If none qualify, choose highest recall then highest saving and set `gate_met=false` in the report.

KR must choose exactly the same number of k=2 rows as KA by shuffling row indices with `random.Random(seed)`; no Bernoulli approximation is allowed.

- [ ] **Step 5: Run tests, create frozen local reports, and commit**

Before real logits exist, use a checked-in eight-class fixture under `tests/fixtures/strategy_results.jsonl`; write mock-derived outputs only under `reports/mock-not-for-reporting/`. The fixture contains four deliberately simple rows with eight logits each:

```jsonl
{"sample_id":"v1","strategy_logits":[4.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],"label_order":["Question","Restatement or Paraphrasing","Reflection of feelings","Self-disclosure","Affirmation and Reassurance","Providing Suggestions","Information","Others"],"gold_label_index":0}
{"sample_id":"v2","strategy_logits":[1.0,3.0,0.0,0.0,0.0,0.0,0.0,0.0],"label_order":["Question","Restatement or Paraphrasing","Reflection of feelings","Self-disclosure","Affirmation and Reassurance","Providing Suggestions","Information","Others"],"gold_label_index":1}
{"sample_id":"v3","strategy_logits":[2.0,1.9,0.0,0.0,0.0,0.0,0.0,0.0],"label_order":["Question","Restatement or Paraphrasing","Reflection of feelings","Self-disclosure","Affirmation and Reassurance","Providing Suggestions","Information","Others"],"gold_label_index":1}
{"sample_id":"v4","strategy_logits":[0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0],"label_order":["Question","Restatement or Paraphrasing","Reflection of feelings","Self-disclosure","Affirmation and Reassurance","Providing Suggestions","Information","Others"],"gold_label_index":7}
```

```powershell
python -m pytest tests/test_calibration.py -v
python -m scripts.calibrate_and_allocate --self-test
git add experiments/rapid_state_adaptive_k
git commit -m "feat: add calibrated adaptive-k allocation"
```

### Task 6: 准备四字段状态提取并完成 mock 断点测试

**Files:**
- Create: `experiments/rapid_state_adaptive_k/scripts/extract_state.py`
- Create: `experiments/rapid_state_adaptive_k/tests/test_state_extraction.py`

**Interfaces:**
- Consumes: base manifest and an object exposing `complete(prompt, tag) -> str | CallResult`。
- Produces: `extract_state_fields(context, client) -> dict[str, str]`、`make_state_card(fields) -> str` and resumable `state_results.jsonl`。

- [ ] **Step 1: Write failing field, dependency, and resume tests**

```python
from scripts.extract_state import extract_state_fields, make_state_card


class RecordingClient:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt, tag):
        self.prompts.append((tag, prompt))
        values = {
            "state_emotion": "Emotion: anxious",
            "state_cause": "Cause: uncertainty about work",
            "state_intention": "Intention: decide what to do next",
            "state_support_need": "Support Need: validation before advice",
        }
        return values[tag]


def test_support_need_receives_the_first_three_fields():
    client = RecordingClient()
    fields = extract_state_fields("User: I feel stuck", client)
    support_prompt = client.prompts[-1][1]
    assert fields["emotion"] == "anxious"
    assert "uncertainty about work" in support_prompt
    assert "decide what to do next" in support_prompt


def test_state_card_is_deterministic():
    fields = {"emotion": "sad", "cause": "loss", "intention": "cope", "support_need": "validation"}
    assert make_state_card(fields) == "Emotion: sad; Cause: loss; Intention: cope; Support need: validation."
```

- [ ] **Step 2: Run tests and confirm failure**

```powershell
python -m pytest tests/test_state_extraction.py -v
```

- [ ] **Step 3: Implement four frozen prompts and strict parsing**

Each prompt must request one single-line labeled field. `Cause` replaces the upstream wording `Event`; Support Need receives the already parsed three values. `parse_field(text, label)` must reject missing or empty labels instead of silently storing the entire response.

Normalize both production and mock clients at the call boundary:

```python
def completion_text(client, prompt, tag):
    result = client.complete(prompt, tag=tag)
    return result.text if hasattr(result, "text") else str(result)
```

The output row must include at least:

```python
{
    "sample_id": sample_id,
    "emotion": fields["emotion"],
    "cause": fields["cause"],
    "intention": fields["intention"],
    "support_need": fields["support_need"],
    "state_card": make_state_card(fields),
    "model_revision": config["model_revision"],
    "prompt_hash": sha256_json(STATE_PROMPTS),
    "status": "ok",
    "attempts": attempt,
    "error": None,
}
```

Before writing the row, compute a state-stage cache key from `data_hash`, `sample_id`, `model_revision`, `prompt_hash`, `decoding_hash`, `condition_id="state"`, and `stage="state_extraction"`; store its SHA256 as `cache_key`. Resume checks must require both matching `sample_id` and matching `cache_key`, so stale outputs from a changed prompt are not skipped.

- [ ] **Step 4: Implement retry, append-after-sample, and mock isolation**

At startup, load `successful_ids(output)` and skip them. Retry the complete four-call sample at most three times. Append one final error row after the third failure. `--mock` must force the output path under `reports/mock-not-for-reporting/` and set `is_mock=true`.

- [ ] **Step 5: Run twice to prove resume and commit**

```powershell
python -m pytest tests/test_state_extraction.py -v
python -m scripts.extract_state --manifest .\manifests\smoke20\base_manifest.jsonl --mock
python -m scripts.extract_state --manifest .\manifests\smoke20\base_manifest.jsonl --mock
git add experiments/rapid_state_adaptive_k
git commit -m "feat: add resumable four-field state extraction"
```

Expected: the second run reports 20 skipped successful samples and appends zero rows.

### Task 7: 构建 F0–F3 与长度匹配 F3 对照

**Files:**
- Create: `experiments/rapid_state_adaptive_k/scripts/build_generation_tasks.py`
- Create: `experiments/rapid_state_adaptive_k/tests/test_generation_tasks.py`

**Interfaces:**
- Consumes: base, calibrated strategy, allocation, and state JSONL files joined by `sample_id`。
- Produces: `length_matched_derangement(rows, seed, attempts) -> dict[str, str]`、`render_state(condition, own_state, shuffled_state) -> str`、two candidate-task rows per sample/condition with immutable cache keys。

- [ ] **Step 1: Write failing F0, leakage, and derangement tests**

```python
from scripts.build_generation_tasks import build_tasks, length_matched_derangement


def test_derangement_never_self_pairs_even_with_equal_lengths():
    rows = [{"sample_id": f"s{i}", "state_card": "same length"} for i in range(6)]
    mapping = length_matched_derangement(rows, seed=42, attempts=200)
    assert set(mapping) == {row["sample_id"] for row in rows}
    assert all(source != target for source, target in mapping.items())


def test_f0_has_empty_state_and_generation_tasks_have_no_gold_fields():
    tasks = build_tasks(base_rows(), strategy_rows(), state_rows(), seed=42)
    f0 = [row for row in tasks if row["condition_id"] == "F0"]
    assert all(row["state_input"] == "" for row in f0)
    assert all("gold_strategy" not in row and "reference_response" not in row for row in tasks)
```

- [ ] **Step 2: Run tests and confirm failure**

```powershell
python -m pytest tests/test_generation_tasks.py -v
```

- [ ] **Step 3: Implement deterministic minimum-cost derangement**

Generate `attempts` seeded random permutations, discard any with fixed points, score the rest by total absolute difference in State Card word counts, and select the minimum `(cost, permutation_sample_ids)` tuple. Require at least two rows; otherwise raise `ValueError("F3 requires at least two samples")`.

- [ ] **Step 4: Build immutable tasks and cache keys**

For every sample and F0–F3 condition, write rank-1 and rank-2 tasks using `top1_strategy` and `top2_strategy`. The common response template must be byte-identical outside the `### State` block. Cache key input must contain:

```python
{
    "data_hash": base["context_hash"],
    "sample_id": sample_id,
    "model_revision": generation_config["model_revision"],
    "prompt_hash": sha256_json(prompt),
    "decoding_hash": sha256_json(decoding_parameters),
    "condition_id": condition_id,
    "stage": "candidate_generation",
    "strategy_rank": rank,
}
```

Validate that the logits and label arrays both have length 8 and that Top-1 differs from Top-2.

- [ ] **Step 5: Run tests, build mock tasks, and commit**

```powershell
python -m pytest tests/test_generation_tasks.py -v
python -m scripts.build_generation_tasks --collection smoke20 --mock-inputs
git add experiments/rapid_state_adaptive_k
git commit -m "feat: build controlled state and candidate tasks"
```

### Task 8: 准备统一候选池和固定 selector

**Files:**
- Create: `experiments/rapid_state_adaptive_k/scripts/generate_and_select.py`
- Create: `experiments/rapid_state_adaptive_k/scripts/judge_responses.py`
- Create: `experiments/rapid_state_adaptive_k/tests/test_generate_and_select.py`

**Interfaces:**
- Consumes: generation tasks and a completion client。
- Produces: append-only `candidate_results.jsonl` and `selector_results.jsonl`; `selector_prompt(context, state_card, candidates, seed_key) -> tuple[str, dict[str, str]]`。

- [ ] **Step 1: Write failing selector-blinding tests**

```python
from scripts.generate_and_select import selector_prompt


def test_selector_is_blinded_to_strategy_origin_and_probability():
    prompt, order = selector_prompt(
        context="User: help",
        state_card="Emotion: sad",
        candidates={"rank1": "response one", "rank2": "response two"},
        seed_key="s1-F2",
    )
    assert set(order) == {"A", "B"}
    assert "response one" in prompt and "response two" in prompt
    forbidden = ["top-1", "top-2", "probability", "gold", "rank1", "rank2"]
    assert all(token not in prompt.lower() for token in forbidden)


def test_judge_parser_requires_every_frozen_metric():
    from scripts.judge_responses import parse_judgement

    parsed = parse_judgement(
        '{"empathy":4,"relevance":5,"helpfulness":4,'
        '"premature_advice":false,"safety_issue":false,"reason":"grounded"}'
    )
    assert parsed["empathy"] == 4
    assert parsed["safety_issue"] is False
```

- [ ] **Step 2: Run tests and confirm failure**

```powershell
python -m pytest tests/test_generate_and_select.py -v
```

- [ ] **Step 3: Implement candidate generation and word-limit audit**

Generate each unique `(sample_id, condition_id, strategy_rank)` exactly once. Store response text, prompt/completion tokens, latency, prompt hash, model revision, status, attempts, error, and `word_count`. Record `word_limit_passed = word_count <= 30`; do not silently truncate model output.

- [ ] **Step 4: Implement deterministic anonymous A/B selector**

Derive left/right order from SHA256 of `seed + seed_key`, not global process order. The selector prompt may contain only generation context, the condition's current state input, Candidate A, and Candidate B. Parse exactly `Choice: A` or `Choice: B`; reject any other output and retry up to three times.

K1 reads rank 1 directly. K2 always uses selector. KA uses selector only when allocation is k=2. KR uses selector only on its frozen k=2 rows. All four policies must reference the same candidate row IDs.

- [ ] **Step 5: Implement the frozen auxiliary judge**

`judge_responses.py` must show the judge only the dialogue context and one anonymous response. Require one JSON object with integer `empathy`, `relevance`, and `helpfulness` in 1–5, booleans `premature_advice` and `safety_issue`, and a short `reason`. Reject missing keys, out-of-range scores, markdown fences, and extra prose. Store model revision, rubric hash, tokens, latency, attempts, and status. The report must call these results auxiliary LLM-judge trends, never human evidence.

- [ ] **Step 6: Run the entire smoke20 path in mock mode twice and commit**

```powershell
python -m pytest tests/test_generate_and_select.py -v
python -m scripts.generate_and_select --collection smoke20 --mock
python -m scripts.generate_and_select --collection smoke20 --mock
python -m scripts.judge_responses --collection smoke20 --mock
python -m scripts.judge_responses --collection smoke20 --mock
git add experiments/rapid_state_adaptive_k
git commit -m "feat: add shared candidate pool and blinded selector"
```

Expected: second run performs zero new mock completions.

### Task 9: 本地评价、人工 pilot 和决策门

**Files:**
- Create: `experiments/rapid_state_adaptive_k/scripts/evaluate_results.py`
- Create: `experiments/rapid_state_adaptive_k/tests/test_evaluate_results.py`

**Interfaces:**
- Consumes: sidecar, strategy results, allocations, candidate, selector and judge results, optional human ratings CSV。
- Produces: `reports/strategy_metrics.json`、`reports/runtime_metrics.json`、`reports/human_pilot_blinded.csv`、`reports/decision_gates.json`。

- [ ] **Step 1: Write failing budget and human-pilot tests**

```python
from scripts.evaluate_results import candidate_saving, summarize_pairwise


def test_candidate_saving_is_relative_to_k2():
    assert candidate_saving([1, 1, 2, 2]) == 0.25


def test_pairwise_summary_keeps_ties_separate():
    result = summarize_pairwise(["left", "right", "tie", "left"])
    assert result == {"left": 2, "right": 1, "tie": 1, "tie_adjusted_left_rate": 0.625}
```

- [ ] **Step 2: Run tests and confirm failure**

```powershell
python -m pytest tests/test_evaluate_results.py -v
```

- [ ] **Step 3: Implement strategy and runtime summaries**

Compute Top-1 accuracy, macro recall over all 8 labels, Top-2 recall, raw/calibrated NLL, Brier, ECE, temperature, KA/KR k distributions, mean candidates, and saving `1 - mean_k / 2`. Runtime summary must separately report cold and cached runs, tokens, latency, errors, retries, word-limit violations, and selector rank-2 selection rate.

Response summary must report auxiliary judge means for Empathy, Relevance, and Helpfulness; rates for Premature Advice and Safety issues; deterministic token-set Jaccard similarity between rank-1/rank-2 responses as the semantic-repetition proxy; and paired per-sample deltas for F0 vs F2 and K1 vs KA. Missing judge rows must reduce the reported denominator rather than being imputed.

- [ ] **Step 4: Build a deterministic 30-item blinded human sheet**

Sample 30 pilot IDs with seed 42. Create two rows per sample: F0 vs F2 and K1 vs KA. Randomize left/right deterministically and include columns `pair_id`, `sample_id`, `comparison`, `context`, `response_left`, `response_right`, `rating`, `notes`; omit condition names from response columns. Accept only `left`, `right`, or `tie` when importing ratings.

- [ ] **Step 5: Encode decision gates without overstating evidence**

`decision_gates.json` must use `pass`, `fail`, or `insufficient_data` for each spec gate. If fewer than three raters exist per row, label the human result `single-rater formative pilot` or `incomplete formative pilot`; never label it formal human evaluation.

- [ ] **Step 6: Run tests and commit**

```powershell
python -m pytest tests/test_evaluate_results.py -v
git add experiments/rapid_state_adaptive_k
git commit -m "feat: add local evaluation and decision gates"
```

### Task 10: 建立无卡总验收脚本和上卡准入记录

**Files:**
- Create: `experiments/rapid_state_adaptive_k/scripts/run_local_checks.ps1`
- Create: `experiments/rapid_state_adaptive_k/tests/test_cli_contracts.py`
- Modify: `experiments/rapid_state_adaptive_k/README.md`

**Interfaces:**
- Consumes: repository checkout with ESConv raw JSON and local Python dependencies。
- Produces: one command that proves all local work is ready; `reports/local_readiness.json` with hashes and explicit `gpu_runs_performed=false`。

- [ ] **Step 1: Write CLI help-contract tests**

```python
import subprocess
import sys


SCRIPTS = [
    "build_manifests",
    "validate_manifests",
    "export_emodynamix_logits",
    "calibrate_and_allocate",
    "extract_state",
    "build_generation_tasks",
    "generate_and_select",
    "judge_responses",
    "evaluate_results",
]


def test_every_pipeline_script_has_cpu_safe_help():
    for name in SCRIPTS:
        result = subprocess.run(
            [sys.executable, "-m", f"scripts.{name}", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run the test and fix every eager GPU import**

```powershell
python -m pytest tests/test_cli_contracts.py -v
```

Expected: every `--help` succeeds on Windows without torch, CUDA, checkpoint, Ollama, or network access.

- [ ] **Step 3: Add the single Windows readiness command**

`run_local_checks.ps1` must stop on the first non-zero exit and run these exact stages:

```powershell
$ErrorActionPreference = "Stop"
python -m pytest tests -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m scripts.build_manifests --config .\configs\experiment.json --output-root .\manifests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m scripts.validate_manifests --root .\manifests --rebuild-check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m scripts.calibrate_and_allocate --self-test
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m scripts.extract_state --manifest .\manifests\smoke20\base_manifest.jsonl --mock
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m scripts.build_generation_tasks --collection smoke20 --mock-inputs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m scripts.generate_and_select --collection smoke20 --mock
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m scripts.judge_responses --collection smoke20 --mock
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m scripts.evaluate_results --collection smoke20 --mock
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$hashes = [ordered]@{}
Get-ChildItem .\configs, .\manifests -Recurse -File |
    Sort-Object FullName |
    ForEach-Object {
        $relative = Resolve-Path -Relative $_.FullName
        $hashes[$relative] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    }

$readiness = [ordered]@{
    schema_version = "1.0"
    gpu_runs_performed = $false
    tests_passed = $true
    manifests_validated = $true
    mock_pipeline_passed = $true
    artifact_sha256 = $hashes
}

New-Item -ItemType Directory -Force .\reports | Out-Null
$readiness |
    ConvertTo-Json -Depth 5 |
    Set-Content .\reports\local_readiness.json -Encoding UTF8
```

- [ ] **Step 4: Document the no-GPU exit criteria and later GPU commands**

README must state that local readiness means tests, real-data manifest generation, deterministic rebuild, calibration fixture, mock state extraction, mock candidate generation, selector blinding, and evaluation parsing all pass. It must also state that mock outputs are non-reportable and that no scientific conclusion exists before real 4090/A800 runs.

Record the future 4090 order without executing it in this phase:

Run from the repository root after the retained AutoDL data disk has been mounted at the same repository paths:

```bash
python -m experiments.rapid_state_adaptive_k.scripts.export_emodynamix_logits \
  --emodynamix-root reproductions/EmoDynamiX \
  --checkpoint reproductions/EmoDynamiX/roberta-hg-esconv-preprocessed-checkpoints/checkpoint-2600.pth \
  --strategies reproductions/EmoDynamiX/data/esconv/strategies.json \
  --valid-pkl reproductions/EmoDynamiX/data/esconv_preprocessed/valid.pkl \
  --test-pkl reproductions/EmoDynamiX/data/esconv_preprocessed/test.pkl \
  --manifest-root experiments/rapid_state_adaptive_k/manifests \
  --preflight-only

python -m experiments.rapid_state_adaptive_k.scripts.export_emodynamix_logits \
  --emodynamix-root reproductions/EmoDynamiX \
  --checkpoint reproductions/EmoDynamiX/roberta-hg-esconv-preprocessed-checkpoints/checkpoint-2600.pth \
  --strategies reproductions/EmoDynamiX/data/esconv/strategies.json \
  --valid-pkl reproductions/EmoDynamiX/data/esconv_preprocessed/valid.pkl \
  --test-pkl reproductions/EmoDynamiX/data/esconv_preprocessed/test.pkl \
  --manifest-root experiments/rapid_state_adaptive_k/manifests \
  --output-root experiments/rapid_state_adaptive_k/reports/strategy
```

If preflight reports any missing path, stop without searching or moving files during paid GPU time; shut down the instance and repair the path mapping locally.

- [ ] **Step 5: Run final local verification**

```powershell
Set-Location .\experiments\rapid_state_adaptive_k
.\scripts\run_local_checks.ps1
git diff --check
git status --short --branch
```

Expected: all tests pass; manifest validation reports zero violations; mock reruns append zero successful duplicates; Git shows only intended source/config/test/manifest changes and ignored `reports/` outputs.

- [ ] **Step 6: Commit the local-ready gate**

```powershell
git add experiments/rapid_state_adaptive_k
git commit -m "test: add no-gpu readiness gate"
git push
```

## No-GPU Completion Gate

Do not rent or start a GPU until every item below is true:

- [ ] `python -m pytest tests -q` passes.
- [ ] Official split counts are 910/195/195.
- [ ] pilot100 contains exactly 100 distinct test dialogues.
- [ ] smoke20 contains exactly 20 rows and is a subset of pilot100.
- [ ] A clean rebuild produces byte-identical manifest SHA256 values.
- [ ] Base manifests contain no gold strategy, reference response, target response, or future turns.
- [ ] EmoDynamiX exporter `--help`, join tests, label-order tests, and duplicate-signature tests pass without CUDA.
- [ ] Calibration self-test covers both improved calibration and `T=1.0` fallback.
- [ ] KA and KR have exactly equal total candidate budgets.
- [ ] F3 has zero self-pairs and minimizes State Card length mismatch deterministically.
- [ ] State extraction, candidate generation, selector, retry, and resume paths pass in isolated mock outputs.
- [ ] Selector prompt contains no strategy rank, probability, gold response, or method identity.
- [ ] `reports/local_readiness.json` records `gpu_runs_performed=false` and hashes of all frozen configs/manifests.

## First GPU Session Boundary

After this plan is complete, the 4090 session may perform only:

1. pull the committed branch;
2. verify checkpoint 2600, `strategies.json`, valid/test preprocessed PKLs, RoBERTa assets, CUDA, and hashes;
3. run `--preflight-only`;
4. export dev_cal, pilot100, and smoke20 raw logits;
5. copy the JSONL plus preflight record back to the repository workspace;
6. stop the 4090 before starting calibration or any A800 work.
