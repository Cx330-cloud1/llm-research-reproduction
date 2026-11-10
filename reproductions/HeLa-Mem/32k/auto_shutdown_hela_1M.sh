#!/usr/bin/env bash
set -u

ROOT="/root/autodl-tmp/paper_repro/HeLa-Mem-personamem"
cd "$ROOT" || exit 1

RUN_LOG="$ROOT/hela_personamem_1M_deferred.log"
CACHE_ROOT="$ROOT/personamem_cache_1M"
RESULTS_ROOT="$ROOT/personamem_results_1M"
RESULT_JSON="$ROOT/results_hela_personamem_1M.json"
RESULT_JSONL="$ROOT/results_hela_personamem_1M.jsonl"
AUDIT="$RESULTS_ROOT/audit.json"
ERRORS="$RESULTS_ROOT/errors.jsonl"

EXPECTED_CACHES=276
EXPECTED_QUESTIONS=2674

# 精确匹配当前 1M 任务及其参数，避免误判其他任务。
PROCESS_RE='^/root/autodl-tmp/envs/rfmem-gpu/bin/python -u eval_personamem_cached.py --phase all --top_k 15 --semantic_top_k 5 --cache_root personamem_cache_1M --results_root personamem_results_1M --output_prefix results_hela_personamem_1M$'

log() {
    echo "[$(date '+%F %T')] $*"
}

is_running() {
    pgrep -f "$PROCESS_RE" >/dev/null 2>&1
}

log "开始监控 HeLa-Mem 1M 任务。"
log "不会启动新实验，也不会删除现有缓存。"

# 等待主进程和使用相同参数的子进程全部退出。
while is_running; do
    log "1M 任务仍在运行，60 秒后再次检查。"
    sleep 60
done

# 留出时间，让最后的文件写入和进程清理完成。
sleep 10
log "检测到 1M 任务进程已退出，开始检查实验结果。"

if [[ ! -f "$RUN_LOG" ]]; then
    log "失败：找不到运行日志，不执行关机。"
    exit 1
fi

if ! grep -Fq \
  "FULL DATASET COUNT AUDIT: PASS (2674 questions / 276 cutoffs)" \
  "$RUN_LOG"; then
    log "失败：完整数据集审计记录不存在，不执行关机。"
    exit 1
fi

READY_COUNT=$(find "$CACHE_ROOT" -type f -name READY | wc -l)
READY_COUNT=$(echo "$READY_COUNT" | tr -d ' ')

log "READY 缓存数量：$READY_COUNT / $EXPECTED_CACHES"

if [[ "$READY_COUNT" -ne "$EXPECTED_CACHES" ]]; then
    log "失败：缓存数量不完整，不执行关机。"
    exit 1
fi

# 检查完整结果、JSONL、审计文件和错误记录。
if ! /root/autodl-tmp/envs/rfmem-gpu/bin/python - \
    "$RESULT_JSON" "$RESULT_JSONL" "$AUDIT" "$ERRORS" \
    "$EXPECTED_QUESTIONS" <<'PY'
import json
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
jsonl_path = Path(sys.argv[2])
audit_path = Path(sys.argv[3])
errors_path = Path(sys.argv[4])
expected = int(sys.argv[5])

for p in (json_path, jsonl_path, audit_path):
    if not p.is_file():
        print(f"FAIL: missing file: {p}")
        sys.exit(1)

try:
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    if not isinstance(rows, list):
        raise ValueError("result JSON is not a list")

    if len(rows) != expected:
        raise ValueError(
            f"result count {len(rows)} != {expected}"
        )

    ids = [r.get("question_id") for r in rows]
    if any(not qid for qid in ids) or len(set(ids)) != expected:
        raise ValueError("missing or duplicate question IDs")

    if any("score" not in r or "error" not in r for r in rows):
        raise ValueError("some result records are incomplete")

    bad = [r for r in rows if r.get("error")]
    if bad:
        raise ValueError(f"{len(bad)} result records have errors")

    with jsonl_path.open(encoding="utf-8") as f:
        jsonl_rows = [line for line in f if line.strip()]

    if len(jsonl_rows) != expected:
        raise ValueError(
            f"JSONL count {len(jsonl_rows)} != {expected}"
        )

    # Also validate every JSONL line.
    for line in jsonl_rows:
        json.loads(line)

    if errors_path.exists() and errors_path.stat().st_size > 0:
        raise ValueError("errors.jsonl is not empty")

    print(f"PASS: {len(rows)} unique question results")
    print(f"PASS: {len(jsonl_rows)} JSONL records")
    print(f"PASS: audit JSON is readable ({type(audit).__name__})")

except Exception as exc:
    print(f"FAIL: {exc}")
    sys.exit(1)
PY
then
    log "失败：结果校验未通过，不执行关机。"
    exit 1
fi

log "全部检查通过，准备尝试关机。"

if ! command -v shutdown >/dev/null 2>&1; then
    log "当前环境没有 shutdown 命令，无法自动关机。"
    exit 2
fi

# 这是不可逆的关机动作，只在以上所有检查通过后执行。
if shutdown -h now; then
    log "已向系统提交关机命令。"
else
    log "关机命令执行失败，请检查平台是否允许容器内关机。"
    exit 3
fi
