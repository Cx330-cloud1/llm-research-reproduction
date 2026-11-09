#!/usr/bin/env bash
set -u

ROOT="/root/autodl-tmp/paper_repro/MemGAS"

SUMMARY="$ROOT/multi_granularity_logs/longmemeval-summary_level.jsonl"
KEYWORD="$ROOT/multi_granularity_logs/longmemeval-keyword_level.jsonl"
GEN_LOG="$ROOT/repro_logs/longmemeval_qwen7b_generation.log"

WATCH_LOG="$ROOT/repro_logs/longmemeval_shutdown_watch.log"
DONE_MARKER="$ROOT/repro_logs/LONGMEMEVAL_GENERATION_VERIFIED_BEFORE_SHUTDOWN"

TARGET=19545
INTERVAL=60

cd "$ROOT" || exit 1

mkdir -p "$ROOT/repro_logs"

log() {
    echo "$(date '+%F %T') | $*" | tee -a "$WATCH_LOG"
}

count_lines() {
    local f="$1"

    if [ -f "$f" ]; then
        wc -l < "$f"
    else
        echo 0
    fi
}

generator_running() {
    pgrep -f "[r]epro_generate_longmemeval_qwen7b.py" >/dev/null 2>&1
}

log "WATCHDOG_STARTED"
log "target summary=$TARGET keyword=$TARGET"

while true; do
    S=$(count_lines "$SUMMARY")
    K=$(count_lines "$KEYWORD")

    log "summary=$S/$TARGET keyword=$K/$TARGET"

    # ---------- 完整数量达到目标 ----------
    if [ "$S" -eq "$TARGET" ] && [ "$K" -eq "$TARGET" ]; then

        # 生成程序可能刚写完最后一行，还在写 metadata/成功标记。
        if generator_running; then
            log "Both counts complete, waiting for generator to exit cleanly..."
            sleep "$INTERVAL"
            continue
        fi

        # 必须有 runner 的最终成功标记。
        if ! grep -q "LONGMEMEVAL_QWEN7B_GENERATION_OK" "$GEN_LOG" 2>/dev/null; then
            log "ERROR: counts complete but success marker missing."
            log "WILL NOT SHUT DOWN."
            exit 20
        fi

        log "Success marker found. Starting strict JSONL validation."

        /root/autodl-tmp/envs/memgas/bin/python - <<'PY'
import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/paper_repro/MemGAS")
TARGET = 19545

paths = {
    "summary": ROOT / "multi_granularity_logs/longmemeval-summary_level.jsonl",
    "keyword": ROOT / "multi_granularity_logs/longmemeval-keyword_level.jsonl",
}

all_keys = {}

for name, path in paths.items():
    keys = []
    empty = 0

    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                raise RuntimeError(
                    f"{name}: blank line at {lineno}"
                )

            obj = json.loads(line)

            if not isinstance(obj, dict) or len(obj) != 1:
                raise RuntimeError(
                    f"{name}: malformed record at {lineno}"
                )

            sid, value = next(iter(obj.items()))

            if not str(value).strip():
                empty += 1

            keys.append(sid)

    print(f"{name}_lines:", len(keys))
    print(f"{name}_unique:", len(set(keys)))
    print(f"{name}_empty:", empty)

    if len(keys) != TARGET:
        raise RuntimeError(
            f"{name}: expected {TARGET} lines, got {len(keys)}"
        )

    if len(set(keys)) != TARGET:
        raise RuntimeError(
            f"{name}: duplicate session IDs found"
        )

    if empty:
        raise RuntimeError(
            f"{name}: {empty} empty generations"
        )

    all_keys[name] = set(keys)

if all_keys["summary"] != all_keys["keyword"]:
    missing_kw = all_keys["summary"] - all_keys["keyword"]
    missing_sum = all_keys["keyword"] - all_keys["summary"]

    raise RuntimeError(
        "summary/keyword SID mismatch: "
        f"missing_keyword={len(missing_kw)}, "
        f"missing_summary={len(missing_sum)}"
    )

print("LONGMEMEVAL_FINAL_JSONL_VALIDATION_OK")
PY

        VALID_STATUS=$?

        if [ "$VALID_STATUS" -ne 0 ]; then
            log "ERROR: JSONL validation failed with status=$VALID_STATUS"
            log "WILL NOT SHUT DOWN."
            exit 30
        fi

        log "FINAL VALIDATION PASSED."

        {
            echo "verified_at=$(date '+%F %T')"
            echo "summary=$S"
            echo "keyword=$K"
            echo "target=$TARGET"
            echo "LONGMEMEVAL_FINAL_JSONL_VALIDATION_OK"
        } > "$DONE_MARKER"

        # 把磁盘缓存尽量刷盘后再关机。
        sync

        log "All LongMemEval generation outputs verified."
        log "AUTO SHUTDOWN IN 10 SECONDS."

        sleep 10

        /usr/bin/shutdown -h now
        exit 0
    fi

    # ---------- 超过目标：异常，绝不关机 ----------
    if [ "$S" -gt "$TARGET" ] || [ "$K" -gt "$TARGET" ]; then
        log "ERROR: output count exceeded expected target."
        log "summary=$S keyword=$K target=$TARGET"
        log "WILL NOT SHUT DOWN."
        exit 40
    fi

    # ---------- 任务中途异常退出 ----------
    if ! generator_running; then
        log "ERROR: generator is no longer running before completion."
        log "summary=$S/$TARGET keyword=$K/$TARGET"
        log "WILL NOT SHUT DOWN."
        exit 50
    fi

    sleep "$INTERVAL"
done
