#!/usr/bin/env bash

set -u

WORKDIR="/root/autodl-tmp/paper_repro/HeLa-Mem-personamem"
CACHE_DIR="$WORKDIR/personamem_cache_1M_prefix"
LOG="$WORKDIR/prefix_cache_build_batch.log"

EXPECTED=276
POLL_SECONDS=60
STABLE_WAIT=120

WATCH_LOG="$WORKDIR/auto_shutdown_monitor.log"
SEGMENT_FILE="$WORKDIR/.auto_shutdown_new_log.tmp"

cd "$WORKDIR" || exit 1

exec >> "$WATCH_LOG" 2>&1

echo "=========================================="
echo "HeLa-Mem AUTO SHUTDOWN MONITOR"
date
echo "=========================================="

# 检查必要文件
if [[ ! -d "$CACHE_DIR" || ! -f "$LOG" ]]; then
    echo "[ERROR] Cache directory or log missing."
    exit 1
fi

# 记录启动监控时的日志位置
LOG_META=$(stat -c '%i %s' "$LOG") || exit 1
read -r LOG_INODE LOG_OFFSET <<< "$LOG_META"

echo "[INFO] Initial log offset: $LOG_OFFSET"
echo "[INFO] Expected READY: $EXPECTED"

# 精确寻找当前实验进程
builder_pids() {
    pgrep -f \
    '[b]uild_personamem_prefix_cache.py --cache_root personamem_cache_1M_prefix --context_limit 31' \
    || true
}

# 统计 READY
ready_count() {
    find "$CACHE_DIR" -type f -name READY 2>/dev/null \
    | wc -l | tr -d '[:space:]'
}

# 检查本次监控启动后产生的成功日志
check_success_log() {

    local inode size

    read -r inode size < <(stat -c '%i %s' "$LOG") || return 1

    # 日志被替换或截断时，禁止自动关机
    [[ "$inode" == "$LOG_INODE" ]] || return 1
    [[ "$size" -ge "$LOG_OFFSET" ]] || return 1

    tail -c +"$((LOG_OFFSET + 1))" "$LOG" > "$SEGMENT_FILE"

    grep -qF "PREFIX ENCODING FINISHED" "$SEGMENT_FILE" &&
    grep -Eq '^[[:space:]]*READY:[[:space:]]*276[[:space:]]*$' "$SEGMENT_FILE" &&
    grep -Eq '^[[:space:]]*EXPECTED:[[:space:]]*276[[:space:]]*$' "$SEGMENT_FILE" &&
    grep -Eq '^\[CONTEXT 31/31\]' "$SEGMENT_FILE"
}

while true; do

    PIDS=$(builder_pids)
    READY=$(ready_count)

    echo "------------------------------------------"
    date
    echo "READY: $READY / $EXPECTED"
    echo "Builder PIDs: ${PIDS:-NONE}"

    # 只有构建进程全部退出才进入最终检查
    if [[ -z "$PIDS" ]]; then

        echo "[CHECK] Builder stopped."
        echo "[CHECK] Waiting ${STABLE_WAIT}s for stability..."

        sleep "$STABLE_WAIT"

        PIDS=$(builder_pids)
        READY=$(ready_count)

        # 再次确认没有构建进程
        if [[ -n "$PIDS" ]]; then
            echo "[WAIT] Builder still exists."
            continue
        fi

        # 缓存数量必须完全达到预期
        if [[ "$READY" -ne "$EXPECTED" ]]; then
            echo "[FAILED] READY incomplete: $READY/$EXPECTED"
            echo "[SAFE] Auto shutdown cancelled."
            exit 2
        fi

        # 检查最终成功日志
        if ! check_success_log; then
            echo "[FAILED] Success marker verification failed."
            echo "[SAFE] Auto shutdown cancelled."
            exit 3
        fi

        # 最终二次检查
        PIDS=$(builder_pids)
        READY=$(ready_count)

        if [[ -n "$PIDS" || "$READY" -ne "$EXPECTED" ]]; then
            echo "[FAILED] Final verification mismatch."
            exit 4
        fi

        echo "=========================================="
        echo "[SUCCESS] ALL CHECKS PASSED"
        echo "READY: $READY / $EXPECTED"
        echo "All 31 contexts completed."
        echo "Builder processes exited."
        echo "=========================================="

        date >> "$WORKDIR/EXPERIMENT_COMPLETED.txt"
        echo "READY=$READY/$EXPECTED" >> "$WORKDIR/EXPERIMENT_COMPLETED.txt"

        sync

        echo "[SHUTDOWN] Calling AutoDL shutdown..."

        if [[ -x /usr/bin/shutdown ]]; then
            /usr/bin/shutdown
        else
            echo "[ERROR] /usr/bin/shutdown not found."
            exit 5
        fi

        exit $?
    fi

    sleep "$POLL_SECONDS"

done
