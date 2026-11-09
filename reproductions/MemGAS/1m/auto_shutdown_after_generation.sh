#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/autodl-tmp/paper_repro/adapters/memgas_personamem_1m

LOG="generation_full.log"
OUT="session_generations_qwen7b.jsonl"
INPUT="unique_sessions.jsonl"
MARKER="MEMGAS_1M_SESSION_GENERATION_OK"
WATCH_LOG="auto_shutdown_watch.log"

echo "[$(date '+%F %T')] Watcher started" >> "$WATCH_LOG"

# 等待生成成功标记；异常退出时禁止自动关机
while true; do
    if grep -qF "$MARKER" "$LOG"; then
        break
    fi

    PID="$(cat generation_full.pid 2>/dev/null || true)"

    if [[ -n "$PID" ]] && ! kill -0 "$PID" 2>/dev/null; then
        sleep 10

        if ! grep -qF "$MARKER" "$LOG"; then
            echo "[$(date '+%F %T')] ERROR: generator exited without success marker; NO SHUTDOWN" >> "$WATCH_LOG"
            exit 2
        fi
        break
    fi

    sleep 60
done

echo "[$(date '+%F %T')] Generation success marker detected" >> "$WATCH_LOG"

# 严格检查输入输出完整性
/root/autodl-tmp/envs/memgas/bin/python - <<'PY' >> auto_shutdown_watch.log 2>&1
import json
from pathlib import Path
from collections import Counter

src = Path("unique_sessions.jsonl")
out = Path("session_generations_qwen7b.jsonl")

def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Malformed JSONL: {path}:{line_no}: {e}")
    return rows

source = read_jsonl(src)
result = read_jsonl(out)

expected = [x["hash"] for x in source]
actual = [x.get("hash") for x in result]

assert len(expected) == 2090, f"Unexpected input count: {len(expected)}"
assert len(set(expected)) == 2090, "Duplicate input hashes"
assert len(result) == 2090, f"Output count: {len(result)}"
assert len(set(actual)) == 2090, "Duplicate output hashes"
assert set(actual) == set(expected), "Missing or unexpected session hashes"

for row in result:
    assert isinstance(row.get("summary"), str) and row["summary"].strip(), \
        f"Empty summary: {row.get('hash')}"
    assert isinstance(row.get("keywords"), str) and row["keywords"].strip(), \
        f"Empty keywords: {row.get('hash')}"

print("input_sessions:", len(source))
print("output_sessions:", len(result))
print("generation_modes:", dict(Counter(
    x.get("generation_mode", "unknown") for x in result
)))
print("MEMGAS_1M_FINAL_OUTPUT_AUDIT_PASS")
PY

echo "[$(date '+%F %T')] AUTO_SHUTDOWN_VALIDATION_OK" >> "$WATCH_LOG"

# 持久化日志后才发起关机
sync
echo "[$(date '+%F %T')] Requesting system shutdown" >> "$WATCH_LOG"
sync

if command -v shutdown >/dev/null 2>&1; then
    if shutdown -h now; then
        echo "[$(date '+%F %T')] SHUTDOWN_COMMAND_ACCEPTED" >> "$WATCH_LOG"
    else
        RC=$?
        echo "[$(date '+%F %T')] SHUTDOWN_COMMAND_FAILED rc=$RC" >> "$WATCH_LOG"
        exit "$RC"
    fi
else
    echo "[$(date '+%F %T')] ERROR: shutdown command unavailable" >> "$WATCH_LOG"
    exit 3
fi
