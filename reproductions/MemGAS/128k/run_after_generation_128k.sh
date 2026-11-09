#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/paper_repro
BASE="$ROOT/adapters/memgas_personamem_128k"
PY=/root/autodl-tmp/envs/memgas/bin/python
GEN_PID="${1:?Usage: run_after_generation_128k.sh GENERATION_PID}"

LOG="$BASE/cache_chain_128k.log"
STATUS="$BASE/cache_chain_128k.exitcode"

mkdir -p "$BASE"
exec >> "$LOG" 2>&1

finish() {
    rc=$?
    printf '%s\n' "$rc" > "$STATUS.tmp"
    mv "$STATUS.tmp" "$STATUS"
}
trap finish EXIT

exec 9>"$BASE/cache_chain_128k.lock"
flock -n 9 || {
    echo "ERROR: another cache-chain supervisor holds the lock"
    exit 90
}

echo "===== CACHE CHAIN START ====="
echo "time=$(date -Is) generation_pid=$GEN_PID"

if [[ ! "$GEN_PID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid generation PID"
    exit 91
fi

# The current generator was started separately. Capture its log while
# its process still exists; fail closed if no regular log is available.
GEN_LOG="$(readlink -f "/proc/$GEN_PID/fd/1" 2>/dev/null || true)"
if [[ -z "$GEN_LOG" || ! -f "$GEN_LOG" ]]; then
    echo "ERROR: cannot identify the generator's regular log file"
    echo "No cache build will be started."
    exit 92
fi
echo "generation_log=$GEN_LOG"

while true; do
    state="$(ps -p "$GEN_PID" -o stat= 2>/dev/null | awk '{print $1}' || true)"
    if [[ -z "$state" || "$state" == Z* ]]; then
        break
    fi
    sleep 30
done

echo "Generation process has exited: $(date -Is)"

# Require the generator's explicit success marker and complete data audit.
if ! grep -Fxq 'MEMGAS_1M_SESSION_GENERATION_OK' "$GEN_LOG"; then
    echo "ERROR: generator success marker missing; cache build blocked"
    exit 93
fi

if grep -q 'Traceback (most recent call last)' "$GEN_LOG"; then
    echo "ERROR: traceback detected in generation log; cache build blocked"
    exit 94
fi

set +e
"$PY" -u "$BASE/preflight_contriever_cache_128k.py"
preflight_rc=$?
set -e

if [[ "$preflight_rc" -ne 0 ]]; then
    echo "ERROR: preflight failed with exit code $preflight_rc"
    echo "Cache build will not start."
    exit 95
fi

if pgrep -af '[b]uild_contriever_cache_128k.py' >/dev/null; then
    echo "ERROR: another cache build process is already active"
    exit 96
fi

echo "===== STARTING 128K CACHE BUILD ====="
set +e
"$PY" -u "$BASE/build_contriever_cache_128k.py"
cache_rc=$?
set -e
echo "cache_builder_exit_code=$cache_rc"

if [[ "$cache_rc" -ne 0 ]]; then
    exit "$cache_rc"
fi

if [[ ! -f "$BASE/personamem_128k_contriever_cache.pt" ]]; then
    echo "ERROR: cache builder returned 0 but output is missing"
    exit 97
fi

if ! grep -Fxq 'MEMGAS_PERSONAMEM_128K_CACHE_OK' "$LOG"; then
    echo "ERROR: final 128K cache success marker missing"
    exit 98
fi

echo "MEMGAS_PERSONAMEM_128K_AUTO_CHAIN_OK"
