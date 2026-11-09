#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/paper_repro
BASE="$ROOT/adapters/memgas_personamem_128k"
PY=/root/autodl-tmp/envs/memgas/bin/python
GEN_LOG="$BASE/session_generation_128k.log"
LOG="$BASE/cache_chain_128k.log"
STATUS="$BASE/cache_chain_128k.exitcode"
GEN_PID="${1:?Usage: script GENERATION_PID}"

mkdir -p "$BASE"
exec >> "$LOG" 2>&1

# Prevent two supervisors from running simultaneously.
exec 9>"$BASE/cache_chain_128k.lock"
if ! flock -n 9; then
    echo "ERROR: another supervisor holds the lock"
    exit 90
fi

if [[ -f "$STATUS" ]]; then
    cp -p "$STATUS" "$STATUS.previous.$(date +%Y%m%d_%H%M%S)"
fi
printf 'RUNNING\n' > "$STATUS.tmp"
mv "$STATUS.tmp" "$STATUS"

finish() {
    rc=$?
    trap - EXIT

    printf '%s\n' "$rc" > "$STATUS.tmp"
    mv "$STATUS.tmp" "$STATUS"

    if (( rc != 0 )); then
        echo "===== FAILURE: exit_code=$rc ====="
        echo "time=$(date -Is)"
        echo "Attempting system shutdown."
        sync || true

        if command -v shutdown >/dev/null 2>&1; then
            if shutdown -h now >> "$LOG" 2>&1; then
                echo "SHUTDOWN_COMMAND_ACCEPTED"
            else
                echo "SHUTDOWN_COMMAND_FAILED"
                echo "The container/platform may prohibit system shutdown."
                echo "Use the provider console or supported API if necessary."
            fi
        else
            echo "SHUTDOWN_COMMAND_UNAVAILABLE"
            echo "Provider-level shutdown is required."
        fi
        sync || true
    else
        echo "===== SUCCESS: exit_code=0 ====="
        echo "The instance will remain running."
    fi

    exit "$rc"
}
trap finish EXIT

die() {
    echo "FATAL: $*"
    exit 1
}

echo "===== FAIL-CLOSED CHAIN START ====="
echo "time=$(date -Is)"
echo "generation_pid=$GEN_PID"

[[ "$GEN_PID" =~ ^[0-9]+$ ]] || die "Invalid generation PID"
[[ -f "$GEN_LOG" ]] || die "Generation log missing"

# Wait for the existing generation process, without killing it.
cmd="$(ps -p "$GEN_PID" -o args= 2>/dev/null || true)"

if [[ -n "$cmd" ]]; then
    [[ "$cmd" == *generate_sessions_qwen7b_128k.py* ]] \
        || die "PID does not match the expected generation script"

    while true; do
        state="$(ps -p "$GEN_PID" -o stat= 2>/dev/null | awk '{print $1}' || true)"
        [[ -z "$state" || "$state" == Z* ]] && break
        sleep 30
    done
fi

echo "Generation process exited: $(date -Is)"

# Require the generator's actual completion marker.
grep -Fxq 'MEMGAS_1M_SESSION_GENERATION_OK' "$GEN_LOG" \
    || die "Generation success marker missing"

# Full CPU-only audit; incomplete output must block the cache build.
"$PY" -u "$BASE/preflight_contriever_cache_128k.py" \
    || die "128K input preflight failed"

# Never overwrite an existing cache.
[[ ! -e "$BASE/personamem_128k_contriever_cache.pt" ]] \
    || die "Cache already exists; refusing to overwrite"

if pgrep -af '[b]uild_contriever_cache_128k.py' >/dev/null; then
    die "Another cache builder is already running"
fi

echo "===== STARTING 128K CACHE BUILD ====="
before_lines="$(wc -l < "$LOG")"

set +e
"$PY" -u "$BASE/build_contriever_cache_128k.py"
build_rc=$?
set -e

(( build_rc == 0 )) || die "Cache builder exited with code $build_rc"

[[ -s "$BASE/personamem_128k_contriever_cache.pt" ]] \
    || die "Cache output is missing or empty"

# Require a success marker emitted during THIS build.
tail -n "+$((before_lines + 1))" "$LOG" \
    | grep -Fxq 'MEMGAS_PERSONAMEM_128K_CACHE_OK' \
    || die "Fresh cache success marker missing"

# Independently inspect the serialized cache on CPU.
"$PY" -u - <<'PY'
from pathlib import Path
import torch

p = Path(
    "/root/autodl-tmp/paper_repro/adapters/memgas_personamem_128k/"
    "personamem_128k_contriever_cache.pt"
)

try:
    obj = torch.load(p, map_location="cpu", weights_only=False)
except TypeError:
    obj = torch.load(p, map_location="cpu")

assert obj["meta"]["dataset"] == "PersonaMem-128K"
assert obj["meta"]["adapter"] == "memgas_personamem_128k"
assert len(obj["session_hashes"]) == 1249
assert len(set(obj["session_hashes"])) == 1249
assert len(obj["question_ids"]) == 2727
assert len(set(obj["question_ids"])) == 2727

dim = obj["meta"]["embedding_dim"]
for key in ("raw", "turn_mean", "summary", "keyword", "hybrid"):
    t = obj[key]
    assert tuple(t.shape) == (1249, dim), (key, t.shape)
    assert torch.isfinite(t).all().item(), key

q = obj["question_embeddings"]
assert tuple(q.shape) == (2727, dim)
assert torch.isfinite(q).all().item()
assert len(obj["question_session_hashes"]) == 2727
assert len(obj["question_session_indices"]) == 2727

print("POST_BUILD_CACHE_AUDIT_PASS")
print("sessions=1249 questions=2727 dim=", dim)
PY

echo "MEMGAS_PERSONAMEM_128K_AUTO_CHAIN_OK"
