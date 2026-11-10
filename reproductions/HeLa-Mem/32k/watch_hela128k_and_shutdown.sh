#!/bin/bash
set -u

PID=51834
ROOT=/root/autodl-tmp/paper_repro/HeLa-Mem-personamem
PY=/root/autodl-tmp/envs/rfmem-gpu/bin/python

cd "$ROOT"

echo "===== WATCHING EXISTING HeLa 128k PID=$PID ====="
date

while kill -0 "$PID" 2>/dev/null; do
    echo "$(date '+%F %T') PID $PID still running..."
    sleep 60
done

echo "===== PROCESS FINISHED - START AUDIT ====="
date

$PY - <<'PY'
import pandas as pd
from pathlib import Path

p = Path("results_hela_personamem_128k.csv")
assert p.exists(), f"result missing: {p}"

df = pd.read_csv(p)

assert len(df) == 2727, f"rows={len(df)}, expected 2727"
assert df["question_id"].nunique() == 2727, \
    f"unique question_id={df['question_id'].nunique()}, expected 2727"

for col in ["model_response", "predicted_answer"]:
    if col in df.columns:
        assert df[col].notna().all(), f"empty values detected in {col}"

print("HELA 128K FINAL AUDIT: PASS")
print("rows:", len(df))
print("unique question_id:", df["question_id"].nunique())

if "score" in df.columns:
    s = df["score"].astype(str).str.lower().isin(["true", "1"])
    print("correct:", int(s.sum()))
    print("accuracy:", float(s.mean()))
PY

STATUS=$?

if [ "$STATUS" -eq 0 ]; then
    echo "===== AUDIT PASS - SHUTTING DOWN ====="
    date
    sync
    /usr/bin/shutdown
else
    echo "===== AUDIT FAILED - NOT SHUTTING DOWN ====="
    exit 1
fi
