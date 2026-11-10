#!/bin/bash
set -euo pipefail

cd /root/autodl-tmp/paper_repro/HeLa-Mem-personamem

PY=/root/autodl-tmp/envs/rfmem-gpu/bin/python

echo "===== START HeLa PersonaMem 128k ====="
date

$PY -u eval_personamem_cached.py \
  --phase all \
  --top_k 15 \
  --semantic_top_k 5 \
  --cache_root personamem_cache_128k \
  --results_root personamem_results_128k \
  --output_prefix results_hela_personamem_128k \
  2>&1 | tee hela_personamem_128k.log

echo "===== RUN FINISHED, START AUDIT ====="

$PY - <<'PY'
import pandas as pd
from pathlib import Path

p = Path("results_hela_personamem_128k.csv")
assert p.exists(), "result CSV missing"

df = pd.read_csv(p)

assert len(df) == 2727, f"rows={len(df)}, expected 2727"
assert df["question_id"].nunique() == 2727, "duplicate/missing question_id"

if "model_response" in df:
    assert df["model_response"].notna().all(), "empty model_response"

if "predicted_answer" in df:
    assert df["predicted_answer"].notna().all(), "empty predicted_answer"

print("HELA 128K FINAL AUDIT: PASS")
print("rows:", len(df))
print("unique:", df["question_id"].nunique())

if "score" in df:
    score = df["score"].astype(str).str.lower().isin(["true", "1"])
    print("correct:", int(score.sum()))
    print("accuracy:", float(score.mean()))
PY

echo "===== ALL PASS - SHUTTING DOWN ====="
date
sync
/usr/bin/shutdown
