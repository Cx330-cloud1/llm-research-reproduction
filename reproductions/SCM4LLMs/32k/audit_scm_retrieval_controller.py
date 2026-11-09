import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import tiktoken

DATA = Path("personamem_scm_32k.jsonl")
MAPS = Path("question_scm_unit_hashes.jsonl")
UNIQUE = Path("unique_scm_memory_units.jsonl")
SUMMARIES = Path("scm_summaries_qwen7b.jsonl")
CACHE = Path("scm_contriever_cache.pt")

OUT = Path("scm_retrieval_controller_audit.json")
OUT_TOPK = Path("scm_top6_retrieval.jsonl")

TOPK = 6
MAX_HISTORY_TOKENS = 2500
MAX_PRE_TURN_TOKENS = 500

try:
    enc = tiktoken.encoding_for_model("text-davinci-003")
except Exception:
    enc = tiktoken.get_encoding("p50k_base")


def toklen(text):
    return len(enc.encode(text))


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

questions = []
with open(DATA, encoding="utf-8") as f:
    for line in f:
        questions.append(json.loads(line))

assert len(questions) == 589

qmap = {}
with open(MAPS, encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        qmap[x["question_id"]] = x["unit_hashes"]

unique = {}
with open(UNIQUE, encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        unique[x["unit_hash"]] = x

summaries = {}
with open(SUMMARIES, encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        summaries[x["unit_hash"]] = x["summary"]

assert len(unique) == 2287
assert len(summaries) == 2287

cache = torch.load(CACHE, map_location="cpu")

unit_hashes = cache["unit_hashes"]
memory_emb = cache["memory_embeddings"].float()
question_ids = cache["question_ids"]
question_emb = cache["question_embeddings"].float()

assert tuple(memory_emb.shape) == (2287, 768)
assert tuple(question_emb.shape) == (589, 768)

hash_to_emb_idx = {
    h: i for i, h in enumerate(unit_hashes)
}

qid_to_qemb_idx = {
    qid: i for i, qid in enumerate(question_ids)
}

# normalize only for cosine computation.
# This does NOT alter stored embeddings.
memory_norm = memory_emb / memory_emb.norm(
    dim=1, keepdim=True
).clamp(min=1e-12)

question_norm = question_emb / question_emb.norm(
    dim=1, keepdim=True
).clamp(min=1e-12)


# ------------------------------------------------------------
# Exact SCM budget accounting
# ------------------------------------------------------------

def previous_turn_budget(unit_hash):
    raw = unique[unit_hash]["raw_text"]
    raw_len = toklen(raw)

    if raw_len < MAX_PRE_TURN_TOKENS:
        return raw_len, "raw"

    summ = summaries[unit_hash]
    return toklen(summ), "summary"


controller_needed = 0
safe_after_all_summary = 0
official_nontermination_risk = 0

raw_budget_vals = []
all_summary_budget_vals = []
pre_raw_count = 0
pre_summary_count = 0

topk_outputs = []


for qi, q in enumerate(questions):
    qid = q["question_id"]
    hashes = qmap[qid]

    assert len(hashes) >= 1

    # Official SCM:
    # retrieval candidates = history[:-1]
    candidate_hashes = hashes[:-1]
    previous_hash = hashes[-1]

    qv = question_norm[qid_to_qemb_idx[qid]]

    if candidate_hashes:
        indices = [
            hash_to_emb_idx[h]
            for h in candidate_hashes
        ]

        mat = memory_norm[indices]

        sims = torch.mv(mat, qv).numpy()

        # exact NumPy behavior used by SCM:
        # arr.argsort()[-k:]
        local_top = np.argsort(sims)[-TOPK:]

        top_indices = [
            int(i) for i in local_top
        ]

        selected_hashes = [
            candidate_hashes[i]
            for i in top_indices
        ]

        selected_sims = [
            float(sims[i])
            for i in top_indices
        ]
    else:
        top_indices = []
        selected_hashes = []
        selected_sims = []

    raw_lengths = [
        toklen(unique[h]["raw_text"])
        for h in selected_hashes
    ]

    summary_lengths = [
        toklen(summaries[h])
        for h in selected_hashes
    ]

    pre_len, pre_mode = previous_turn_budget(
        previous_hash
    )

    if pre_mode == "raw":
        pre_raw_count += 1
    else:
        pre_summary_count += 1

    raw_budget = sum(raw_lengths) + pre_len
    all_summary_budget = (
        sum(summary_lengths) + pre_len
    )

    raw_budget_vals.append(raw_budget)
    all_summary_budget_vals.append(
        all_summary_budget
    )

    needs_controller = (
        raw_budget > MAX_HISTORY_TOKENS
    )

    if needs_controller:
        controller_needed += 1

        if all_summary_budget <= MAX_HISTORY_TOKENS:
            safe_after_all_summary += 1
        else:
            official_nontermination_risk += 1

    # Official output later restores chronological order.
    chronological = sorted(
        zip(
            top_indices,
            selected_hashes,
            selected_sims,
            raw_lengths,
            summary_lengths,
        ),
        key=lambda x: x[0]
    )

    topk_outputs.append({
        "question_id": qid,
        "history_units": len(hashes),
        "candidate_units": len(candidate_hashes),

        # order from np.argsort()[-k:]
        # = controller iteration order
        "controller_top_indices": top_indices,
        "controller_top_hashes": selected_hashes,
        "controller_top_sims": selected_sims,

        # order used in final concatenation
        "chronological_top_indices": [
            x[0] for x in chronological
        ],
        "chronological_top_hashes": [
            x[1] for x in chronological
        ],

        "raw_lengths": raw_lengths,
        "summary_lengths": summary_lengths,

        "previous_hash": previous_hash,
        "previous_mode": pre_mode,
        "previous_budget_tokens": pre_len,

        "raw_total_budget": raw_budget,
        "all_summary_total_budget":
            all_summary_budget,

        "needs_controller": needs_controller,
        "official_nontermination_risk": (
            needs_controller
            and all_summary_budget >
            MAX_HISTORY_TOKENS
        ),
    })


def stats(vals):
    vals = sorted(vals)

    def pct(p):
        idx = int(round(
            (len(vals) - 1) * p
        ))
        return vals[idx]

    return {
        "min": vals[0],
        "median": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": vals[-1],
    }


audit = {
    "questions": len(questions),

    "topk": TOPK,
    "max_history_tokens":
        MAX_HISTORY_TOKENS,
    "max_previous_turn_tokens":
        MAX_PRE_TURN_TOKENS,

    "controller_needed":
        controller_needed,

    "controller_not_needed":
        len(questions) -
        controller_needed,

    "safe_if_all_selected_turns_summary":
        safe_after_all_summary,

    "official_nontermination_risk":
        official_nontermination_risk,

    "previous_turn_raw":
        pre_raw_count,

    "previous_turn_summary":
        pre_summary_count,

    "raw_budget_stats":
        stats(raw_budget_vals),

    "all_summary_budget_stats":
        stats(all_summary_budget_vals),

    "similarity":
        "cosine",

    "retrieval_candidates":
        "all memory units except last; last unit is previous turn",
}


with open(
    OUT_TOPK,
    "w",
    encoding="utf-8",
) as f:
    for x in topk_outputs:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False
            ) + "\n"
        )


with open(
    OUT,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        audit,
        f,
        ensure_ascii=False,
        indent=2,
    )


print(
    json.dumps(
        audit,
        ensure_ascii=False,
        indent=2,
    )
)

print("SCM_RETRIEVAL_CONTROLLER_AUDIT_OK")
