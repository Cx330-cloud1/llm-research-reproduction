import json
import hashlib
from collections import Counter
from pathlib import Path
import tiktoken

SRC = Path("personamem_scm_32k.jsonl")
OUT_UNIQUE = Path("unique_scm_memory_units.jsonl")
OUT_MAP = Path("question_scm_unit_hashes.jsonl")
OUT_AUDIT = Path("scm_memory_unit_audit.json")

# SCM official code uses get_token_count_davinci()
# We use the davinci tokenizer-equivalent encoding for the same >300 decision.
try:
    enc = tiktoken.encoding_for_model("text-davinci-003")
except Exception:
    enc = tiktoken.get_encoding("p50k_base")


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


unique = {}
question_maps = []
total_instances = 0
summary_needed_instances = 0
summary_needed_unique = set()

size_dist = Counter()
token_lengths = []

with open(SRC, encoding="utf-8") as f:
    for line in f:
        q = json.loads(line)

        hashes = []

        for unit in q["scm_memory_units"]:
            raw = unit["raw_text"]
            uid = h(raw)

            # SCM summarize_embed_one_turn() wraps raw dialogue
            # in the official summarization prompt, and summarizes
            # only when the resulting prompt > 300 tokens.
            prompt = (
                "Below is a conversation between a user and an AI assistant. "
                "Please write a summary for each of them in one sentence and "
                "list them in separate paragraphs, while trying to preserve "
                "the key information of the user’s question and the assistant’s "
                "answer as much as possible.\n\n"
                "conversation content: \n\n"
                f"{raw}\n\n"
                "Summary:\n"
            )

            n_tok = len(enc.encode(prompt))

            total_instances += 1
            token_lengths.append(n_tok)
            size_dist[len(unit["messages"])] += 1

            if n_tok > 300:
                summary_needed_instances += 1
                summary_needed_unique.add(uid)

            if uid not in unique:
                unique[uid] = {
                    "unit_hash": uid,
                    "raw_text": raw,
                    "messages": unit["messages"],
                    "summary_prompt_tokens": n_tok,
                    "needs_summary": n_tok > 300,
                }

            hashes.append(uid)

        question_maps.append({
            "question_id": q["question_id"],
            "unit_hashes": hashes,
        })


with open(OUT_UNIQUE, "w", encoding="utf-8") as f:
    for x in unique.values():
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

with open(OUT_MAP, "w", encoding="utf-8") as f:
    for x in question_maps:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")


vals = sorted(token_lengths)

def percentile(p):
    if not vals:
        return None
    i = int(round((len(vals) - 1) * p))
    return vals[i]


audit = {
    "questions": len(question_maps),
    "total_unit_instances": total_instances,
    "unique_units": len(unique),
    "reused_instances": total_instances - len(unique),
    "reuse_ratio": 1 - len(unique) / total_instances,
    "summary_needed_instances": summary_needed_instances,
    "summary_needed_unique": len(summary_needed_unique),
    "summary_threshold_tokens": 300,
    "prompt_token_stats": {
        "min": min(vals),
        "median": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(vals),
    },
    "unit_size_distribution": dict(size_dist),
}

with open(OUT_AUDIT, "w", encoding="utf-8") as f:
    json.dump(audit, f, ensure_ascii=False, indent=2)

print(json.dumps(audit, ensure_ascii=False, indent=2))
print("SCM_MEMORY_UNIT_AUDIT_OK")
