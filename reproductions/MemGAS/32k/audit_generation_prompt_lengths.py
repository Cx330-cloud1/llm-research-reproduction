import ast
import json
from pathlib import Path

from transformers import AutoTokenizer

BASE = Path("/root/autodl-tmp/paper_repro/adapters/memgas_personamem")
MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"

RETRIEVAL = BASE / "personamem_memgas_retrieval.jsonl"
GEN_CACHE = BASE / "session_generations_qwen7b.jsonl"

PROMPT_MULTIGRAN = """
You are an intelligent dialog bot. You will be shown History Dialogs and corresponding multi-granular information.
Filter the History Dialogs, summaries, and keywords to extract only the parts directly relevant to the Question. Preserve original tokens, do not paraphrase. Remove irrelevant turns, redundant info, and non-essential details.

History Dialogs: {retrieved_texts}

Question: {question}
Answer:
"""

tok = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
)

gens = {}
with open(GEN_CACHE, encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        gens[x["hash"]] = x

rows = []
with open(RETRIEVAL, encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

lengths = []

for x in rows:
    retrieved_texts = ""

    for item in x["retrieval_results"]["ranked_items"][:3]:
        h = item["session_hash"]
        session = item["session"]

        # Match MemGAS generation representation:
        # Python/list-like session object inserted as text.
        session_text = str(session)

        g = gens[h]

        retrieved_texts += (
            "\nSession Content:\n"
            f"{session_text}\n\n"
            "Session Summary:\n"
            f"{g['summary']}\n"
            "Session Keyword:\n"
            f"{g['keywords']}\n"
        )

    prompt = PROMPT_MULTIGRAN.format(
        retrieved_texts=retrieved_texts,
        question=x["question"],
    )

    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    n = len(tok(text, add_special_tokens=False)["input_ids"])

    lengths.append((n, x["question_id"]))

lengths.sort(reverse=True)

vals = [x[0] for x in lengths]

print("questions:", len(vals))
print("max:", max(vals))
print("p99:", vals[max(0, int(len(vals)*0.01)-1)])
print("p95:", vals[max(0, int(len(vals)*0.05)-1)])
print("median:", sorted(vals)[len(vals)//2])

print("\nTOP20")
for n, qid in lengths[:20]:
    print(n, qid)

print("GENERATION_PROMPT_LENGTH_AUDIT_OK")
