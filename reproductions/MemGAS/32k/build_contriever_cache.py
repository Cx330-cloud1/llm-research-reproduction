import csv
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

BASE = Path("/root/autodl-tmp/paper_repro/adapters/memgas_personamem")
MODEL_PATH = Path("/root/autodl-tmp/paper_repro/MemGAS/facebook/contriever")
QUESTIONS_CSV = Path(
    "/root/autodl-tmp/paper_repro/shared_data/personamem_32k/questions_32k.csv"
)

UNIQUE_PATH = BASE / "unique_sessions.jsonl"
GEN_PATH = BASE / "session_generations_qwen7b.jsonl"
OUT_PATH = BASE / "personamem_contriever_cache.pt"

BATCH_SIZE = 32
DEVICE = "cuda:0"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


print("===== LOAD INPUTS =====", flush=True)

sessions = load_jsonl(UNIQUE_PATH)
generations = load_jsonl(GEN_PATH)

assert len(sessions) == 305, len(sessions)
assert len(generations) == 305, len(generations)

gen_by_hash = {}
for x in generations:
    h = x["hash"]
    assert h not in gen_by_hash
    gen_by_hash[h] = x

session_hashes = [x["hash"] for x in sessions]

assert len(set(session_hashes)) == 305
assert set(session_hashes) == set(gen_by_hash)

questions = []
with open(QUESTIONS_CSV, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        questions.append(row)

assert len(questions) == 589, len(questions)

print("unique_sessions:", len(sessions), flush=True)
print("questions:", len(questions), flush=True)


print("\n===== LOAD CONTRIEVER =====", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)

model = AutoModel.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
).to(DEVICE)

model.eval()

print("model:", type(model).__name__, flush=True)
print("hidden_size:", model.config.hidden_size, flush=True)


@torch.inference_mode()
def embed_texts(texts, label):
    """
    MemGAS/Contriever-compatible:
      tokenizer(... padding=True, truncation=True)
      masked mean pooling
      no explicit L2 normalization
    """
    outputs_all = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
        }

        outputs = model(**inputs)

        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)

        emb = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1)
        )

        outputs_all.append(emb.cpu())

        done = min(start + BATCH_SIZE, len(texts))
        print(
            f"{label}: {done}/{len(texts)}",
            flush=True,
        )

    result = torch.cat(outputs_all, dim=0)

    assert result.shape[0] == len(texts)
    assert result.shape[1] == model.config.hidden_size

    return result


# ------------------------------------------------------------
# 1. Raw session
# IMPORTANT:
# MemGAS embedding uses '\n'.join(session),
# NOT the '\n\n'.join(session) used by generation.
# ------------------------------------------------------------

raw_texts = []
turn_lists = []
summary_texts = []
keyword_texts = []
hybrid_texts = []

for x in sessions:
    h = x["hash"]
    session = x["session"]

    assert isinstance(session, list)
    assert len(session) > 0

    raw = "\n".join(session)

    g = gen_by_hash[h]
    summary = g["summary"]
    keywords = g["keywords"]

    assert isinstance(summary, str)
    assert isinstance(keywords, str)

    raw_texts.append(raw)
    turn_lists.append(session)
    summary_texts.append(summary)
    keyword_texts.append(keywords)

    # Match MemGAS hybrid construction:
    # summary + keyword + raw session
    hybrid_texts.append(
        f"{summary} {keywords} {raw}"
    )


print("\n===== SESSION LEVEL =====", flush=True)
raw_emb = embed_texts(raw_texts, "raw")


print("\n===== TURN LEVEL =====", flush=True)

# Official MemGAS turn representation:
# embed each turn/message independently,
# then mean within each session.

flat_turns = []
turn_ranges = []

offset = 0

for session in turn_lists:
    start = offset
    flat_turns.extend(session)
    offset += len(session)
    turn_ranges.append((start, offset))

print("total_turn_like_messages:", len(flat_turns), flush=True)

flat_turn_emb = embed_texts(
    flat_turns,
    "turn-message",
)

turn_mean_emb = []

for start, end in turn_ranges:
    assert end > start
    turn_mean_emb.append(
        flat_turn_emb[start:end].mean(dim=0)
    )

turn_mean_emb = torch.stack(
    turn_mean_emb,
    dim=0,
)


print("\n===== SUMMARY LEVEL =====", flush=True)
summary_emb = embed_texts(
    summary_texts,
    "summary",
)


print("\n===== KEYWORD LEVEL =====", flush=True)
keyword_emb = embed_texts(
    keyword_texts,
    "keyword",
)


print("\n===== HYBRID LEVEL =====", flush=True)
hybrid_emb = embed_texts(
    hybrid_texts,
    "hybrid",
)


print("\n===== QUESTION EMBEDDINGS =====", flush=True)

question_texts = [
    x["user_question_or_message"]
    for x in questions
]

question_emb = embed_texts(
    question_texts,
    "question",
)


# ------------------------------------------------------------
# Audit
# ------------------------------------------------------------

expected_session_shape = (305, model.config.hidden_size)
expected_question_shape = (589, model.config.hidden_size)

for name, tensor in {
    "raw": raw_emb,
    "turn_mean": turn_mean_emb,
    "summary": summary_emb,
    "keyword": keyword_emb,
    "hybrid": hybrid_emb,
}.items():
    assert tuple(tensor.shape) == expected_session_shape, (
        name,
        tensor.shape,
    )
    assert torch.isfinite(tensor).all(), name

assert tuple(question_emb.shape) == expected_question_shape
assert torch.isfinite(question_emb).all()


cache = {
    "meta": {
        "dataset": "PersonaMem-32k",
        "adapter": "memgas_personamem",
        "embedding_model": str(MODEL_PATH),
        "embedding_model_class": type(model).__name__,
        "pooling": "masked_mean",
        "l2_normalized": False,
        "tokenizer_padding": True,
        "tokenizer_truncation": True,
        "raw_session_join": "\\n",
        "num_unique_sessions": 305,
        "num_questions": 589,
        "num_turn_like_messages": len(flat_turns),
        "embedding_dim": model.config.hidden_size,
    },

    "session_hashes": session_hashes,

    "raw": raw_emb,
    "turn_mean": turn_mean_emb,
    "summary": summary_emb,
    "keyword": keyword_emb,
    "hybrid": hybrid_emb,

    "question_ids": [
        x["question_id"]
        for x in questions
    ],

    "persona_ids": [
        x["persona_id"]
        for x in questions
    ],

    "question_types": [
        x["question_type"]
        for x in questions
    ],

    "question_texts": question_texts,
    "question_embeddings": question_emb,
}


torch.save(cache, OUT_PATH)

print("\n===== FINAL =====", flush=True)
print("saved:", OUT_PATH, flush=True)

for k in [
    "raw",
    "turn_mean",
    "summary",
    "keyword",
    "hybrid",
    "question_embeddings",
]:
    print(k, tuple(cache[k].shape), flush=True)

print("PERSONAMEM_CONTRIEVER_CACHE_OK", flush=True)
