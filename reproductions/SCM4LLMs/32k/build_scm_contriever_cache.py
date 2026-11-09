import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

MODEL = "/root/autodl-tmp/paper_repro/MemGAS/facebook/contriever"

UNIQUE = Path("unique_scm_memory_units.jsonl")
DATA = Path("personamem_scm_32k.jsonl")
OUT = Path("scm_contriever_cache.pt")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
)

model = AutoModel.from_pretrained(
    MODEL,
    local_files_only=True,
).cuda().eval()


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def embed_texts(texts, batch_size=16):
    all_emb = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        tok = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        tok = {
            k: v.cuda()
            for k, v in tok.items()
        }

        with torch.inference_mode():
            out = model(**tok)
            emb = mean_pool(
                out.last_hidden_state,
                tok["attention_mask"],
            )

        all_emb.append(
            emb.cpu()
        )

        print(
            f"[EMBED] {min(start + batch_size, len(texts))}/{len(texts)}",
            flush=True,
        )

    return torch.cat(all_emb, dim=0)


# ---------------------------------------------------------
# unique memory units
# ---------------------------------------------------------

units = []

with open(UNIQUE, encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        units.append(x)

unit_hashes = [
    x["unit_hash"]
    for x in units
]

unit_texts = [
    x["raw_text"]
    for x in units
]

assert len(unit_hashes) == 2287
assert len(set(unit_hashes)) == 2287

print("Embedding unique memories:", len(unit_texts))

memory_embeddings = embed_texts(
    unit_texts,
    batch_size=16,
)

assert memory_embeddings.shape == (2287, 768)


# ---------------------------------------------------------
# PersonaMem questions
# ---------------------------------------------------------

questions = []

with open(DATA, encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        questions.append(x)

assert len(questions) == 589

question_ids = [
    x["question_id"]
    for x in questions
]

question_texts = [
    x["question"]
    for x in questions
]

print("Embedding questions:", len(question_texts))

question_embeddings = embed_texts(
    question_texts,
    batch_size=32,
)

assert question_embeddings.shape == (589, 768)


cache = {
    "unit_hashes": unit_hashes,
    "memory_embeddings": memory_embeddings,
    "question_ids": question_ids,
    "question_embeddings": question_embeddings,
    "embedding_dim": 768,
    "normalized": False,
    "memory_embedding_source": "raw_text",
}

torch.save(cache, OUT)

print("memory_embeddings:", tuple(memory_embeddings.shape))
print("question_embeddings:", tuple(question_embeddings.shape))
print("saved:", OUT)
print("SCM_CONTRIEVER_CACHE_OK")
