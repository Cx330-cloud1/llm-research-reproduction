import json
import argparse
import hashlib
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

MODEL = "/root/autodl-tmp/paper_repro/MemGAS/facebook/contriever"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def fingerprint(keys):
    return hashlib.sha256(
        "\n".join(keys).encode()
    ).hexdigest()


def embed(texts, keys, name, tokenizer, model,
          device, workdir, batch_size=16):

    ckpt = workdir / f"{name}.partial.pt"
    fp = fingerprint(keys)

    chunks = []
    done = 0

    if ckpt.exists():
        old = torch.load(
            ckpt, map_location="cpu", weights_only=False
        )
        assert old["fingerprint"] == fp
        previous = old["embeddings"].float()
        done = len(previous)
        assert done <= len(texts)
        assert previous.shape[1] == 768
        chunks.append(previous)
        print(f"[RESUME] {name}: {done}/{len(texts)}", flush=True)

    for start in range(done, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        tok = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}

        with torch.inference_mode():
            output = model(**tok)
            hidden = output.last_hidden_state
            mask = tok["attention_mask"].unsqueeze(-1).float()

            emb = (
                (hidden * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1e-9)
            )

        chunks.append(emb.cpu().float())
        current = min(start + batch_size, len(texts))

        if current % (batch_size * 100) < batch_size \
                or current == len(texts):
            merged = torch.cat(chunks, dim=0)
            atomic_save({
                "fingerprint": fp,
                "embeddings": merged,
            }, ckpt)
            chunks = [merged]
            print(
                f"[{name}] {current}/{len(texts)} CHECKPOINT",
                flush=True,
            )

    result = torch.cat(chunks, dim=0)
    assert result.shape == (len(texts), 768)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--expected-unique", type=int, required=True)
    ap.add_argument("--expected-questions", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    base = Path(args.dataset_dir)
    workdir = base / "cache_work"
    workdir.mkdir(exist_ok=True)

    out = base / "scm_contriever_cache.pt"

    if out.exists():
        raise RuntimeError(f"缓存已经存在，拒绝覆盖: {out}")

    unique = load_jsonl(
        base / "unique_scm_memory_units.jsonl"
    )
    questions = load_jsonl(
        base / "question_metadata.jsonl"
    )
    mappings = load_jsonl(
        base / "question_scm_unit_hashes.jsonl"
    )

    hashes = [x["unit_hash"] for x in unique]
    qids = [str(x["question_id"]) for x in questions]

    assert len(unique) == args.expected_unique
    assert len(set(hashes)) == len(hashes)
    assert len(questions) == args.expected_questions
    assert len(set(qids)) == len(qids)

    assert set(qids) == {
        str(x["question_id"]) for x in mappings
    }

    known = set(hashes)
    for row in mappings:
        assert all(h in known for h in row["unit_hashes"])

    memory_texts = [x["raw_text"] for x in unique]
    question_texts = [
        x.get("question", x.get("user_question_or_message"))
        for x in questions
    ]

    assert all(isinstance(x, str) and x.strip()
               for x in memory_texts + question_texts)

    assert torch.cuda.is_available(), "CUDA不可用，停止构建"

    device = torch.device("cuda:0")

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Memory units:", len(unique), flush=True)
    print("Questions:", len(questions), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        MODEL, local_files_only=True
    ).to(device).eval()

    assert model.config.hidden_size == 768

    memory_emb = embed(
        memory_texts, hashes, "memory_embeddings",
        tokenizer, model, device, workdir, args.batch_size
    )

    question_emb = embed(
        question_texts, qids, "question_embeddings",
        tokenizer, model, device, workdir, args.batch_size
    )

    cache = {
        "unit_hashes": hashes,
        "memory_embeddings": memory_emb,
        "question_ids": qids,
        "question_embeddings": question_emb,
        "embedding_dim": 768,
        "normalized": False,
        "memory_embedding_source": "raw_text",
    }

    atomic_save(cache, out)

    check = torch.load(out, map_location="cpu", weights_only=False)

    assert check["memory_embeddings"].shape == (
        args.expected_unique, 768
    )
    assert check["question_embeddings"].shape == (
        args.expected_questions, 768
    )

    for p in workdir.glob("*.partial.pt"):
        p.unlink()

    print("CACHE:", out, flush=True)
    print("MEMORY:", tuple(memory_emb.shape), flush=True)
    print("QUESTIONS:", tuple(question_emb.shape), flush=True)
    print("SCM_CONTRIEVER_CACHE_OK", flush=True)


if __name__ == "__main__":
    main()
