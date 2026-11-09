import csv
import json
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

ROOT = Path("/root/autodl-tmp/paper_repro")
BASE = ROOT / "adapters/memgas_personamem_1m"
DATA = ROOT / "shared_data/personamem_1m"

UNIQUE_PATH = BASE / "unique_sessions.jsonl"
GEN_PATH = BASE / "session_generations_qwen7b.jsonl"
MAP_PATH = BASE / "question_session_hashes.jsonl"
QUESTIONS_CSV = DATA / "questions_1M.csv"

MODEL_PATH = ROOT / "MemGAS/facebook/contriever"
OUT_PATH = BASE / "personamem_1m_contriever_cache.pt"
WORK = BASE / "cache_work"

BATCH_SIZE = int(os.environ.get("MEMGAS_EMB_BATCH_SIZE", "32"))
DEVICE = "cuda:0"

EXPECTED_SESSIONS = 2090
EXPECTED_QUESTIONS = 2674


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Invalid JSON: {path}:{line_no}: {e}"
                )
    return rows


def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def check_tensor(tensor, rows, dim, name):
    assert isinstance(tensor, torch.Tensor), name
    assert tuple(tensor.shape) == (rows, dim), (
        name, tuple(tensor.shape), (rows, dim)
    )
    assert torch.isfinite(tensor).all().item(), (
        f"Non-finite values in {name}"
    )


def main():
    if OUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUT_PATH}\n"
            "Refusing to overwrite an existing cache."
        )

    assert torch.cuda.is_available(), "CUDA is unavailable"
    assert BATCH_SIZE > 0
    assert MODEL_PATH.exists(), f"Missing model: {MODEL_PATH}"

    WORK.mkdir(parents=True, exist_ok=True)

    print("===== LOAD INPUTS =====", flush=True)

    sessions = load_jsonl(UNIQUE_PATH)
    generations = load_jsonl(GEN_PATH)
    mappings = load_jsonl(MAP_PATH)

    assert len(sessions) == EXPECTED_SESSIONS, len(sessions)
    assert len(generations) == EXPECTED_SESSIONS, len(generations)
    assert len(mappings) == EXPECTED_QUESTIONS, len(mappings)

    session_hashes = [x["hash"] for x in sessions]
    assert len(set(session_hashes)) == EXPECTED_SESSIONS

    gen_by_hash = {}
    for x in generations:
        h = x["hash"]
        assert h not in gen_by_hash, f"Duplicate generation: {h}"
        assert isinstance(x.get("summary"), str) and x["summary"].strip()
        assert isinstance(x.get("keywords"), str) and x["keywords"].strip()
        gen_by_hash[h] = x

    assert set(session_hashes) == set(gen_by_hash)

    questions = []
    with QUESTIONS_CSV.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)
        required = {
            "question_id",
            "persona_id",
            "question_type",
            "user_question_or_message",
            "correct_answer",
            "shared_context_id",
            "end_index_in_shared_context",
        }
        missing = required - set(reader.fieldnames or [])
        assert not missing, f"Missing CSV columns: {missing}"
        questions = list(reader)

    assert len(questions) == EXPECTED_QUESTIONS

    question_ids = [x["question_id"] for x in questions]
    assert len(set(question_ids)) == EXPECTED_QUESTIONS

    map_by_id = {}
    for x in mappings:
        qid = x["question_id"]
        assert qid not in map_by_id, f"Duplicate mapping: {qid}"
        map_by_id[qid] = x

    assert set(question_ids) == set(map_by_id)

    session_index_by_hash = {
        h: i for i, h in enumerate(session_hashes)
    }
    question_session_hashes = []

    for q in questions:
        qid = q["question_id"]
        m = map_by_id[qid]
        hashes = m["session_hashes"]

        assert isinstance(hashes, list) and len(hashes) > 0
        assert all(h in session_index_by_hash for h in hashes), (
            f"Unknown session hash in question {qid}"
        )
        assert str(m["shared_context_id"]) == str(q["shared_context_id"])
        assert int(m["end_index"]) == int(
            q["end_index_in_shared_context"]
        )

        question_session_hashes.append(hashes)

    print("sessions:", len(sessions), flush=True)
    print("questions:", len(questions), flush=True)
    print("session instances across questions:",
          sum(map(len, question_session_hashes)), flush=True)
    print("embedding batch size:", BATCH_SIZE, flush=True)

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

    dim = model.config.hidden_size
    print("device:", DEVICE, flush=True)
    print("hidden_size:", dim, flush=True)

    @torch.inference_mode()
    def embed_texts(texts, label):
        assert len(texts) > 0, f"Empty input: {label}"
        chunks = []
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_no, start in enumerate(
            range(0, len(texts), BATCH_SIZE), 1
        ):
            batch = texts[start:start + BATCH_SIZE]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)

            # Match the original MemGAS masked mean pooling.
            emb = (
                (hidden * mask).sum(dim=1)
                / mask.sum(dim=1)
            )

            chunks.append(emb.float().cpu())

            done = min(start + len(batch), len(texts))
            if batch_no % 100 == 0 or batch_no == total_batches:
                print(
                    f"{label}: {done}/{len(texts)} "
                    f"({done / len(texts) * 100:.1f}%)",
                    flush=True,
                )

        result = torch.cat(chunks, dim=0)
        del chunks

        assert tuple(result.shape) == (len(texts), dim)
        assert torch.isfinite(result).all().item()

        return result

    def stage(name, texts):
        path = WORK / f"{name}.pt"

        if path.exists():
            tensor = torch.load(path, map_location="cpu")
            check_tensor(tensor, len(texts), dim, name)
            print(
                f"RESUME {name}: loaded {tuple(tensor.shape)}",
                flush=True,
            )
            return tensor

        print(f"\n===== EMBED {name} =====", flush=True)
        tensor = embed_texts(texts, name)
        atomic_save(tensor, path)
        print(f"CHECKPOINT SAVED: {path}", flush=True)
        return tensor

    # Build session texts in exactly the same way as the original.
    raw_texts = []
    turn_lists = []
    summary_texts = []
    keyword_texts = []
    hybrid_texts = []

    for x in sessions:
        h = x["hash"]
        session = x["session"]
        assert isinstance(session, list) and len(session) > 0
        assert all(isinstance(t, str) for t in session)

        raw = "\n".join(session)
        g = gen_by_hash[h]

        summary = g["summary"]
        keywords = g["keywords"]

        raw_texts.append(raw)
        turn_lists.append(session)
        summary_texts.append(summary)
        keyword_texts.append(keywords)
        hybrid_texts.append(f"{summary} {keywords} {raw}")

    # 1. Session-level embedding.
    raw_emb = stage("raw", raw_texts)

    # 2. Turn-level: encode each message independently,
    # then average message embeddings within each session.
    flat_turns = []
    turn_ranges = []
    offset = 0

    for session in turn_lists:
        start = offset
        flat_turns.extend(session)
        offset += len(session)
        turn_ranges.append((start, offset))

    print("\ntotal turn-like messages:", len(flat_turns), flush=True)

    turn_mean_path = WORK / "turn_mean.pt"

    if turn_mean_path.exists():
        turn_mean_emb = torch.load(
            turn_mean_path, map_location="cpu"
        )
        check_tensor(
            turn_mean_emb, EXPECTED_SESSIONS, dim, "turn_mean"
        )
        print("RESUME turn_mean", flush=True)
    else:
        flat_turn_emb = stage("turn_messages", flat_turns)

        means = []
        for start, end in turn_ranges:
            assert end > start
            means.append(flat_turn_emb[start:end].mean(dim=0))

        turn_mean_emb = torch.stack(means, dim=0)
        check_tensor(
            turn_mean_emb, EXPECTED_SESSIONS, dim, "turn_mean"
        )
        atomic_save(turn_mean_emb, turn_mean_path)
        print("CHECKPOINT SAVED: turn_mean", flush=True)

        del flat_turn_emb

        # turn_mean is safely checkpointed, so the larger intermediate
        # message tensor is no longer required.
        flat_path = WORK / "turn_messages.pt"
        if flat_path.exists():
            flat_path.unlink()

    # 3. Summary-level.
    summary_emb = stage("summary", summary_texts)

    # 4. Keyword-level.
    keyword_emb = stage("keyword", keyword_texts)

    # 5. Hybrid-level.
    hybrid_emb = stage("hybrid", hybrid_texts)

    # 6. Question embeddings.
    question_texts = [
        x["user_question_or_message"] for x in questions
    ]
    assert all(isinstance(x, str) and x.strip() for x in question_texts)

    question_emb = stage("question", question_texts)

    print("\n===== FINAL CACHE AUDIT =====", flush=True)

    expected_session_shape = (EXPECTED_SESSIONS, dim)
    expected_question_shape = (EXPECTED_QUESTIONS, dim)

    for name, tensor in {
        "raw": raw_emb,
        "turn_mean": turn_mean_emb,
        "summary": summary_emb,
        "keyword": keyword_emb,
        "hybrid": hybrid_emb,
    }.items():
        assert tuple(tensor.shape) == expected_session_shape, (
            name, tensor.shape
        )
        assert torch.isfinite(tensor).all().item(), name

    assert tuple(question_emb.shape) == expected_question_shape
    assert torch.isfinite(question_emb).all().item()

    cache = {
        "meta": {
            "dataset": "PersonaMem-1M",
            "adapter": "memgas_personamem_1m",
            "embedding_model": str(MODEL_PATH),
            "embedding_model_class": type(model).__name__,
            "pooling": "masked_mean",
            "l2_normalized": False,
            "tokenizer_padding": True,
            "tokenizer_truncation": True,
            "raw_session_join": "\\n",
            "hybrid_format": "{summary} {keywords} {raw}",
            "turn_aggregation": "mean_of_message_embeddings",
            "num_unique_sessions": EXPECTED_SESSIONS,
            "num_questions": EXPECTED_QUESTIONS,
            "num_turn_like_messages": len(flat_turns),
            "embedding_dim": dim,
            "batch_size": BATCH_SIZE,
        },
        "session_hashes": session_hashes,
        "session_index_by_hash": session_index_by_hash,
        "raw": raw_emb,
        "turn_mean": turn_mean_emb,
        "summary": summary_emb,
        "keyword": keyword_emb,
        "hybrid": hybrid_emb,
        "question_ids": question_ids,
        "persona_ids": [x["persona_id"] for x in questions],
        "question_types": [x["question_type"] for x in questions],
        "question_texts": question_texts,
        "question_answers": [x["correct_answer"] for x in questions],
        "question_topics": [x.get("topic", "") for x in questions],
        "question_context_ids": [
            x["shared_context_id"] for x in questions
        ],
        "question_end_indices": [
            int(x["end_index_in_shared_context"]) for x in questions
        ],
        "question_session_hashes": question_session_hashes,
        "question_session_indices": [
            [session_index_by_hash[h] for h in hashes]
            for hashes in question_session_hashes
        ],
        "all_options": [x.get("all_options", "") for x in questions],
        "question_embeddings": question_emb,
    }

    print("saving:", OUT_PATH, flush=True)
    atomic_save(cache, OUT_PATH)

    # Reload and verify the actual serialized file.
    saved = torch.load(OUT_PATH, map_location="cpu")
    assert saved["meta"]["dataset"] == "PersonaMem-1M"
    assert len(saved["session_hashes"]) == EXPECTED_SESSIONS
    assert len(saved["question_ids"]) == EXPECTED_QUESTIONS

    for key in [
        "raw", "turn_mean", "summary", "keyword", "hybrid"
    ]:
        assert tuple(saved[key].shape) == expected_session_shape

    assert tuple(saved["question_embeddings"].shape) == expected_question_shape
    assert len(saved["question_session_hashes"]) == EXPECTED_QUESTIONS
    assert len(saved["question_session_indices"]) == EXPECTED_QUESTIONS

    print("\n===== CACHE BUILD SUCCESS =====", flush=True)
    print("cache:", OUT_PATH, flush=True)
    print("sessions:", len(saved["session_hashes"]), flush=True)
    print("questions:", len(saved["question_ids"]), flush=True)
    print("embedding_dim:", dim, flush=True)
    print("MEMGAS_PERSONAMEM_1M_CACHE_OK", flush=True)


if __name__ == "__main__":
    main()
