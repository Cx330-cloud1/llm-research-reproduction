import os
import json
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

ROOT = Path("/root/autodl-tmp/paper_repro/MemGAS")
DATA = ROOT / "data/process_data/longmemeval_s.json"
MODEL_DIR = ROOT / "facebook/contriever"

SUMMARY = ROOT / "multi_granularity_logs/longmemeval-summary_level.jsonl"
KEYWORD = ROOT / "multi_granularity_logs/longmemeval-keyword_level.jsonl"

OUT = ROOT / "data/process_embs/longmemeval_s-contriever-emb.pt"
CKPT = ROOT / "repro_logs/longmemeval_s_embedding_checkpoints"
MANIFEST = CKPT / "manifest.json"

BATCH_SIZE = 64
EXPECTED_SESSIONS = 19545


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_granularity(path):
    result = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            overlap = result.keys() & obj.keys()
            if overlap:
                raise ValueError(f"Duplicate IDs in {path}: {list(overlap)[:3]}")
            result.update(obj)
    return result


def atomic_torch_save(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def safe_torch_load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    for p in [DATA, SUMMARY, KEYWORD, MODEL_DIR / "config.json"]:
        if not p.exists():
            raise FileNotFoundError(f"Required input missing: {p}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start")

    if OUT.exists():
        raise RuntimeError(
            f"Final output already exists: {OUT}\n"
            "Refusing to overwrite it."
        )

    data = json.loads(DATA.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 500:
        raise ValueError(f"Expected 500 records, got {len(data)}")

    print("Loading summary/keyword maps...", flush=True)
    summaries = read_granularity(SUMMARY)
    keywords = read_granularity(KEYWORD)

    session_ids = {
        sid
        for conv in data
        for sid in conv["sessions_ids"]
    }

    print("records:", len(data))
    print("unique_session_ids:", len(session_ids))
    print("summary_ids:", len(summaries))
    print("keyword_ids:", len(keywords))

    if len(session_ids) != EXPECTED_SESSIONS:
        raise ValueError(f"Unexpected unique session count: {len(session_ids)}")

    if session_ids != set(summaries):
        raise ValueError("Summary SID set does not match processed data")

    if session_ids != set(keywords):
        raise ValueError("Keyword SID set does not match processed data")

    for name, mapping in [("summary", summaries), ("keyword", keywords)]:
        empty = [k for k, v in mapping.items() if not str(v).strip()]
        if empty:
            raise ValueError(f"{name} has {len(empty)} empty values")

    # Fingerprint inputs so checkpoints cannot silently be reused
    # after the source data or generated texts have changed.
    manifest = {
        "format_version": 1,
        "record_count": len(data),
        "session_count": len(session_ids),
        "data_sha256": sha256(DATA),
        "summary_sha256": sha256(SUMMARY),
        "keyword_sha256": sha256(KEYWORD),
        "model_config_sha256": sha256(MODEL_DIR / "config.json"),
        "batch_size": BATCH_SIZE,
        "pooling": "attention-mask mean pooling",
        "truncation": True,
    }

    CKPT.mkdir(parents=True, exist_ok=True)
    checkpoint_files = list(CKPT.glob("record_*.pt"))

    if MANIFEST.exists():
        old_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if old_manifest != manifest:
            raise RuntimeError(
                "Checkpoint manifest differs from current inputs. "
                "Do not reuse these checkpoints."
            )
    else:
        if checkpoint_files:
            raise RuntimeError(
                "Checkpoints exist but manifest is missing; refusing unsafe resume."
            )
        tmp = MANIFEST.with_name(MANIFEST.name + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp, MANIFEST)

    for p in checkpoint_files:
        idx = int(p.stem.split("_")[1])
        if idx < 0 or idx >= len(data):
            raise RuntimeError(f"Unexpected checkpoint index: {p}")

    device = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Loading local Contriever:", MODEL_DIR, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    ).to(device)
    model.eval()

    hidden_size = model.config.hidden_size

    @torch.inference_mode()
    def embed_texts(texts):
        if not texts:
            return torch.empty((0, hidden_size), dtype=torch.float32)

        vectors = []
        loader = DataLoader(texts, batch_size=BATCH_SIZE, shuffle=False)

        for batch in loader:
            inputs = tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            token_embeddings = model(**inputs)[0]
            mask = inputs["attention_mask"]

            # Same attention-mask mean pooling as official construct_emb.py.
            token_embeddings = token_embeddings.masked_fill(
                ~mask[..., None].bool(), 0.0
            )
            sentence_embeddings = (
                token_embeddings.sum(dim=1)
                / mask.sum(dim=1)[..., None]
            )
            vectors.append(sentence_embeddings.detach().cpu())

        return torch.cat(vectors, dim=0)

    all_emb = []

    for i, conv in enumerate(
        tqdm(data, desc="LongMemEval-S embedding", unit="conversation")
    ):
        cp = CKPT / f"record_{i:05d}.pt"

        if cp.exists():
            item = safe_torch_load(cp)
            if item.get("conversation_id") != conv["conversation_id"]:
                raise RuntimeError(f"Checkpoint/data mismatch: {cp}")
            all_emb.append(item)
            continue

        questions = [qa["question"] for qa in conv["qa"]]
        sessions = ["\n".join(s) for s in conv["sessions"]]
        turns = [turn for session in conv["sessions"] for turn in session]

        if len(sessions) != len(conv["sessions_ids"]):
            raise ValueError(f"Session/SID alignment error at record {i}")

        summarys = [str(summaries[sid]) for sid in conv["sessions_ids"]]
        keyword_texts = [str(keywords[sid]) for sid in conv["sessions_ids"]]

        hybrid = [
            f"{summary} {keyword} {session}"
            for session, summary, keyword
            in zip(sessions, summarys, keyword_texts)
        ]

        # Keep the official embedding fields and output schema.
        item = {
            "conversation_id": conv["conversation_id"],
            "questions": embed_texts(questions),
            "sessions": embed_texts(sessions),
            "turns": embed_texts(turns),
            "summarys": embed_texts(summarys),
            "keywords": embed_texts(keyword_texts),
            "hybrid": embed_texts(hybrid),
        }

        atomic_torch_save(item, cp)
        all_emb.append(item)

        if (i + 1) % 10 == 0 or i == len(data) - 1:
            print(
                f"\nCHECKPOINT {i + 1}/{len(data)} "
                f"conversation_id={conv['conversation_id']}",
                flush=True,
            )

    if len(all_emb) != len(data):
        raise RuntimeError(f"Output record count mismatch: {len(all_emb)}")

    for i, (item, conv) in enumerate(zip(all_emb, data)):
        if item["conversation_id"] != conv["conversation_id"]:
            raise RuntimeError(f"Final order mismatch at record {i}")

        for field in [
            "questions", "sessions", "turns",
            "summarys", "keywords", "hybrid"
        ]:
            if not isinstance(item[field], torch.Tensor):
                raise TypeError(f"Record {i}: {field} is not a tensor")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(all_emb, OUT)

    print("OUTPUT:", OUT, flush=True)
    print("OUTPUT_RECORDS:", len(all_emb), flush=True)
    print("LONGMEMEVAL_S_EMBEDDING_OK", flush=True)


if __name__ == "__main__":
    main()
