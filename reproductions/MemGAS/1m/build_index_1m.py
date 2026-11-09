import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/paper_repro")

QUESTIONS = ROOT / "shared_data/personamem_1m/questions_1M.csv"
CONTEXTS = ROOT / "shared_data/personamem_1m/shared_contexts_1M.jsonl"

OUT_DIR = ROOT / "adapters/memgas_personamem_1m"

UNIQUE_OUT = OUT_DIR / "unique_sessions.jsonl"
MAP_OUT = OUT_DIR / "question_session_hashes.jsonl"
AUDIT_OUT = OUT_DIR / "index_audit.json"


def split_sessions(visible_messages):
    """
    Same PersonaMem -> MemGAS session conversion as the 32K adapter:
      - every system message starts a new session
      - system content is retained
      - message contents are retained verbatim
    """
    sessions = []
    current = []

    for msg in visible_messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            if current:
                sessions.append(current)
            current = [content]
        else:
            if not current:
                current = []
            current.append(content)

    if current:
        sessions.append(current)

    return sessions


def session_hash(session):
    """
    Adapter-only deterministic identifier.
    Hash values themselves do not participate in retrieval scoring.
    """
    payload = json.dumps(
        session,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("===== LOAD CONTEXTS =====", flush=True)

    contexts = {}

    with CONTEXTS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            assert len(row) == 1

            cid, messages = next(iter(row.items()))

            assert cid not in contexts
            assert isinstance(messages, list)

            contexts[cid] = messages

    print("contexts:", len(contexts), flush=True)

    print("===== LOAD QUESTIONS =====", flush=True)

    with QUESTIONS.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:
        questions = list(csv.DictReader(f))

    qids = [q["question_id"] for q in questions]

    assert len(qids) == len(set(qids))

    referenced = {
        q["shared_context_id"]
        for q in questions
    }

    missing = referenced - set(contexts)

    if missing:
        raise KeyError(
            f"missing shared contexts: {sorted(missing)[:10]}"
        )

    print("questions:", len(questions), flush=True)
    print("referenced contexts:", len(referenced), flush=True)

    unique_sessions = {}
    question_rows = []

    total_session_instances = 0
    future_leakage_count = 0
    bad_end_index_count = 0

    session_min = None
    session_max = None
    visible_min = None
    visible_max = None

    print("===== BUILD INDEX =====", flush=True)

    for i, row in enumerate(questions, 1):
        qid = row["question_id"]
        cid = row["shared_context_id"]

        full_context = contexts[cid]
        end_index = int(row["end_index_in_shared_context"])

        if not (0 <= end_index <= len(full_context)):
            bad_end_index_count += 1
            raise ValueError(
                f"bad end_index={end_index} "
                f"context_len={len(full_context)} "
                f"question_id={qid}"
            )

        # Critical PersonaMem leakage boundary.
        visible_context = full_context[:end_index]

        sessions = split_sessions(visible_context)

        reconstructed = sum(len(s) for s in sessions)

        if reconstructed != len(visible_context):
            future_leakage_count += 1
            raise AssertionError(
                f"message preservation mismatch "
                f"qid={qid} "
                f"visible={len(visible_context)} "
                f"reconstructed={reconstructed}"
            )

        if not sessions:
            raise AssertionError(
                f"no visible sessions for question {qid}"
            )

        hashes = []

        for session in sessions:
            h = session_hash(session)

            if h in unique_sessions:
                if unique_sessions[h] != session:
                    raise RuntimeError(
                        "SHA256 collision / inconsistent session"
                    )
            else:
                unique_sessions[h] = session

            hashes.append(h)

        question_rows.append({
            "question_id": qid,
            "shared_context_id": cid,
            "end_index": end_index,
            "session_hashes": hashes,
        })

        sc = len(sessions)
        vm = len(visible_context)

        total_session_instances += sc

        session_min = sc if session_min is None else min(session_min, sc)
        session_max = sc if session_max is None else max(session_max, sc)

        visible_min = vm if visible_min is None else min(visible_min, vm)
        visible_max = vm if visible_max is None else max(visible_max, vm)

        if i % 100 == 0 or i == len(questions):
            print(
                f"questions indexed: {i}/{len(questions)} "
                f"unique_sessions={len(unique_sessions)}",
                flush=True,
            )

    # Write unique session contents once.
    with UNIQUE_OUT.open("w", encoding="utf-8") as f:
        for h, session in unique_sessions.items():
            f.write(
                json.dumps(
                    {
                        "hash": h,
                        "session": session,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Per question: only ordered session identifiers.
    with MAP_OUT.open("w", encoding="utf-8") as f:
        for x in question_rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    audit = {
        "dataset": "PersonaMem-1M",
        "questions": len(questions),
        "context_records": len(contexts),
        "referenced_contexts": len(referenced),
        "unused_contexts": len(set(contexts) - referenced),
        "unique_sessions": len(unique_sessions),
        "session_instances": total_session_instances,
        "reuse_ratio": (
            1.0 - len(unique_sessions) / total_session_instances
            if total_session_instances else 0.0
        ),
        "session_count_min": session_min,
        "session_count_max": session_max,
        "visible_message_min": visible_min,
        "visible_message_max": visible_max,
        "bad_end_index_count": bad_end_index_count,
        "future_leakage_count": future_leakage_count,
        "hash_scheme": "sha256(canonical-json-session)",
        "note": (
            "Low-disk adapter: does not materialize duplicated "
            "full per-question session JSON."
        ),
    }

    AUDIT_OUT.write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("===== AUDIT =====")
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    print()
    print("unique_out:", UNIQUE_OUT)
    print("map_out:", MAP_OUT)
    print("audit_out:", AUDIT_OUT)
    print("MEMGAS_PERSONAMEM_1M_INDEX_OK")


if __name__ == "__main__":
    main()
