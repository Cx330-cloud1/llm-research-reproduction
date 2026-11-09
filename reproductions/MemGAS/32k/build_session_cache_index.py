import json
import hashlib
from pathlib import Path

ROOT = Path("/root/autodl-tmp/paper_repro")
DATA = ROOT / "adapters/memgas_personamem/personamem_memgas.json"

UNIQUE_OUT = ROOT / "adapters/memgas_personamem/unique_sessions.jsonl"
MAP_OUT = ROOT / "adapters/memgas_personamem/question_session_hashes.jsonl"

data = json.load(open(DATA, encoding="utf-8"))

unique = {}
question_maps = []

for item in data:
    hashes = []

    for session in item["sessions"]:
        canonical = json.dumps(
            session,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        h = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

        if h in unique:
            assert unique[h]["session"] == session
        else:
            unique[h] = {
                "hash": h,
                "session": session,
                # Same joining rule as MemGAS multigran_generation.py
                "text": "\n\n".join(session),
            }

        hashes.append(h)

    question_maps.append({
        "conversation_id": item["conversation_id"],
        "question_id": item["adapter_meta"]["question_id"],
        "session_hashes": hashes,
    })

with UNIQUE_OUT.open("w", encoding="utf-8") as f:
    for x in unique.values():
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

with MAP_OUT.open("w", encoding="utf-8") as f:
    for x in question_maps:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

print("unique_sessions:", len(unique))
print("question_maps:", len(question_maps))

assert len(unique) == 305
assert len(question_maps) == 589

print("SESSION_INDEX_BUILD_OK")
