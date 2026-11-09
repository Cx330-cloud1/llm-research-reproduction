import csv
import json
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/paper_repro")
BASE = ROOT / "adapters/memgas_personamem_128k"
DATA = ROOT / "shared_data/personamem_32k"

SESSION_PATH = BASE / "unique_sessions.jsonl"
GEN_PATH = BASE / "session_generations_qwen7b.jsonl"
MAP_PATH = BASE / "question_session_hashes.jsonl"
QUESTIONS_PATH = DATA / "questions_128k.csv"
MODEL_PATH = ROOT / "MemGAS/facebook/contriever"
CACHE_PATH = BASE / "personamem_128k_contriever_cache.pt"
BUILDER = BASE / "build_contriever_cache_128k.py"

EXPECTED_SESSIONS = 1249
EXPECTED_QUESTIONS = 2727
errors = []
blockers = []

def check(condition, message):
    if not condition:
        errors.append(message)

def read_jsonl(path):
    if not path.is_file():
        errors.append(f"Missing file: {path}")
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                errors.append(f"Invalid JSON {path}:{line_no}: {exc}")
    return rows

print("===== PERSONAMEM-128K CACHE PREFLIGHT =====")
sessions = read_jsonl(SESSION_PATH)
generations = read_jsonl(GEN_PATH)
mappings = read_jsonl(MAP_PATH)

check(len(sessions) == EXPECTED_SESSIONS,
      f"Expected {EXPECTED_SESSIONS} sessions; got {len(sessions)}")

session_hashes = [x.get("hash") for x in sessions]
check(all(isinstance(h, str) and h for h in session_hashes),
      "Missing/invalid session hash")
session_set = set(session_hashes)
check(len(session_set) == len(session_hashes),
      "Duplicate session hashes")

try:
    with QUESTIONS_PATH.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        questions = list(reader)
        columns = set(reader.fieldnames or [])
except Exception as exc:
    errors.append(f"Cannot read questions CSV: {exc}")
    questions = []
    columns = set()

required = {
    "question_id", "persona_id", "question_type",
    "user_question_or_message", "correct_answer",
    "shared_context_id", "end_index_in_shared_context",
}
check(required <= columns, f"Missing CSV columns: {sorted(required-columns)}")
check(len(questions) == EXPECTED_QUESTIONS,
      f"Expected {EXPECTED_QUESTIONS} questions; got {len(questions)}")

qids = [x.get("question_id") for x in questions]
qid_set = set(qids)
check(len(qid_set) == len(qids), "Duplicate question IDs")

map_ids = [x.get("question_id") for x in mappings]
map_set = set(map_ids)
check(len(mappings) == EXPECTED_QUESTIONS,
      f"Expected {EXPECTED_QUESTIONS} mappings; got {len(mappings)}")
check(len(map_set) == len(map_ids), "Duplicate mapping question IDs")
check(map_set == qid_set, "Mapping IDs do not match question IDs")

q_by_id = {x["question_id"]: x for x in questions}
m_by_id = {x["question_id"]: x for x in mappings
           if x.get("question_id") is not None}

session_instances = 0
for qid in qid_set & map_set:
    q, m = q_by_id[qid], m_by_id[qid]
    check(str(m.get("shared_context_id")) == str(q.get("shared_context_id")),
          f"Context mismatch: {qid}")
    try:
        check(int(m["end_index"]) == int(q["end_index_in_shared_context"]),
              f"End-index mismatch: {qid}")
    except (KeyError, ValueError, TypeError):
        errors.append(f"Invalid end-index mapping: {qid}")

    hs = m.get("session_hashes")
    check(isinstance(hs, list) and len(hs) > 0,
          f"Missing session hashes: {qid}")
    if isinstance(hs, list):
        session_instances += len(hs)
        check(all(h in session_set for h in hs),
              f"Unknown session hash in mapping: {qid}")

gen_hashes = [x.get("hash") for x in generations]
gen_set = set(gen_hashes)
check(len(gen_set) == len(gen_hashes), "Duplicate generation hashes")
check(gen_set <= session_set, "Generation contains unexpected session hashes")

valid_generations = 0
for x in generations:
    summary = x.get("summary")
    keywords = x.get("keywords")
    valid = (
        isinstance(summary, str) and bool(summary.strip())
        and isinstance(keywords, str) and bool(keywords.strip())
    )
    check(valid, f"Empty/invalid summary or keywords: {x.get('hash')}")
    valid_generations += int(valid)

missing = session_set - gen_set
if missing or len(generations) != EXPECTED_SESSIONS:
    blockers.append(
        f"Generation incomplete: {len(gen_set)}/{EXPECTED_SESSIONS}; "
        f"missing={len(missing)}"
    )

check(MODEL_PATH.exists(), f"Missing local Contriever model: {MODEL_PATH}")
check(not CACHE_PATH.exists(),
      f"Cache already exists; refusing to overwrite: {CACHE_PATH}")

builder_text = BUILDER.read_text(encoding="utf-8")
for stale in (
    "adapters/memgas_personamem_1m",
    "shared_data/personamem_1m",
    "questions_1M.csv",
    "personamem_1m_contriever_cache.pt",
    "PersonaMem-1M",
    "memgas_personamem_1m",
):
    check(stale not in builder_text, f"Stale 1M reference: {stale}")

print("Unique sessions:", len(session_set))
print("Questions:", len(qid_set))
print("Mappings:", len(map_set))
print("Session instances across questions:", session_instances)
print("Generation records:", len(gen_set))
print("Valid summary+keywords:", valid_generations)
print("Missing generations:", len(missing))
print("Contriever model exists:", MODEL_PATH.exists())
print("Cache output already exists:", CACHE_PATH.exists())

if errors:
    print("\nPREFLIGHT_ERRORS:")
    for x in errors:
        print(" -", x)
    print("PREFLIGHT_FAIL")
    sys.exit(1)

if blockers:
    print("\nPREFLIGHT_BLOCKED:")
    for x in blockers:
        print(" -", x)
    print("PREFLIGHT_BLOCKED_GENERATION_INCOMPLETE")
    sys.exit(2)

print("PREFLIGHT_PASS")
