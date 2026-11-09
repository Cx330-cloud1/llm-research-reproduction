import csv
import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/paper_repro")

QUESTIONS = ROOT / "shared_data/personamem_32k/questions_32k.csv"
CONTEXTS = ROOT / "shared_data/personamem_32k/shared_contexts_32k.jsonl"

OUT = ROOT / "adapters/memgas_personamem/personamem_memgas.json"
AUDIT = ROOT / "adapters/memgas_personamem/adapter_audit.json"


def load_contexts():
    contexts = {}

    with CONTEXTS.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assert len(row) == 1

            context_id, messages = next(iter(row.items()))

            assert context_id not in contexts
            assert isinstance(messages, list)

            contexts[context_id] = messages

    return contexts


def split_sessions(visible_messages):
    """
    PersonaMem -> MemGAS minimal session conversion.

    Rule:
      - every `system` message begins a new session
      - system content is retained
      - all message contents are retained verbatim
      - no merging / summarization / synthetic turns
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
                # Defensive fallback. Audited PersonaMem contexts normally
                # begin with system, but do not silently drop messages.
                current = []
            current.append(content)

    if current:
        sessions.append(current)

    return sessions


contexts = load_contexts()
output = []

audit = {
    "num_questions": 0,
    "num_contexts": len(contexts),
    "future_leakage_count": 0,
    "bad_end_index_count": 0,
    "empty_session_count": 0,
    "session_count_min": None,
    "session_count_max": None,
    "visible_message_min": None,
    "visible_message_max": None,
}


with QUESTIONS.open(encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        question_id = row["question_id"]
        shared_context_id = row["shared_context_id"]

        if shared_context_id not in contexts:
            raise KeyError(
                f"missing shared_context_id={shared_context_id} "
                f"for question_id={question_id}"
            )

        full_context = contexts[shared_context_id]
        end_index = int(row["end_index_in_shared_context"])

        if not (0 <= end_index <= len(full_context)):
            audit["bad_end_index_count"] += 1
            raise ValueError(
                f"bad end_index={end_index}, "
                f"context_len={len(full_context)}, "
                f"question_id={question_id}"
            )

        # CRITICAL PERSONAMEM RULE:
        # only messages strictly before end_index are visible.
        visible_context = full_context[:end_index]

        sessions = split_sessions(visible_context)

        if any(len(s) == 0 for s in sessions):
            audit["empty_session_count"] += 1
            raise AssertionError(f"empty session: {question_id}")

        session_ids = [
            f"session_{i+1}"
            for i in range(len(sessions))
        ]

        # PersonaMem has no session timestamps.
        # Keep deterministic structural placeholders only.
        session_dates = [
            f"session_{i+1}"
            for i in range(len(sessions))
        ]

        # This is deliberately NOT LoCoMo retrieval evaluation format:
        # there is no fabricated answer_session_ids.
        qa = [{
            "question": row["user_question_or_message"],
            "answer": row["correct_answer"],
            "question_type": row["question_type"],
            "question_date": "",
            "question_id": question_id,
            "all_options": row["all_options"],
        }]

        item = {
            "conversation_id": f"personamem-{question_id}",
            "sessions": sessions,
            "sessions_ids": session_ids,
            "sessions_dates": session_dates,
            "qa": qa,

            # Adapter-only provenance for leakage auditing.
            "adapter_meta": {
                "question_id": question_id,
                "persona_id": row["persona_id"],
                "shared_context_id": shared_context_id,
                "end_index": end_index,
                "visible_message_count": len(visible_context),
                "full_context_message_count": len(full_context),
                "question_type": row["question_type"],
            },
        }

        # Strong leakage invariant:
        # nothing at or after end_index is inserted into sessions.
        reconstructed_count = sum(len(s) for s in sessions)

        if reconstructed_count != len(visible_context):
            audit["future_leakage_count"] += 1
            raise AssertionError(
                f"message preservation mismatch: {question_id}; "
                f"visible={len(visible_context)}, "
                f"reconstructed={reconstructed_count}"
            )

        output.append(item)

        sc = len(sessions)
        vm = len(visible_context)

        audit["session_count_min"] = (
            sc if audit["session_count_min"] is None
            else min(audit["session_count_min"], sc)
        )
        audit["session_count_max"] = (
            sc if audit["session_count_max"] is None
            else max(audit["session_count_max"], sc)
        )

        audit["visible_message_min"] = (
            vm if audit["visible_message_min"] is None
            else min(audit["visible_message_min"], vm)
        )
        audit["visible_message_max"] = (
            vm if audit["visible_message_max"] is None
            else max(audit["visible_message_max"], vm)
        )


audit["num_questions"] = len(output)

OUT.write_text(
    json.dumps(output, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

AUDIT.write_text(
    json.dumps(audit, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("ADAPTER_BUILD_DONE")
print("output:", OUT)
print("questions:", len(output))
print("contexts:", len(contexts))
print("future_leakage_count:", audit["future_leakage_count"])
print(
    "session_count_range:",
    audit["session_count_min"],
    audit["session_count_max"],
)
print(
    "visible_message_range:",
    audit["visible_message_min"],
    audit["visible_message_max"],
)
