import csv
import json
from collections import Counter
from pathlib import Path

DATA = Path("/root/autodl-tmp/paper_repro/shared_data/personamem_32k")
QUESTIONS = DATA / "questions_32k.csv"
CONTEXTS = DATA / "shared_contexts_32k.jsonl"

OUT = Path(
    "/root/autodl-tmp/paper_repro/adapters/"
    "scm_personamem/personamem_scm_32k.jsonl"
)
AUDIT = OUT.with_name("adapter_audit.json")


# ------------------------------------------------------------
# Load shared contexts
# ------------------------------------------------------------

contexts = {}

with open(CONTEXTS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        x = json.loads(line)

        assert isinstance(x, dict)
        assert len(x) == 1

        sid, msgs = next(iter(x.items()))

        assert isinstance(sid, str)
        assert isinstance(msgs, list)

        contexts[sid] = msgs

def role_of(m):
    return str(m.get("role", "")).lower()


def content_of(m):
    return str(m.get("content", ""))


def make_unit(messages):
    """
    A transparent SCM-compatible memory unit.

    Original role/content is retained verbatim in raw_text.
    """
    raw_parts = []

    for m in messages:
        role = role_of(m)
        content = content_of(m)

        raw_parts.append(
            f"{role.capitalize()}: {content}"
        )

    return {
        "messages": messages,
        "raw_text": "\n\n".join(raw_parts),
    }


def to_scm_units(messages):
    """
    Minimal mapping:
      user immediately followed by assistant -> one SCM turn
      everything else -> standalone turn-like memory unit

    No content is dropped, merged across non-adjacent messages,
    reordered, or synthesized.
    """

    units = []
    i = 0

    while i < len(messages):
        cur = messages[i]

        if (
            role_of(cur) == "user"
            and i + 1 < len(messages)
            and role_of(messages[i + 1]) == "assistant"
        ):
            units.append(
                make_unit(
                    [cur, messages[i + 1]]
                )
            )
            i += 2

        else:
            units.append(make_unit([cur]))
            i += 1

    return units


rows = []
role_counts = Counter()
unit_size_counts = Counter()

future_leakage = 0
content_mismatch = 0


with open(
    QUESTIONS,
    encoding="utf-8-sig",
    newline="",
) as f:

    for q in csv.DictReader(f):

        sid = str(q["shared_context_id"])
        end_idx = int(
            q["end_index_in_shared_context"]
        )

        full = contexts[sid]

        # CRITICAL PersonaMem rule
        visible = full[:end_idx]

        # audit exact prefix
        if visible != full[:end_idx]:
            future_leakage += 1

        for m in visible:
            role_counts[role_of(m)] += 1

        units = to_scm_units(visible)

        # ensure exact original stream recoverable
        recovered = []

        for u in units:
            recovered.extend(u["messages"])
            unit_size_counts[
                len(u["messages"])
            ] += 1

        if recovered != visible:
            content_mismatch += 1

        rows.append({
            "question_id": q["question_id"],
            "persona_id": q["persona_id"],
            "question_type": q["question_type"],

            "question": q[
                "user_question_or_message"
            ],

            "correct_answer": q[
                "correct_answer"
            ],

            "all_options": q["all_options"],

            "shared_context_id": sid,
            "end_index": end_idx,

            "visible_message_count": len(
                visible
            ),

            "scm_memory_units": units,
        })


assert len(rows) == 589
assert future_leakage == 0
assert content_mismatch == 0


with open(
    OUT,
    "w",
    encoding="utf-8",
) as f:
    for x in rows:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            ) + "\n"
        )


unit_counts = [
    len(x["scm_memory_units"])
    for x in rows
]

audit = {
    "questions": len(rows),
    "future_leakage": future_leakage,
    "content_mismatch": content_mismatch,

    "roles": dict(role_counts),

    "unit_size_distribution": dict(
        unit_size_counts
    ),

    "memory_units_per_question": {
        "min": min(unit_counts),
        "max": max(unit_counts),
        "avg": sum(unit_counts)
               / len(unit_counts),
    },

    "mapping_rule": (
        "adjacent user->assistant messages are one "
        "SCM memory turn; all other original messages "
        "become standalone turn-like memory units"
    ),
}


with open(
    AUDIT,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        audit,
        f,
        ensure_ascii=False,
        indent=2,
    )


print("questions:", len(rows))
print(
    "future_leakage:",
    future_leakage
)
print(
    "content_mismatch:",
    content_mismatch
)
print(
    "role_counts:",
    dict(role_counts)
)
print(
    "unit_size_distribution:",
    dict(unit_size_counts)
)
print(
    "memory_units/question:",
    audit["memory_units_per_question"]
)

print("SCM_PERSONAMEM_ADAPTER_OK")
