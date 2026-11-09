#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an isolated SCM adapter from a PersonaMem CSV + shared-context JSONL."""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


REQUIRED_COLUMNS = {
    "question_id",
    "persona_id",
    "question_type",
    "user_question_or_message",
    "correct_answer",
    "all_options",
    "shared_context_id",
    "end_index_in_shared_context",
}


def role_of(message):
    return str(message.get("role", "")).lower()


def make_unit(messages):
    raw_parts = [
        f"{role_of(message).capitalize()}: {str(message.get('content', ''))}"
        for message in messages
    ]
    return {
        "messages": messages,
        "raw_text": "\n\n".join(raw_parts),
    }


def to_scm_units(messages):
    # Preserve the original adapter rule: only adjacent user->assistant
    # messages are paired. All other messages remain standalone units.
    units = []
    i = 0
    while i < len(messages):
        current = messages[i]
        if (
            role_of(current) == "user"
            and i + 1 < len(messages)
            and role_of(messages[i + 1]) == "assistant"
        ):
            units.append(make_unit([current, messages[i + 1]]))
            i += 2
        else:
            units.append(make_unit([current]))
            i += 1
    return units


def load_contexts(path):
    contexts = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or len(row) != 1:
                raise ValueError(f"{path}:{line_no}: each row must be a one-key object")
            context_id, messages = next(iter(row.items()))
            context_id = str(context_id)
            if context_id in contexts:
                raise ValueError(f"Duplicate shared_context_id: {context_id}")
            if not isinstance(messages, list):
                raise TypeError(f"Context {context_id}: messages must be a list")
            if not all(isinstance(m, dict) for m in messages):
                raise TypeError(f"Context {context_id}: every message must be an object")
            contexts[context_id] = messages
    return contexts


def atomic_write_jsonl(rows, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--expected-questions", type=int, required=True)
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    questions_path = Path(args.questions).resolve()
    contexts_path = Path(args.contexts).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_path = out_dir / args.data_file
    audit_path = out_dir / "adapter_audit.json"

    if not questions_path.is_file():
        raise FileNotFoundError(questions_path)
    if not contexts_path.is_file():
        raise FileNotFoundError(contexts_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite and (out_path.exists() or audit_path.exists()):
        raise FileExistsError(
            f"Refusing to overwrite existing output: {out_path} or {audit_path}"
        )

    print("===== LOAD CONTEXTS =====", flush=True)
    contexts = load_contexts(contexts_path)
    print("context records:", len(contexts), flush=True)

    rows = []
    question_ids = set()
    referenced_contexts = set()
    role_counts = Counter()
    unit_size_counts = Counter()
    negative_end_indices = 0
    visible_message_counts = []
    unit_counts = []
    total_units = 0

    print("===== BUILD QUESTION-SCOPED SCM UNITS =====", flush=True)
    with questions_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - columns
        if missing_columns:
            raise KeyError(f"CSV missing columns: {sorted(missing_columns)}")

        for row_no, q in enumerate(reader, 2):
            qid = str(q["question_id"])
            if not qid or qid in question_ids:
                raise ValueError(f"Empty or duplicate question_id at CSV row {row_no}: {qid!r}")
            question_ids.add(qid)

            context_id = str(q["shared_context_id"])
            if context_id not in contexts:
                raise KeyError(f"Question {qid} references missing context {context_id}")
            referenced_contexts.add(context_id)

            full = contexts[context_id]
            try:
                end_index = int(q["end_index_in_shared_context"])
            except Exception as exc:
                raise ValueError(f"Question {qid}: invalid end_index") from exc

            # Negative indices are valid Python slice boundaries when >= -len(full).
            if end_index < -len(full) or end_index > len(full):
                raise ValueError(
                    f"Question {qid}: end_index={end_index}, context_len={len(full)}"
                )
            if end_index < 0:
                negative_end_indices += 1

            visible = full[:end_index]
            if not visible:
                raise ValueError(f"Question {qid}: visible context is empty")

            for message in visible:
                role_counts[role_of(message)] += 1

            units = to_scm_units(visible)
            recovered = []
            for unit in units:
                recovered.extend(unit["messages"])
                unit_size_counts[len(unit["messages"])] += 1

            if recovered != visible:
                raise AssertionError(f"Lossless SCM-unit reconstruction failed: {qid}")
            if len(recovered) != len(visible):
                raise AssertionError(f"Message count mismatch: {qid}")

            rows.append({
                "question_id": qid,
                "persona_id": q["persona_id"],
                "question_type": q["question_type"],
                "question": q["user_question_or_message"],
                "correct_answer": q["correct_answer"],
                "all_options": q["all_options"],
                "shared_context_id": context_id,
                "end_index": end_index,
                "visible_message_count": len(visible),
                "scm_memory_units": units,
            })

            visible_message_counts.append(len(visible))
            unit_counts.append(len(units))
            total_units += len(units)

            if len(rows) % 100 == 0:
                print(f"questions built: {len(rows)}/{args.expected_questions}", flush=True)

    if len(rows) != args.expected_questions:
        raise AssertionError(
            f"Question count mismatch: {len(rows)} != {args.expected_questions}"
        )

    print("===== WRITE ADAPTER =====", flush=True)
    atomic_write_jsonl(rows, out_path)

    audit = {
        "dataset": args.dataset_name,
        "questions": len(rows),
        "context_records": len(contexts),
        "referenced_contexts": len(referenced_contexts),
        "unused_contexts": len(set(contexts) - referenced_contexts),
        "negative_end_index_count": negative_end_indices,
        "future_leakage_count": 0,
        "lossless_reconstruction_failures": 0,
        "visible_message_min": min(visible_message_counts),
        "visible_message_max": max(visible_message_counts),
        "scm_memory_unit_instances": total_units,
        "memory_units_per_question": {
            "min": min(unit_counts),
            "max": max(unit_counts),
            "avg": total_units / len(rows),
        },
        "unit_size_distribution": dict(sorted(unit_size_counts.items())),
        "role_counts_in_visible_prefixes": dict(sorted(role_counts.items())),
        "mapping_rule": (
            "Only adjacent user->assistant messages form one SCM unit; "
            "all other messages become standalone units. "
            "The visible history is exactly full_context[:end_index]."
        ),
        "source_questions": str(questions_path),
        "source_contexts": str(contexts_path),
        "output": str(out_path),
    }

    tmp = audit_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, audit_path)

    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    print("ADAPTER_BUILD_OK", flush=True)
    print("data:", out_path, flush=True)
    print("audit:", audit_path, flush=True)


if __name__ == "__main__":
    main()
