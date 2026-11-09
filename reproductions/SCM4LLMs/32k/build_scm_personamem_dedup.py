#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Low-disk SCM adapter builder for PersonaMem.

Writes:
  unique_scm_memory_units.jsonl  one record per unique SCM unit
  question_scm_unit_hashes.jsonl ordered unit hashes per question
  question_metadata.jsonl        question metadata without duplicated units
  adapter_audit.json             boundary and reconstruction audit

The unit mapping rule is unchanged:
  adjacent user -> assistant messages form one unit;
  every other message is a standalone unit.

The visible context is exactly full_context[:end_index].
"""

import argparse
import csv
import hashlib
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
    raw_text = "\n\n".join(raw_parts)
    return {
        "messages": messages,
        "raw_text": raw_text,
    }


def to_scm_units(messages):
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
                raise ValueError(
                    f"{path}:{line_no}: each row must be a one-key object"
                )
            context_id, messages = next(iter(row.items()))
            context_id = str(context_id)
            if context_id in contexts:
                raise ValueError(f"Duplicate shared_context_id: {context_id}")
            if not isinstance(messages, list):
                raise TypeError(f"Context {context_id}: messages must be a list")
            if not all(isinstance(m, dict) for m in messages):
                raise TypeError(
                    f"Context {context_id}: every message must be an object"
                )
            contexts[context_id] = messages
    return contexts


def write_jsonl_line(handle, obj):
    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def fsync_close(handle):
    handle.flush()
    os.fsync(handle.fileno())
    handle.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--expected-questions", type=int, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    questions_path = Path(args.questions).resolve()
    contexts_path = Path(args.contexts).resolve()
    out_dir = Path(args.out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    unique_path = out_dir / "unique_scm_memory_units.jsonl"
    mapping_path = out_dir / "question_scm_unit_hashes.jsonl"
    metadata_path = out_dir / "question_metadata.jsonl"
    audit_path = out_dir / "adapter_audit.json"

    final_paths = [unique_path, mapping_path, metadata_path, audit_path]
    temp_paths = [Path(str(p) + ".tmp") for p in final_paths]

    if not args.overwrite:
        existing = [str(p) for p in final_paths + temp_paths if p.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing output(s): " + ", ".join(existing)
            )
    else:
        for p in final_paths + temp_paths:
            if p.exists():
                p.unlink()

    if not questions_path.is_file():
        raise FileNotFoundError(questions_path)
    if not contexts_path.is_file():
        raise FileNotFoundError(contexts_path)

    print("===== LOAD CONTEXTS =====", flush=True)
    contexts = load_contexts(contexts_path)
    print("context records:", len(contexts), flush=True)

    unique_hashes = set()
    question_ids = set()
    referenced_contexts = set()
    role_counts = Counter()
    unit_size_counts = Counter()
    negative_end_indices = 0
    visible_message_counts = []
    unit_counts = []
    total_units = 0
    unique_units = 0
    future_leakage_count = 0
    reconstruction_failures = 0

    print("===== BUILD DEDUPLICATED SCM ADAPTER =====", flush=True)

    with (
        temp_paths[0].open("w", encoding="utf-8") as unique_f,
        temp_paths[1].open("w", encoding="utf-8") as mapping_f,
        temp_paths[2].open("w", encoding="utf-8") as metadata_f,
        questions_path.open(encoding="utf-8-sig", newline="") as source_f,
    ):
        reader = csv.DictReader(source_f)
        columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - columns
        if missing_columns:
            raise KeyError(f"CSV missing columns: {sorted(missing_columns)}")

        for row_no, q in enumerate(reader, 2):
            qid = str(q["question_id"])
            if not qid or qid in question_ids:
                raise ValueError(
                    f"Empty or duplicate question_id at CSV row {row_no}: {qid!r}"
                )
            question_ids.add(qid)

            context_id = str(q["shared_context_id"])
            if context_id not in contexts:
                raise KeyError(
                    f"Question {qid} references missing context {context_id}"
                )
            referenced_contexts.add(context_id)

            full = contexts[context_id]
            try:
                end_index = int(q["end_index_in_shared_context"])
            except Exception as exc:
                raise ValueError(
                    f"Question {qid}: invalid end_index"
                ) from exc

            # Preserve Python slicing semantics, including valid negative indices.
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
            hashes = []

            for unit in units:
                recovered.extend(unit["messages"])
                unit_size_counts[len(unit["messages"])] += 1
                raw_text = unit["raw_text"]
                unit_hash = hashlib.sha256(
                    raw_text.encode("utf-8")
                ).hexdigest()
                hashes.append(unit_hash)

                if unit_hash not in unique_hashes:
                    unique_hashes.add(unit_hash)
                    unique_units += 1
                    write_jsonl_line(
                        unique_f,
                        {
                            "unit_hash": unit_hash,
                            "raw_text": raw_text,
                            "messages": unit["messages"],
                        },
                    )

            if recovered != visible:
                reconstruction_failures += 1
                raise AssertionError(
                    f"Lossless SCM-unit reconstruction failed: {qid}"
                )
            if len(recovered) != len(visible):
                reconstruction_failures += 1
                raise AssertionError(f"Message count mismatch: {qid}")

            write_jsonl_line(
                mapping_f,
                {
                    "question_id": qid,
                    "unit_hashes": hashes,
                },
            )
            write_jsonl_line(
                metadata_f,
                {
                    "question_id": qid,
                    "persona_id": q["persona_id"],
                    "question_type": q["question_type"],
                    "question": q["user_question_or_message"],
                    "correct_answer": q["correct_answer"],
                    "all_options": q["all_options"],
                    "shared_context_id": context_id,
                    "end_index": end_index,
                    "visible_message_count": len(visible),
                    "scm_memory_unit_count": len(units),
                },
            )

            visible_message_counts.append(len(visible))
            unit_counts.append(len(units))
            total_units += len(units)

            if len(question_ids) % 100 == 0:
                print(
                    f"questions built: {len(question_ids)}/{args.expected_questions} "
                    f"unique_units={unique_units}",
                    flush=True,
                )

        if len(question_ids) != args.expected_questions:
            raise AssertionError(
                f"Question count mismatch: {len(question_ids)} "
                f"!= {args.expected_questions}"
            )

        fsync_close(unique_f)
        fsync_close(mapping_f)
        fsync_close(metadata_f)

    audit = {
        "dataset": args.dataset_name,
        "questions": len(question_ids),
        "context_records": len(contexts),
        "referenced_contexts": len(referenced_contexts),
        "unused_contexts": len(set(contexts) - referenced_contexts),
        "negative_end_index_count": negative_end_indices,
        "future_leakage_count": future_leakage_count,
        "lossless_reconstruction_failures": reconstruction_failures,
        "visible_message_min": min(visible_message_counts),
        "visible_message_max": max(visible_message_counts),
        "scm_memory_unit_instances": total_units,
        "unique_scm_memory_units": unique_units,
        "deduplication_ratio": (
            1.0 - unique_units / total_units if total_units else 0.0
        ),
        "memory_units_per_question": {
            "min": min(unit_counts),
            "max": max(unit_counts),
            "avg": total_units / len(unit_counts),
        },
        "unit_size_distribution": dict(sorted(unit_size_counts.items())),
        "role_counts_in_visible_prefixes": dict(sorted(role_counts.items())),
        "mapping_rule": (
            "Only adjacent user->assistant messages form one SCM unit; "
            "all other messages become standalone units. "
            "The visible history is exactly full_context[:end_index]. "
            "Unique units are stored once and referenced by ordered SHA256 hashes."
        ),
        "source_questions": str(questions_path),
        "source_contexts": str(contexts_path),
        "unique_units_output": str(unique_path),
        "question_hashes_output": str(mapping_path),
        "question_metadata_output": str(metadata_path),
    }

    with temp_paths[3].open("w", encoding="utf-8") as audit_f:
        json.dump(audit, audit_f, ensure_ascii=False, indent=2)
        audit_f.flush()
        os.fsync(audit_f.fileno())

    # Publish outputs only after all four temporary files are complete.
    for tmp, final in zip(temp_paths, final_paths):
        os.replace(tmp, final)

    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    print("SCM_DEDUP_ADAPTER_BUILD_OK", flush=True)
    for p in final_paths:
        print("output:", p, flush=True)


if __name__ == "__main__":
    main()
