#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit deduplicated SCM adapters for PersonaMem-128K and PersonaMem-1M."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import tiktoken

ROOT = Path("/root/autodl-tmp/paper_repro")
DATASETS = {
    "PersonaMem-128K": ROOT / "adapters/scm_personamem_128k_dedup",
    "PersonaMem-1M": ROOT / "adapters/scm_personamem_1m",
}
PROMPT_PREFIX = (
    "Below is a conversation between a user and an AI assistant. "
    "Please write a summary for each of them in one sentence and "
    "list them in separate paragraphs, while trying to preserve "
    "the key information of the user’s question and the assistant’s "
    "answer as much as possible.\n\n"
    "conversation content: \n\n"
)
PROMPT_SUFFIX = "\n\nSummary:\n"
SUMMARY_THRESHOLD = 300


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as exc:
                    raise RuntimeError(
                        f"Invalid JSON at {path}:{line_no}"
                    ) from exc


def main():
    try:
        enc = tiktoken.encoding_for_model("text-davinci-003")
        encoding_name = "text-davinci-003"
    except Exception:
        enc = tiktoken.get_encoding("p50k_base")
        encoding_name = "p50k_base"

    all_ok = True

    for dataset, base in DATASETS.items():
        print(f"\n===== {dataset} =====", flush=True)

        unique_path = base / "unique_scm_memory_units.jsonl"
        mapping_path = base / "question_scm_unit_hashes.jsonl"
        metadata_path = base / "question_metadata.jsonl"
        adapter_audit_path = base / "adapter_audit.json"
        output_path = base / "scm_memory_unit_audit.json"

        for path in (
            unique_path, mapping_path, metadata_path, adapter_audit_path
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        with adapter_audit_path.open(encoding="utf-8") as f:
            adapter_audit = json.load(f)

        unit_hashes = set()
        needs_summary = {}
        token_lengths = []
        duplicate_units = 0
        hash_mismatches = 0
        raw_reconstruction_mismatches = 0

        for row in read_jsonl(unique_path):
            h = row["unit_hash"]
            raw = row["raw_text"]

            if h in unit_hashes:
                duplicate_units += 1
                continue
            unit_hashes.add(h)

            actual_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            if actual_hash != h:
                hash_mismatches += 1

            messages = row["messages"]
            raw_parts = [
                f"{str(m.get('role', '')).lower().capitalize()}: "
                f"{str(m.get('content', ''))}"
                for m in messages
            ]
            reconstructed_raw = "\n\n".join(raw_parts)
            if reconstructed_raw != raw:
                raw_reconstruction_mismatches += 1

            prompt_tokens = len(
                enc.encode(PROMPT_PREFIX + raw + PROMPT_SUFFIX)
            )
            token_lengths.append(prompt_tokens)
            needs_summary[h] = prompt_tokens > SUMMARY_THRESHOLD

        metadata_ids = set()
        duplicate_metadata_ids = 0
        metadata_count = 0
        for row in read_jsonl(metadata_path):
            qid = str(row["question_id"])
            metadata_count += 1
            if qid in metadata_ids:
                duplicate_metadata_ids += 1
            metadata_ids.add(qid)

        mapping_ids = set()
        duplicate_mapping_ids = 0
        mapping_count = 0
        total_instances = 0
        summary_needed_instances = 0
        missing_hash_references = 0
        max_units_per_question = 0
        min_units_per_question = None

        for row in read_jsonl(mapping_path):
            qid = str(row["question_id"])
            mapping_count += 1
            if qid in mapping_ids:
                duplicate_mapping_ids += 1
            mapping_ids.add(qid)

            hashes = row["unit_hashes"]
            if not isinstance(hashes, list) or not hashes:
                raise ValueError(f"Empty/invalid unit_hashes for {qid}")

            n = len(hashes)
            total_instances += n
            max_units_per_question = max(max_units_per_question, n)
            min_units_per_question = (
                n if min_units_per_question is None
                else min(min_units_per_question, n)
            )

            for h in hashes:
                if h not in unit_hashes:
                    missing_hash_references += 1
                elif needs_summary[h]:
                    summary_needed_instances += 1

        missing_metadata_ids = sorted(mapping_ids - metadata_ids)[:5]
        missing_mapping_ids = sorted(metadata_ids - mapping_ids)[:5]

        checks = {
            "question_count_matches_adapter": (
                mapping_count == adapter_audit["questions"]
                and metadata_count == adapter_audit["questions"]
            ),
            "mapping_metadata_ids_match": (
                mapping_ids == metadata_ids
                and duplicate_mapping_ids == 0
                and duplicate_metadata_ids == 0
            ),
            "unique_count_matches_adapter": (
                len(unit_hashes) == adapter_audit["unique_scm_memory_units"]
            ),
            "unit_instances_match_adapter": (
                total_instances == adapter_audit["scm_memory_unit_instances"]
            ),
            "no_duplicate_unique_units": duplicate_units == 0,
            "all_unit_hashes_valid": hash_mismatches == 0,
            "all_raw_text_reconstructed": raw_reconstruction_mismatches == 0,
            "all_mapping_hashes_resolve": missing_hash_references == 0,
        }

        sorted_tokens = sorted(token_lengths)

        def percentile(p):
            if not sorted_tokens:
                return None
            idx = int(round((len(sorted_tokens) - 1) * p))
            return sorted_tokens[idx]

        summary_needed_unique = sum(needs_summary.values())
        audit = {
            "dataset": dataset,
            "questions": mapping_count,
            "unique_units": len(unit_hashes),
            "unit_instances": total_instances,
            "deduplication_ratio": (
                1.0 - len(unit_hashes) / total_instances
                if total_instances else 0.0
            ),
            "duplicate_unique_rows": duplicate_units,
            "hash_mismatches": hash_mismatches,
            "raw_text_reconstruction_mismatches": raw_reconstruction_mismatches,
            "missing_hash_references": missing_hash_references,
            "duplicate_mapping_question_ids": duplicate_mapping_ids,
            "duplicate_metadata_question_ids": duplicate_metadata_ids,
            "missing_metadata_ids_sample": missing_metadata_ids,
            "missing_mapping_ids_sample": missing_mapping_ids,
            "summary_tokenizer": encoding_name,
            "summary_threshold_tokens": SUMMARY_THRESHOLD,
            "summary_needed_unique": summary_needed_unique,
            "summary_short_path_unique": len(unit_hashes) - summary_needed_unique,
            "summary_needed_instances": summary_needed_instances,
            "prompt_token_stats": {
                "min": min(sorted_tokens) if sorted_tokens else None,
                "median": percentile(0.50),
                "p95": percentile(0.95),
                "p99": percentile(0.99),
                "max": max(sorted_tokens) if sorted_tokens else None,
            },
            "units_per_question": {
                "min": min_units_per_question,
                "max": max_units_per_question,
            },
            "checks": checks,
            "all_checks_pass": all(checks.values()),
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(audit, f, ensure_ascii=False, indent=2)

        for key, value in checks.items():
            print(f"{'PASS' if value else 'FAIL'}: {key}", flush=True)

        print(
            f"questions={mapping_count} unique_units={len(unit_hashes)} "
            f"instances={total_instances}",
            flush=True,
        )
        print(
            f"summary_needed_unique={summary_needed_unique} "
            f"short_path_unique={len(unit_hashes) - summary_needed_unique}",
            flush=True,
        )
        print("audit:", output_path, flush=True)

        all_ok = all_ok and all(checks.values())

    if not all_ok:
        raise SystemExit("SCM_MEMORY_UNIT_AUDIT_FAILED")

    print("\nSCM_BOTH_DATASETS_AUDIT_PASS", flush=True)


if __name__ == "__main__":
    main()
