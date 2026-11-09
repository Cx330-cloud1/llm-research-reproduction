import json
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import tiktoken


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_json(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_jsonl(rows, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def stats(vals):
    vals = sorted(vals)
    def pct(p):
        return vals[round((len(vals)-1)*p)]
    return {
        "min": vals[0],
        "median": pct(.5),
        "p95": pct(.95),
        "p99": pct(.99),
        "max": vals[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--expected-questions", type=int, required=True)
    ap.add_argument("--expected-unique", type=int, required=True)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--max-history-tokens", type=int, default=2500)
    ap.add_argument("--max-pre-turn-tokens", type=int, default=500)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base = Path(args.dataset_dir)
    OUT = base / "scm_top6_retrieval.jsonl"
    AUDIT = base / "scm_retrieval_controller_audit.json"

    if not args.overwrite and (OUT.exists() or AUDIT.exists()):
        raise RuntimeError("输出文件已存在，拒绝覆盖")

    questions = load(base / "question_metadata.jsonl")
    mappings = load(base / "question_scm_unit_hashes.jsonl")
    units = load(base / "unique_scm_memory_units.jsonl")
    summary_rows = load(base / "scm_summaries_qwen7b.jsonl")

    cache = torch.load(
        base / "scm_contriever_cache.pt",
        map_location="cpu",
        weights_only=False,
    )

    qids = [str(x["question_id"]) for x in questions]
    hashes = [x["unit_hash"] for x in units]

    assert len(qids) == args.expected_questions
    assert len(set(qids)) == len(qids)
    assert len(hashes) == args.expected_unique
    assert len(set(hashes)) == len(hashes)

    qmap = {str(x["question_id"]): x["unit_hashes"] for x in mappings}
    unique = {x["unit_hash"]: x for x in units}
    summaries = {x["unit_hash"]: x["summary"] for x in summary_rows}

    assert len(qmap) == len(qids)
    assert set(qmap) == set(qids)
    assert set(summaries) == set(hashes)
    assert cache["unit_hashes"] == hashes
    assert cache["question_ids"] == qids

    assert len(summary_rows) == args.expected_unique

    for row in mappings:
        assert all(h in unique for h in row["unit_hashes"])

    memory_emb = cache["memory_embeddings"].float()
    question_emb = cache["question_embeddings"].float()

    assert memory_emb.shape == (args.expected_unique, 768)
    assert question_emb.shape == (args.expected_questions, 768)
    assert torch.isfinite(memory_emb).all()
    assert torch.isfinite(question_emb).all()

    hash_idx = {h:i for i,h in enumerate(hashes)}
    qid_idx = {q:i for i,q in enumerate(qids)}

    memory_norm = torch.nn.functional.normalize(memory_emb, dim=1)
    question_norm = torch.nn.functional.normalize(question_emb, dim=1)

    try:
        enc = tiktoken.encoding_for_model("text-davinci-003")
    except Exception:
        enc = tiktoken.get_encoding("p50k_base")

    def toklen(text):
        return len(enc.encode(text))

    topk_outputs = []
    raw_budgets = []
    summary_budgets = []

    controller_needed = 0
    safe_after_summary = 0
    nontermination_risk = 0
    previous_raw = 0
    previous_summary = 0

    print("===== SCM RETRIEVAL START =====", flush=True)
    print("questions:", len(questions), flush=True)
    print("unique units:", len(units), flush=True)

    for qi, qid in enumerate(qids):
        unit_hashes = qmap[qid]

        assert len(unit_hashes) > args.topk

        candidates = unit_hashes[:-1]
        previous_hash = unit_hashes[-1]

        qv = question_norm[qid_idx[qid]]
        indices = [hash_idx[h] for h in candidates]

        sims = torch.mv(memory_norm[indices], qv).numpy()

        # Preserve original SCM NumPy Top-K behavior.
        top_indices = np.argsort(sims)[-args.topk:]

        selected = [candidates[int(i)] for i in top_indices]
        selected_sims = [float(sims[int(i)]) for i in top_indices]

        raw_lengths = [
            toklen(unique[h]["raw_text"]) for h in selected
        ]
        summary_lengths = [
            toklen(summaries[h]) for h in selected
        ]

        previous_raw_len = toklen(unique[previous_hash]["raw_text"])

        if previous_raw_len < args.max_pre_turn_tokens:
            pre_len = previous_raw_len
            pre_mode = "raw"
            previous_raw += 1
        else:
            pre_len = toklen(summaries[previous_hash])
            pre_mode = "summary"
            previous_summary += 1

        raw_budget = sum(raw_lengths) + pre_len
        summary_budget = sum(summary_lengths) + pre_len

        needs_controller = raw_budget > args.max_history_tokens

        if needs_controller:
            controller_needed += 1
            if summary_budget <= args.max_history_tokens:
                safe_after_summary += 1
            else:
                nontermination_risk += 1

        raw_budgets.append(raw_budget)
        summary_budgets.append(summary_budget)

        chronological = sorted(
            zip(top_indices.tolist(), selected, selected_sims),
            key=lambda x: x[0],
        )

        topk_outputs.append({
            "question_id": qid,
            "history_units": len(unit_hashes),
            "candidate_units": len(candidates),
            "controller_top_indices": top_indices.tolist(),
            "controller_top_hashes": selected,
            "controller_top_sims": selected_sims,
            "chronological_top_indices": [x[0] for x in chronological],
            "chronological_top_hashes": [x[1] for x in chronological],
            "raw_lengths": raw_lengths,
            "summary_lengths": summary_lengths,
            "previous_hash": previous_hash,
            "previous_mode": pre_mode,
            "previous_budget_tokens": pre_len,
            "raw_total_budget": raw_budget,
            "all_summary_total_budget": summary_budget,
            "needs_controller": needs_controller,
            "official_nontermination_risk": (
                needs_controller and summary_budget > args.max_history_tokens
            ),
        })

        if (qi + 1) % 100 == 0 or qi + 1 == len(qids):
            print(
                f"[RETRIEVAL] {qi+1}/{len(qids)}",
                flush=True,
            )

    audit = {
        "questions": len(questions),
        "unique_units": len(units),
        "topk": args.topk,
        "max_history_tokens": args.max_history_tokens,
        "max_previous_turn_tokens": args.max_pre_turn_tokens,
        "controller_needed": controller_needed,
        "controller_not_needed": len(questions)-controller_needed,
        "safe_if_all_selected_turns_summary": safe_after_summary,
        "official_nontermination_risk": nontermination_risk,
        "previous_turn_raw": previous_raw,
        "previous_turn_summary": previous_summary,
        "raw_budget_stats": stats(raw_budgets),
        "all_summary_budget_stats": stats(summary_budgets),
        "similarity": "cosine",
        "retrieval_candidates": "all memory units except last",
    }

    assert len(topk_outputs) == args.expected_questions
    assert all(
        len(x["chronological_top_hashes"]) == args.topk
        for x in topk_outputs
    )

    write_jsonl(topk_outputs, OUT)
    write_json(audit, AUDIT)

    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    print("SCM_RETRIEVAL_MULTI_OK", flush=True)


if __name__ == "__main__":
    main()
