import os
import json
import shutil
import argparse
import tempfile
import traceback
from pathlib import Path
from collections import OrderedDict

import pandas as pd

# ---------------------------------------------------------------------
# PersonaMem adaptation settings
# ---------------------------------------------------------------------
os.environ["HEBBIAN_LOCAL_QWEN"] = "1"
os.environ["HEBBIAN_EMBED_MODEL"] = "/root/autodl-tmp/models/all-MiniLM-L6-v2"
os.environ["HEBBIAN_USE_TIME_DECAY"] = "false"
os.environ.setdefault("HEBBIAN_USE_KEYWORD_MATCH", "true")
os.environ.setdefault("HEBBIAN_LEARNING_RATE", "0.02")
os.environ.setdefault("HEBBIAN_DECAY_RATE", "0.995")
os.environ.setdefault("HEBBIAN_ACTIVATION_ALPHA", "0.1")
os.environ.setdefault("HEBBIAN_SPREADING_THRESHOLD", "0.4")
os.environ.setdefault("HEBBIAN_MAX_FLIPPED", "3")
os.environ.setdefault("HEBBIAN_KEYWORD_WEIGHT", "0.7")
os.environ.setdefault("HEBBIAN_KNOWLEDGE_BUFFER_SIZE", "10")

from eval_personamem_local import (
    QUESTION_PATH,
    CONTEXT_PATH,
    load_rows_with_context,
    context_to_turns,
    format_retrieval,
    INSTRUCTION,
    generate_answers,
    extract_answer,
    report,
)
from eval_personamem_local_full import (
    LocalQwenClient,
    add_semantic_context,
)
from hela_mem.hebbian_memory import HebbianMemoryGraph
from hela_mem.hebbian_retriever import HebbianRetriever
from hela_mem.hebbian_knowledge_memory import HebbianKnowledgeMemory
from hela_mem.encode_longmemeval import process_incremental_buffer, BUFFER_SIZE
from hela_mem.local_qwen_bridge import get_llm

TIMESTAMP = "unknown"
EXPECTED_FULL_QUESTIONS = 2674
EXPECTED_FULL_CUTOFFS = 276


def cutoff_key(shared_context_id, end_idx):
    return f"{shared_context_id}__end_{int(end_idx)}"


def cache_dir_for(cache_root, shared_context_id, end_idx):
    return Path(cache_root) / str(shared_context_id) / f"end_{int(end_idx)}"


def cache_is_ready(cache_dir):
    cache_dir = Path(cache_dir)
    required = [
        cache_dir / "READY",
        cache_dir / "memory_hebbian.json",
        cache_dir / "memory_long_term.json",
        cache_dir / "metadata.json",
    ]
    return all(p.exists() for p in required)


def load_dataset_records(num_items=None):
    records = []
    for row, context in load_rows_with_context(str(QUESTION_PATH), str(CONTEXT_PATH)):
        end_idx = int(row["end_index_in_shared_context"])
        if end_idx >= 0 and len(context) != end_idx:
            raise AssertionError(
                f"no-future-leak violation for {row['question_id']}: "
                f"len(context)={len(context)} end_idx={end_idx}"
            )
        records.append((row, context))
        if num_items is not None and len(records) >= num_items:
            break
    return records


def unique_cutoffs(records):
    out = OrderedDict()
    for row, context in records:
        key = cutoff_key(row["shared_context_id"], row["end_index_in_shared_context"])
        if key not in out:
            out[key] = (row, context)
    return out


def force_save(graph, kb):
    graph.save()
    if hasattr(kb, "save"):
        kb.save(force=True)
        return
    kg = getattr(kb, "knowledge_graph", None)
    if kg is not None and hasattr(kg, "save"):
        kg.save()


def state_counts(kb):
    kg = getattr(kb, "knowledge_graph", None)
    knowledge_nodes = len(getattr(kg, "nodes", []) or []) if kg is not None else 0
    try:
        assistant_count = len(kb.get_assistant_knowledge() or [])
    except Exception:
        assistant_count = 0
    return knowledge_nodes, assistant_count


def build_cutoff_cache(row, context, cache_root, client, force=False):
    shared_id = row["shared_context_id"]
    end_idx = int(row["end_index_in_shared_context"])
    qid = row["question_id"]
    final_dir = cache_dir_for(cache_root, shared_id, end_idx)

    if cache_is_ready(final_dir) and not force:
        print(f"[CACHE HIT] {cutoff_key(shared_id, end_idx)}")
        return final_dir

    if final_dir.exists():
        print(f"[CACHE INCOMPLETE] rebuilding {final_dir}")
        shutil.rmtree(final_dir, ignore_errors=True)

    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.tmp_", dir=str(parent)))

    try:
        mem_path = tmp_dir / "memory_hebbian.json"
        kb_path = tmp_dir / "memory_long_term.json"

        graph = HebbianMemoryGraph(file_path=str(mem_path))
        kb = HebbianKnowledgeMemory(file_path=str(kb_path))
        retriever = HebbianRetriever(graph, profile_memory=kb)

        turns = context_to_turns(context)
        incremental_buffer = []

        print(
            f"[ENCODE] key={cutoff_key(shared_id, end_idx)} "
            f"qid={qid} messages={len(context)} turns={len(turns)}"
        )

        for turn_idx, (user_text, assistant_text) in enumerate(turns, start=1):
            retriever.process_conversation_turn(
                user_text,
                assistant_text,
                timestamp=TIMESTAMP,
            )

            incremental_buffer.append(
                {
                    "user_text": user_text,
                    "ai_text": assistant_text,
                    "timestamp": TIMESTAMP,
                }
            )

            if len(incremental_buffer) >= BUFFER_SIZE:
                process_incremental_buffer(
                    incremental_buffer,
                    kb,
                    shared_id,
                    client,
                )
                incremental_buffer = []
                print(f"  encoded through turn {turn_idx}")

        if incremental_buffer:
            process_incremental_buffer(
                incremental_buffer,
                kb,
                shared_id,
                client,
            )

        force_save(graph, kb)

        knowledge_nodes, assistant_count = state_counts(kb)
        profile = kb.get_raw_user_profile(shared_id) or ""

        metadata = {
            "cache_version": 1,
            "shared_context_id": shared_id,
            "end_index_in_shared_context": end_idx,
            "source_question_id": qid,
            "messages": len(context),
            "turns": len(turns),
            "episodic_nodes": len(graph.nodes),
            "knowledge_nodes": knowledge_nodes,
            "assistant_knowledge_count": assistant_count,
            "profile_present": bool(profile.strip()),
            "time_decay": False,
            "timestamp_policy": TIMESTAMP,
            "buffer_size": BUFFER_SIZE,
            "llm_backend": "Qwen3-8B-Base",
        }

        with open(tmp_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # READY must be created last in temp dir.
        (tmp_dir / "READY").write_text("ok\n", encoding="utf-8")

        # Rename temp dir atomically within the same filesystem.
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        os.replace(str(tmp_dir), str(final_dir))

        print(
            f"[CACHE READY] {cutoff_key(shared_id, end_idx)} "
            f"episodic={metadata['episodic_nodes']} "
            f"knowledge={knowledge_nodes} profile={metadata['profile_present']}"
        )
        return final_dir

    except Exception:
        print(f"[CACHE BUILD ERROR] {cutoff_key(shared_id, end_idx)}")
        traceback.print_exc()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def copy_cache_for_question(cache_dir, work_root, question_id):
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    dst = work_root / question_id
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(cache_dir, dst)
    return dst


def load_cached_state(work_dir):
    work_dir = Path(work_dir)
    graph = HebbianMemoryGraph(file_path=str(work_dir / "memory_hebbian.json"))
    kb = HebbianKnowledgeMemory(file_path=str(work_dir / "memory_long_term.json"))
    retriever = HebbianRetriever(graph, profile_memory=kb)
    return graph, kb, retriever


def result_path_for(results_root, question_id):
    return Path(results_root) / "items" / f"{question_id}.json"


def valid_existing_result(path, question_id):
    path = Path(path)
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj.get("question_id") == question_id and "score" in obj and "error" in obj
    except Exception:
        return False


def evaluate_question(row, cache_dir, results_root, llm, top_k, semantic_top_k):
    qid = row["question_id"]
    shared_id = row["shared_context_id"]
    end_idx = int(row["end_index_in_shared_context"])
    out_path = result_path_for(results_root, qid)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if valid_existing_result(out_path, qid):
        print(f"[QA HIT] {qid}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    question = row["user_question_or_message"]
    correct_answer = row["correct_answer"]
    all_options = row["all_options"]

    base_result = {
        "score": False,
        "persona_id": row.get("persona_id"),
        "question_id": qid,
        "shared_context_id": shared_id,
        "end_index_in_shared_context": end_idx,
        "cache_key": cutoff_key(shared_id, end_idx),
        "question": question,
        "question_type": row.get("question_type"),
        "topic": row.get("topic"),
        "correct_answer": correct_answer,
        "model_response": "",
        "predicted_answer": "",
        "episodic_retrieved": 0,
        "semantic_retrieved": 0,
        "profile_present": False,
        "assistant_knowledge_count": 0,
        "knowledge_nodes": 0,
        "error": "",
    }

    work_parent = Path(results_root) / ".qa_work"
    work_parent.mkdir(parents=True, exist_ok=True)
    work_dir = None

    try:
        if not cache_is_ready(cache_dir):
            raise RuntimeError(f"cache not READY: {cache_dir}")

        work_dir = copy_cache_for_question(cache_dir, work_parent, qid)
        graph, kb, _retriever = load_cached_state(work_dir)

        # Retrieval happens on the private per-question copy because
        # Hebbian retrieval reinforces edges.
        episodic_results = graph.retrieve(question, top_k=top_k)
        history = format_retrieval(episodic_results)

        kb_results = add_semantic_context(
            history,
            kb,
            question,
            shared_id,
            semantic_top_k=semantic_top_k,
        )

        prompt = question + "\n\n" + INSTRUCTION + "\n\n" + all_options
        response = generate_answers(llm, [prompt], [history])[0]
        score, predicted = extract_answer(response, correct_answer)

        knowledge_nodes, assistant_count = state_counts(kb)
        profile = kb.get_raw_user_profile(shared_id) or ""

        base_result.update(
            {
                "score": bool(score),
                "model_response": response,
                "predicted_answer": predicted,
                "episodic_retrieved": len(episodic_results),
                "semantic_retrieved": len(kb_results),
                "profile_present": bool(profile.strip()),
                "assistant_knowledge_count": assistant_count,
                "knowledge_nodes": knowledge_nodes,
            }
        )

        print(
            f"[QA] {qid} score={bool(score)} "
            f"epi={len(episodic_results)} sem={len(kb_results)}"
        )

    except Exception as e:
        base_result["error"] = f"{type(e).__name__}: {e}"
        print(f"[QA ERROR] {qid}: {base_result['error']}")
        traceback.print_exc()

    finally:
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)

    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(base_result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(out_path))
    return base_result


def aggregate_results(records, results_root, output_prefix, require_full=False):
    rows = []
    missing = []
    seen = set()

    for row, _context in records:
        qid = row["question_id"]
        p = result_path_for(results_root, qid)
        if not p.exists():
            missing.append(qid)
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        if qid in seen:
            raise AssertionError(f"duplicate question_id in dataset: {qid}")
        seen.add(qid)
        rows.append(obj)

    failures = [r for r in rows if r.get("error")]

    if require_full:
        if missing:
            raise AssertionError(f"full-run audit failed: missing={len(missing)}")
        if len(rows) != len(records):
            raise AssertionError(
                f"full-run audit failed: rows={len(rows)}, expected={len(records)}"
            )

    df = pd.DataFrame(rows)
    csv_path = Path(f"{output_prefix}.csv")
    json_path = Path(f"{output_prefix}.json")
    errors_path = Path(results_root) / "errors.jsonl"
    audit_path = Path(results_root) / "audit.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(errors_path, "w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    audit = {
        "expected_questions": len(records),
        "dataset_rows": len(records),
        "result_rows": len(rows),
        "missing": len(missing),
        "missing_question_ids": missing,
        "duplicate_question_ids": 0,
        "error_rows": len(failures),
        "scored_rows": len(rows) - len(failures),
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== AUDIT ===")
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    if len(df):
        report(df)

    print("\nSaved:")
    print(csv_path)
    print(json_path)
    print(errors_path)
    print(audit_path)
    return audit


def main():
    parser = argparse.ArgumentParser(description="HeLa-Mem PersonaMem persistent real-cutoff cache runner")
    parser.add_argument("--phase", choices=["encode", "qa", "all"], default="all")
    parser.add_argument("--num_items", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--semantic_top_k", type=int, default=5)
    parser.add_argument("--cache_root", default="personamem_cache")
    parser.add_argument("--results_root", default="personamem_results")
    parser.add_argument("--output_prefix", default="results_hela_personamem_cached")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    records = load_dataset_records(args.num_items)
    cutoffs = unique_cutoffs(records)

    print("=" * 70)
    print("HeLa-Mem PersonaMem persistent-cache runner")
    print("=" * 70)
    print("questions:", len(records))
    print("unique real cutoffs:", len(cutoffs))
    print("phase:", args.phase)
    print("cache_root:", args.cache_root)
    print("results_root:", args.results_root)

    if args.num_items is None:
        if len(records) == 0 or len(cutoffs) == 0:
            raise AssertionError("empty PersonaMem dataset or cutoff set")
        print(f"FULL DATASET COUNT AUDIT: PASS ({len(records)} questions / {len(cutoffs)} cutoffs)")

    if args.dry_run:
        print("DRY RUN: no model loaded, no cache written, no QA performed")
        return

    # One shared vLLM instance per process.
    llm = get_llm()
    client = LocalQwenClient()

    if args.phase in {"encode", "all"}:
        print("\n=== ENCODE PHASE ===")
        for i, (_key, (row, context)) in enumerate(cutoffs.items(), start=1):
            print(f"\n[{i}/{len(cutoffs)}]")
            build_cutoff_cache(
                row=row,
                context=context,
                cache_root=args.cache_root,
                client=client,
                force=args.force_rebuild,
            )

    if args.phase in {"qa", "all"}:
        print("\n=== QA PHASE ===")
        for i, (row, _context) in enumerate(records, start=1):
            print(f"\n[{i}/{len(records)}]")
            cache_dir = cache_dir_for(
                args.cache_root,
                row["shared_context_id"],
                row["end_index_in_shared_context"],
            )
            evaluate_question(
                row=row,
                cache_dir=cache_dir,
                results_root=args.results_root,
                llm=llm,
                top_k=args.top_k,
                semantic_top_k=args.semantic_top_k,
            )

        aggregate_results(
            records=records,
            results_root=args.results_root,
            output_prefix=args.output_prefix,
            require_full=(args.num_items is None),
        )


if __name__ == "__main__":
    main()
