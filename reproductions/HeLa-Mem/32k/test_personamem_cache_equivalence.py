import os
import json
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict

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

from eval_personamem_local import (
    QUESTION_PATH,
    CONTEXT_PATH,
    load_rows_with_context,
    context_to_turns,
)

from eval_personamem_local_full import LocalQwenClient

from hela_mem.hebbian_memory import HebbianMemoryGraph
from hela_mem.hebbian_retriever import HebbianRetriever
from hela_mem.hebbian_knowledge_memory import HebbianKnowledgeMemory
from hela_mem.encode_longmemeval import (
    process_incremental_buffer,
    BUFFER_SIZE,
)

from hela_mem.local_qwen_bridge import get_llm


TIMESTAMP = "unknown"


def make_state(root, name):
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)

    graph_path = d / "memory_hebbian.json"
    kb_path = d / "memory_long_term.json"

    graph = HebbianMemoryGraph(file_path=str(graph_path))
    kb = HebbianKnowledgeMemory(file_path=str(kb_path))
    retriever = HebbianRetriever(graph, profile_memory=kb)

    return d, graph, kb, retriever


def add_turn(retriever, user_text, assistant_text):
    retriever.process_conversation_turn(
        user_text,
        assistant_text,
        timestamp=TIMESTAMP,
    )


def buffer_item(user_text, assistant_text):
    return {
        "user_text": user_text,
        "ai_text": assistant_text,
        "timestamp": TIMESTAMP,
    }


def flush_buffer(buf, kb, item_id, client):
    if not buf:
        return

    process_incremental_buffer(
        buf,
        kb,
        item_id,
        client,
    )


def force_save(graph, kb):
    graph.save()

    if hasattr(kb, "save"):
        kb.save()

    if hasattr(kb, "knowledge_graph"):
        try:
            kb.knowledge_graph.save()
        except Exception:
            pass


def copy_state_files(src_dir, dst_dir):
    src = Path(src_dir)
    dst = Path(dst_dir)

    if dst.exists():
        shutil.rmtree(dst)

    dst.mkdir(parents=True)

    files = list(src.glob("*.json"))

    if not files:
        raise RuntimeError(
            f"No state JSON found in {src}"
        )

    for p in files:
        shutil.copy2(p, dst / p.name)


def load_state(directory):
    d = Path(directory)

    graph = HebbianMemoryGraph(
        file_path=str(d / "memory_hebbian.json")
    )

    kb = HebbianKnowledgeMemory(
        file_path=str(d / "memory_long_term.json")
    )

    retriever = HebbianRetriever(
        graph,
        profile_memory=kb,
    )

    return graph, kb, retriever


def normalize(x):
    if x is None:
        return ""
    return "\n".join(
        line.rstrip()
        for line in str(x).strip().splitlines()
    )


def graph_contents(graph):
    out = []

    for node in graph.nodes:
        if isinstance(node, dict):
            out.append(
                normalize(node.get("content", ""))
            )
        else:
            out.append(normalize(node))

    return out


def assistant_knowledge(kb):
    try:
        vals = kb.get_assistant_knowledge()
    except Exception:
        return []

    return sorted(
        normalize(x.get("knowledge", ""))
        for x in vals
        if x.get("knowledge", "").strip()
    )


def semantic_knowledge(kb):
    kg = getattr(kb, "knowledge_graph", None)
    if kg is None:
        return []

    return sorted(graph_contents(kg))


def build_reference(
    root,
    turns,
    item_id,
    client,
):
    _, graph, kb, retriever = make_state(
        root,
        "reference",
    )

    buf = []

    for user_text, assistant_text in turns:

        add_turn(
            retriever,
            user_text,
            assistant_text,
        )

        buf.append(
            buffer_item(
                user_text,
                assistant_text,
            )
        )

        if len(buf) >= BUFFER_SIZE:
            flush_buffer(
                buf,
                kb,
                item_id,
                client,
            )
            buf = []

    # Official full-cutoff behavior:
    # flush final incomplete buffer.
    if buf:
        flush_buffer(
            buf,
            kb,
            item_id,
            client,
        )

    force_save(graph, kb)

    return graph, kb


def create_full_batch_checkpoints(
    root,
    turns,
    max_turns,
    item_id,
    client,
):
    state_dir, graph, kb, retriever = make_state(
        root,
        "main_state",
    )

    cp_root = Path(root) / "checkpoints"
    cp_root.mkdir()

    # checkpoint 0
    force_save(graph, kb)
    copy_state_files(
        state_dir,
        cp_root / "turn_0",
    )

    buf = []

    for i, (user_text, assistant_text) in enumerate(
        turns[:max_turns],
        start=1,
    ):

        add_turn(
            retriever,
            user_text,
            assistant_text,
        )

        buf.append(
            buffer_item(
                user_text,
                assistant_text,
            )
        )

        # Main reusable state is flushed ONLY
        # on complete official 10-turn batches.
        if len(buf) == BUFFER_SIZE:

            flush_buffer(
                buf,
                kb,
                item_id,
                client,
            )

            buf = []

            force_save(graph, kb)

            copy_state_files(
                state_dir,
                cp_root / f"turn_{i}",
            )

            print(
                f"  checkpoint saved: turn {i}"
            )

    return cp_root


def build_from_checkpoint(
    root,
    cp_root,
    all_turns,
    target_turns,
    item_id,
    client,
):
    checkpoint_turn = (
        target_turns // BUFFER_SIZE
    ) * BUFFER_SIZE

    cp_dir = (
        Path(cp_root)
        / f"turn_{checkpoint_turn}"
    )

    if not cp_dir.exists():
        raise RuntimeError(
            f"checkpoint missing: {cp_dir}"
        )

    work_dir = (
        Path(root)
        / f"cached_target_{target_turns}"
    )

    copy_state_files(
        cp_dir,
        work_dir,
    )

    graph, kb, retriever = load_state(
        work_dir
    )

    remainder = all_turns[
        checkpoint_turn:target_turns
    ]

    buf = []

    for user_text, assistant_text in remainder:

        add_turn(
            retriever,
            user_text,
            assistant_text,
        )

        buf.append(
            buffer_item(
                user_text,
                assistant_text,
            )
        )

    # Flush remainder ONLY in snapshot copy.
    if buf:
        flush_buffer(
            buf,
            kb,
            item_id,
            client,
        )

    force_save(graph, kb)

    return graph, kb, checkpoint_turn


def compare_states(
    label,
    ref_graph,
    ref_kb,
    cached_graph,
    cached_kb,
    question,
    item_id,
):

    checks = {}

    ref_epi = graph_contents(ref_graph)
    cache_epi = graph_contents(cached_graph)

    checks["episodic_node_count"] = (
        len(ref_epi) == len(cache_epi)
    )

    checks["episodic_content"] = (
        ref_epi == cache_epi
    )

    ref_sem = semantic_knowledge(ref_kb)
    cache_sem = semantic_knowledge(cached_kb)

    checks["knowledge_node_count"] = (
        len(ref_sem) == len(cache_sem)
    )

    checks["knowledge_content"] = (
        ref_sem == cache_sem
    )

    ref_profile = normalize(
        ref_kb.get_raw_user_profile(item_id)
    )
    cache_profile = normalize(
        cached_kb.get_raw_user_profile(item_id)
    )

    checks["profile"] = (
        ref_profile == cache_profile
    )

    if ref_profile != cache_profile:
        print("\n--- REFERENCE PROFILE ---")
        print(ref_profile)

        print("\n--- CACHED PROFILE ---")
        print(cache_profile)

        print("\n--- PROFILE LENGTHS ---")
        print(
            "reference:",
            len(ref_profile),
            "cached:",
            len(cache_profile),
        )

    ref_ak = assistant_knowledge(ref_kb)
    cache_ak = assistant_knowledge(cached_kb)

    checks["assistant_knowledge"] = (
        ref_ak == cache_ak
    )

    # Retrieval may update Hebbian weights,
    # so do it only after structural comparison.
    ref_results = ref_graph.retrieve(
        question,
        top_k=15,
    )

    cache_results = cached_graph.retrieve(
        question,
        top_k=15,
    )

    ref_ret = [
        normalize(x["node"]["content"])
        for x in ref_results
    ]

    cache_ret = [
        normalize(x["node"]["content"])
        for x in cache_results
    ]

    checks["retrieval_content"] = (
        ref_ret == cache_ret
    )

    print("\n" + "=" * 70)
    print("COMPARISON:", label)
    print("=" * 70)

    print(
        "episodic nodes:",
        len(ref_epi),
        len(cache_epi),
    )

    print(
        "knowledge nodes:",
        len(ref_sem),
        len(cache_sem),
    )

    print(
        "assistant knowledge:",
        len(ref_ak),
        len(cache_ak),
    )

    for name, ok in checks.items():
        print(
            f"{name:25s}:",
            "PASS" if ok else "FAIL",
        )

    overall = all(checks.values())

    print(
        "\nOVERALL:",
        "PASS" if overall else "FAIL",
    )

    if not overall:
        print("\nDO NOT USE CACHE VERSION YET.")

    return overall


def main():

    # Initialize Qwen exactly once.
    get_llm()
    client = LocalQwenClient()

    grouped = defaultdict(list)

    for row, context in load_rows_with_context(
        str(QUESTION_PATH),
        str(CONTEXT_PATH),
    ):
        grouped[row["shared_context_id"]].append(
            (row, context)
        )

    # Pick the globally cheapest shared context with >=2 cutoffs.
    # Minimize the SECOND cutoff turn count so the equivalence test stays cheap.
    chosen = None
    best_second_turns = None

    for shared_id, entries in grouped.items():

        unique = {}

        for row, context in entries:
            end_idx = int(
                row["end_index_in_shared_context"]
            )
            unique[end_idx] = (
                row,
                context,
            )

        candidates = []

        for end_idx in sorted(unique):

            row, context = unique[end_idx]
            turns = context_to_turns(context)

            if len(turns) >= 10:
                candidates.append(
                    (
                        len(turns),
                        end_idx,
                        row,
                        context,
                    )
                )

        if len(candidates) >= 2:
            pair = candidates[:2]
            second_turns = pair[1][0]

            if (
                best_second_turns is None
                or second_turns < best_second_turns
            ):
                best_second_turns = second_turns
                chosen = (
                    shared_id,
                    pair,
                )

    if chosen is None:
        raise RuntimeError(
            "Could not find context with two usable cutoffs"
        )

    shared_id, targets = chosen

    print("=" * 70)
    print("CACHE EQUIVALENCE TEST")
    print("=" * 70)
    print("shared_context_id:", shared_id)

    for x in targets:
        print(
            "target:",
            "turns=",
            x[0],
            "end_idx=",
            x[1],
            "question_id=",
            x[2]["question_id"],
        )

    # Largest sliced context contains all turns
    # required by both selected targets.
    largest = max(
        targets,
        key=lambda x: x[0],
    )

    all_turns = context_to_turns(
        largest[3]
    )

    max_turns = largest[0]

    root = tempfile.mkdtemp(
        prefix="hela_cache_ab_"
    )

    try:

        # Build reusable complete-10-turn checkpoints once.
        print("\nBuilding reusable checkpoints...")

        cp_root = create_full_batch_checkpoints(
            root=os.path.join(root, "cached"),
            turns=all_turns,
            max_turns=max_turns,
            item_id=shared_id,
            client=client,
        )

        all_pass = True

        for idx, (
            target_turn_count,
            end_idx,
            row,
            context,
        ) in enumerate(targets):

            print(
                "\n" + "#" * 70
            )
            print(
                f"TARGET {idx + 1}: "
                f"{target_turn_count} turns"
            )
            print(
                "#" * 70
            )

            target_turns = context_to_turns(
                context
            )

            # A: slow reference
            print(
                "\n[A] Building from scratch..."
            )

            ref_root = os.path.join(
                root,
                f"ref_{idx}",
            )

            ref_graph, ref_kb = build_reference(
                ref_root,
                target_turns,
                shared_id,
                client,
            )

            # B: checkpoint resume
            print(
                "\n[B] Restoring checkpoint..."
            )

            cached_root = os.path.join(
                root,
                f"resume_{idx}",
            )

            cached_graph, cached_kb, cp_turn = (
                build_from_checkpoint(
                    cached_root,
                    cp_root,
                    all_turns,
                    target_turn_count,
                    shared_id,
                    client,
                )
            )

            print(
                f"checkpoint turn={cp_turn}, "
                f"replayed={target_turn_count-cp_turn}"
            )

            ok = compare_states(
                label=(
                    f"{target_turn_count} turns / "
                    f"end_idx={end_idx}"
                ),
                ref_graph=ref_graph,
                ref_kb=ref_kb,
                cached_graph=cached_graph,
                cached_kb=cached_kb,
                question=row[
                    "user_question_or_message"
                ],
                item_id=shared_id,
            )

            all_pass = all_pass and ok

        print("\n" + "=" * 70)

        if all_pass:
            print(
                "FINAL RESULT: CACHE EQUIVALENCE PASS"
            )
            print(
                "Safe to proceed to persistent cutoff cache."
            )
        else:
            print(
                "FINAL RESULT: CACHE EQUIVALENCE FAIL"
            )
            print(
                "Do NOT run the 589-item cached experiment."
            )

        print("=" * 70)

    finally:
        shutil.rmtree(
            root,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
