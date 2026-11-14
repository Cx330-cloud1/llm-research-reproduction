import os

# 必须在导入 HeLa-Mem 模块之前设置
os.environ["HELA_DEFER_KB_SAVES"] = "1"
os.environ["HEBBIAN_KNOWLEDGE_BUFFER_SIZE"] = "10"

import gc
import json
import copy
import shutil
import argparse
import tempfile

from pathlib import Path
from collections import defaultdict

from eval_personamem_cached import (
    load_dataset_records,
    unique_cutoffs,
    LocalQwenClient,
    HebbianMemoryGraph,
    HebbianRetriever,
    HebbianKnowledgeMemory,
    process_incremental_buffer,
    BUFFER_SIZE,
    get_llm,
)

from eval_personamem_local import context_to_turns


TIMESTAMP = "unknown"

EXPECTED_QUESTIONS = 2674
EXPECTED_CUTOFFS = 276


# ============================================================
# 1. Cache utilities
# ============================================================

def cache_dir_for(root, shared_id, end_idx):
    return (
        Path(root)
        / str(shared_id)
        / f"end_{int(end_idx)}"
    )


def cache_is_ready(cache_dir):
    cache_dir = Path(cache_dir)

    required = [
        cache_dir / "READY",
        cache_dir / "memory_hebbian.json",
        cache_dir / "memory_long_term.json",
        cache_dir / "memory_long_term_kb_graph.json",
        cache_dir / "metadata.json",
    ]

    return all(
        p.is_file() and p.stat().st_size > 0
        for p in required
    )


def get_state_counts(kb):
    knowledge_nodes = len(
        kb.knowledge_graph.nodes
    )

    assistant_count = len(
        kb.get_assistant_knowledge() or []
    )

    profile = (
        kb.get_raw_user_profile(
            CURRENT_SHARED_ID
        ) or ""
    )

    return (
        knowledge_nodes,
        assistant_count,
        bool(profile.strip()),
    )


# ============================================================
# 2. Save independent cutoff snapshot
# ============================================================

def save_cutoff_snapshot(
    graph,
    kb,
    tail_buffer,
    row,
    context,
    cache_root,
    client,
    force=False,
):
    shared_id = str(row["shared_context_id"])

    end_idx = int(
        row["end_index_in_shared_context"]
    )

    qid = row["question_id"]

    final_dir = cache_dir_for(
        cache_root,
        shared_id,
        end_idx,
    )

    if cache_is_ready(final_dir) and not force:
        print(
            f"[CACHE HIT] {shared_id[:12]} "
            f"end_{end_idx}"
        )
        return

    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".end_{end_idx}.tmp_",
            dir=str(parent),
        )
    )

    try:
        # Critical:
        # Each cutoff gets independent graph and KB objects.
        snap_graph = copy.deepcopy(graph)
        snap_kb = copy.deepcopy(kb)

        # Redirect all save paths to this temporary directory.
        graph_path = tmp_dir / "memory_hebbian.json"
        kb_path = tmp_dir / "memory_long_term.json"
        kb_graph_path = (
            tmp_dir / "memory_long_term_kb_graph.json"
        )

        snap_graph.file_path = str(graph_path)

        snap_kb.file_path = str(kb_path)
        snap_kb.hebbian_kb_path = str(kb_graph_path)

        snap_kb.knowledge_graph.file_path = str(
            kb_graph_path
        )

        # Keep deferred saving enabled.
        snap_kb.defer_saves = True

        # ----------------------------------------------------
        # Important:
        # Process incomplete batch ONLY on the snapshot.
        # Never mutate the canonical prefix KB.
        # ----------------------------------------------------

        if tail_buffer:
            process_incremental_buffer(
                copy.deepcopy(tail_buffer),
                snap_kb,
                shared_id,
                client,
            )

        # Save all three independent memory states.
        snap_graph.save()
        snap_kb.save(force=True)

        knowledge_nodes = len(
            snap_kb.knowledge_graph.nodes
        )

        assistant_count = len(
            snap_kb.get_assistant_knowledge() or []
        )

        profile = (
            snap_kb.get_raw_user_profile(shared_id)
            or ""
        )

        metadata = {
            "cache_version": 1,
            "shared_context_id": shared_id,
            "end_index_in_shared_context": end_idx,
            "source_question_id": qid,

            "messages": len(context),
            "turns": len(context_to_turns(context)),

            "episodic_nodes": len(snap_graph.nodes),
            "knowledge_nodes": knowledge_nodes,
            "assistant_knowledge_count": assistant_count,

            "profile_present": bool(profile.strip()),

            "time_decay": False,
            "timestamp_policy": TIMESTAMP,
            "buffer_size": BUFFER_SIZE,
            "llm_backend": "Qwen3-8B-Base",

            "cache_build_mode": "shared_prefix_exact",
        }

        with open(
            tmp_dir / "metadata.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Validate all JSON files before publishing.
        for filename in [
            "memory_hebbian.json",
            "memory_long_term.json",
            "memory_long_term_kb_graph.json",
            "metadata.json",
        ]:
            path = tmp_dir / filename

            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(
                    f"Missing or empty cache file: {filename}"
                )

            with open(path, encoding="utf-8") as f:
                json.load(f)

        # READY must be created last.
        (tmp_dir / "READY").write_text(
            "ok\n",
            encoding="utf-8",
        )

        # Atomic publication.
        if final_dir.exists():
            shutil.rmtree(final_dir)

        os.replace(
            str(tmp_dir),
            str(final_dir),
        )

        print(
            f"[CACHE READY] "
            f"{shared_id[:12]} end_{end_idx} "
            f"episodic={metadata['episodic_nodes']} "
            f"knowledge={knowledge_nodes} "
            f"assistant={assistant_count} "
            f"tail={len(tail_buffer)}"
        )

        del snap_graph
        del snap_kb
        gc.collect()

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# 3. Shared-prefix encoding
# ============================================================

CURRENT_SHARED_ID = None


def encode_shared_context(
    shared_id,
    cutoff_items,
    cache_root,
    client,
    force=False,
):
    global CURRENT_SHARED_ID

    CURRENT_SHARED_ID = shared_id

    cutoff_items.sort(
        key=lambda x: x[0]
    )

    # The longest cutoff defines the canonical stream.
    max_row = cutoff_items[-1][1]
    max_context = cutoff_items[-1][2]

    all_turns = context_to_turns(max_context)

    # --------------------------------------------------------
    # Verify every cutoff is a true prefix.
    # --------------------------------------------------------

    for end_idx, row, context, turns in cutoff_items:

        if turns != all_turns[:len(turns)]:
            raise AssertionError(
                f"Prefix mismatch: "
                f"{shared_id}, end={end_idx}"
            )

    # If all cutoff caches are already complete, skip group.
    if not force:
        all_ready = all(
            cache_is_ready(
                cache_dir_for(
                    cache_root,
                    shared_id,
                    end_idx,
                )
            )
            for end_idx, _, _, _ in cutoff_items
        )

        if all_ready:
            print(
                f"[GROUP COMPLETE] "
                f"{shared_id[:12]}"
            )
            return

    print("\n" + "=" * 72)
    print(
        f"[GROUP START] {shared_id[:12]} "
        f"cutoffs={len(cutoff_items)} "
        f"max_turns={len(all_turns)}"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Canonical state.
    #
    # It contains:
    # 1. All episodic turns processed so far.
    # 2. Knowledge extracted from complete 10-turn batches.
    # 3. An uncommitted partial knowledge buffer.
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="hela_prefix_canonical_"
    ) as scratch:

        scratch = Path(scratch)

        graph = HebbianMemoryGraph(
            file_path=str(
                scratch / "canonical_hebbian.json"
            )
        )

        kb = HebbianKnowledgeMemory(
            file_path=str(
                scratch / "canonical_long_term.json"
            )
        )

        if not kb.defer_saves:
            raise RuntimeError(
                "Deferred KB saving is not enabled."
            )

        retriever = HebbianRetriever(
            graph,
            profile_memory=kb,
        )

        buffer = []
        current_turn = 0

        # Episodic graph batch configuration.
        episodic_pending = []

        episodic_batch_size = max(
            1,
            int(os.environ.get(
                "HELA_EPISODIC_BATCH_SIZE", "64"
            )),
        )

        embedding_batch_size = max(
            1,
            int(os.environ.get(
                "HELA_EMBED_BATCH_SIZE", "64"
            )),
        )

        keyword_batch_size = max(
            1,
            int(os.environ.get(
                "HELA_KEYWORD_BATCH_SIZE", "16"
            )),
        )

        # ----------------------------------------------------
        # Process cutoffs in ascending order.
        # ----------------------------------------------------

        for cutoff_index, (
            end_idx,
            row,
            context,
            turns,
        ) in enumerate(cutoff_items, start=1):

            target_turn = len(turns)

            if target_turn < current_turn:
                raise AssertionError(
                    "Cutoff turns are not monotonically increasing."
                )

            # Advance canonical stream only once.
            while current_turn < target_turn:

                user_text, assistant_text = (
                    all_turns[current_turn]
                )

                # Queue episodic memory instead of
                # generating embedding and keywords immediately.
                episodic_pending.append({
                    "content": f"User: {user_text}\nAI: {assistant_text}",
                    "role": "interaction",
                    "timestamp": TIMESTAMP,
                })

                if len(episodic_pending) >= episodic_batch_size:
                    graph.add_memories_batch(
                        episodic_pending,
                        embedding_batch_size=embedding_batch_size,
                        keyword_batch_size=keyword_batch_size,
                    )

                    episodic_pending = []

                buffer.append({
                    "user_text": user_text,
                    "ai_text": assistant_text,
                    "timestamp": TIMESTAMP,
                })

                current_turn += 1

                # Process only complete batches on canonical state.
                if len(buffer) >= BUFFER_SIZE:

                    if len(buffer) != BUFFER_SIZE:
                        raise AssertionError(
                            "Unexpected incremental buffer size."
                        )

                    process_incremental_buffer(
                        list(buffer),
                        kb,
                        shared_id,
                        client,
                    )

                    buffer = []

            # ------------------------------------------------
            # Snapshot this cutoff.
            #
            # The remaining buffer is replayed only on a copy.
            # The canonical state remains untouched.
            # ------------------------------------------------

            if episodic_pending:
                graph.add_memories_batch(
                    episodic_pending,
                    embedding_batch_size=embedding_batch_size,
                    keyword_batch_size=keyword_batch_size,
                )

                episodic_pending = []

            save_cutoff_snapshot(
                graph=graph,
                kb=kb,
                tail_buffer=buffer,
                row=row,
                context=context,
                cache_root=cache_root,
                client=client,
                force=force,
            )

            print(
                f"[PROGRESS] "
                f"group={shared_id[:12]} "
                f"cutoff={cutoff_index}/{len(cutoff_items)} "
                f"turn={current_turn}/{len(all_turns)} "
                f"pending_tail={len(buffer)}"
            )

        print(
            f"[GROUP FINISHED] "
            f"{shared_id[:12]} "
            f"turns={current_turn}"
        )

    gc.collect()


# ============================================================
# 4. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache_root",
        default="personamem_cache_1M_prefix",
    )

    parser.add_argument(
        "--context_limit",
        type=int,
        default=0,
        help="0 means all shared contexts.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("HeLa-Mem Shared Prefix Cache Builder")
    print("=" * 72)

    print("BUFFER_SIZE:", BUFFER_SIZE)
    print("DEFER SAVES:", os.environ.get(
        "HELA_DEFER_KB_SAVES"
    ))
    print("CACHE ROOT:", args.cache_root)

    if BUFFER_SIZE != 10:
        raise RuntimeError(
            f"Expected BUFFER_SIZE=10, got {BUFFER_SIZE}"
        )

    records = load_dataset_records()

    cutoffs = unique_cutoffs(records)

    if len(records) != EXPECTED_QUESTIONS:
        raise AssertionError(
            f"Expected {EXPECTED_QUESTIONS} questions, "
            f"got {len(records)}"
        )

    if len(cutoffs) != EXPECTED_CUTOFFS:
        raise AssertionError(
            f"Expected {EXPECTED_CUTOFFS} cutoffs, "
            f"got {len(cutoffs)}"
        )

    groups = defaultdict(list)

    for row, context in cutoffs.values():

        shared_id = str(row["shared_context_id"])

        end_idx = int(
            row["end_index_in_shared_context"]
        )

        turns = context_to_turns(context)

        groups[shared_id].append(
            (
                end_idx,
                row,
                context,
                turns,
            )
        )

    print("Questions:", len(records))
    print("Unique cutoffs:", len(cutoffs))
    print("Shared contexts:", len(groups))

    if len(groups) != 31:
        raise AssertionError(
            f"Expected 31 shared contexts, got {len(groups)}"
        )

    # Initialize model once.
    print("\n[MODEL INIT]")

    get_llm()

    client = LocalQwenClient()

    print("[MODEL READY]")

    total_groups = len(groups)

    for group_index, (shared_id, items) in enumerate(
        groups.items(),
        start=1,
    ):

        if (
            args.context_limit > 0
            and group_index > args.context_limit
        ):
            break

        print(
            f"\n[CONTEXT {group_index}/{total_groups}] "
            f"{shared_id[:12]}"
        )

        encode_shared_context(
            shared_id=shared_id,
            cutoff_items=items,
            cache_root=args.cache_root,
            client=client,
            force=args.force,
        )

    # Final audit.
    ready_count = len(
        list(
            Path(args.cache_root).rglob("READY")
        )
    )

    print("\n" + "=" * 72)
    print("PREFIX ENCODING FINISHED")
    print("=" * 72)
    print("READY:", ready_count)
    print("EXPECTED:", EXPECTED_CUTOFFS)

    if args.context_limit == 0 and ready_count != EXPECTED_CUTOFFS:
        raise RuntimeError(
            f"Expected {EXPECTED_CUTOFFS} READY caches, "
            f"got {ready_count}"
        )


if __name__ == "__main__":
    main()
