import csv
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import igraph as ig
import torch

BASE = Path("/root/autodl-tmp/paper_repro/adapters/memgas_personamem_128k")
MEMGAS = Path("/root/autodl-tmp/paper_repro/MemGAS")
QUESTIONS_CSV = Path(
    "/root/autodl-tmp/paper_repro/shared_data/personamem_32k/questions_128k.csv"
)

# Needed because run_retrieval.py imports eval_utils as a bare module.
sys.path.insert(0, str(MEMGAS))
sys.path.insert(0, str(MEMGAS / "src" / "retrieval"))

from src.construct.construct_asso import get_edges
from src.retrieval.run_retrieval import (
    multi_granularity_routing,
    run_ppr,
)

CACHE_PATH = BASE / "personamem_128k_contriever_cache.pt"
SESSION_MAP_PATH = BASE / "question_session_hashes.jsonl"
UNIQUE_PATH = BASE / "unique_sessions.jsonl"

OUT_PATH = BASE / "personamem_memgas_retrieval_128k_smoke.jsonl"
AUDIT_PATH = BASE / "personamem_memgas_retrieval_128k_smoke_audit.json"

# Official README MemGAS parameters.
ARGS = SimpleNamespace(
    num_seednodes=15,
    mem_threshold=30,
    n_components=2,
    damping=0.1,
    temp=0.2,
)


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def extract_question_id(row):
    for k in ("question_id", "qid", "id"):
        if k in row:
            return str(row[k])
    raise KeyError(
        f"Cannot find question id. keys={list(row.keys())}"
    )


def extract_hashes(row):
    """
    Accept only a small set of plausible adapter field names.
    We do NOT infer/reorder sessions.
    """
    for k in (
        "session_hashes",
        "hashes",
        "sessions",
    ):
        if k in row:
            v = row[k]
            if not isinstance(v, list):
                raise TypeError(f"{k} is not a list")
            return v
    raise KeyError(
        f"Cannot find session hash list. keys={list(row.keys())}"
    )


print("===== LOAD DATA =====", flush=True)
if OUT_PATH.exists() or AUDIT_PATH.exists():
    raise FileExistsError("128K retrieval output already exists; refusing overwrite")


cache = torch.load(
    CACHE_PATH,
    map_location="cpu",
)

assert cache["meta"]["dataset"] == "PersonaMem-128K"
assert cache["meta"]["adapter"] == "memgas_personamem_128k"
assert cache["meta"]["num_unique_sessions"] == 1249
assert cache["meta"]["num_questions"] == 2727
assert len(cache["session_hashes"]) == 1249
assert len(cache["question_ids"]) == 2727
assert cache["meta"]["embedding_dim"] == 768

question_session_rows = load_jsonl(
    SESSION_MAP_PATH
)

unique_rows = load_jsonl(
    UNIQUE_PATH
)

assert len(question_session_rows) == 2727
assert len(unique_rows) == 1249


# ------------------------------------------------------------
# Unique session lookup
# ------------------------------------------------------------

unique_by_hash = {}

for x in unique_rows:
    h = x["hash"]
    assert h not in unique_by_hash
    unique_by_hash[h] = x

cache_hash_to_idx = {
    h: i
    for i, h in enumerate(cache["session_hashes"])
}

assert set(cache_hash_to_idx) == set(unique_by_hash)


# ------------------------------------------------------------
# Question metadata
# ------------------------------------------------------------

questions = []

with open(
    QUESTIONS_CSV,
    encoding="utf-8-sig",
    newline="",
) as f:
    for row in csv.DictReader(f):
        questions.append(row)

assert len(questions) == 2727

question_by_id = {
    str(x["question_id"]): x
    for x in questions
}

cache_question_to_idx = {
    str(qid): i
    for i, qid in enumerate(cache["question_ids"])
}

assert set(question_by_id) == set(cache_question_to_idx)


# ------------------------------------------------------------
# question -> ordered session hashes
# ------------------------------------------------------------

question_sessions = {}

for row in question_session_rows:
    qid = extract_question_id(row)
    hashes = extract_hashes(row)

    assert qid not in question_sessions
    assert len(hashes) >= 1

    for h in hashes:
        assert h in cache_hash_to_idx, (
            qid,
            h,
        )

    question_sessions[qid] = hashes

assert set(question_sessions) == set(question_by_id)


print("questions:", len(question_by_id), flush=True)
print("unique sessions:", len(unique_by_hash), flush=True)


def build_official_graph(raw, turn, summary, keyword):
    """
    Faithfully mirrors construct_asso.py.

    Four nodes per session:
      session
      turn
      summary
      keyword

    CRITICAL:
      edge construction begins at node index 5*4 == 20,
      exactly as official code.
    """

    n_sessions = raw.size(0)

    node_embs = []

    for i in range(n_sessions):
        node_embs.append(raw[i])
        node_embs.append(turn[i])
        node_embs.append(summary[i])
        node_embs.append(keyword[i])

    node_ids = []

    for i in range(n_sessions):
        sid = f"session_{i+1}"

        node_ids.extend([
            f"{sid}-session",
            f"{sid}-turn",
            f"{sid}-summary",
            f"{sid}-keyword",
        ])

    all_edges = []

    # EXACT official behavior.
    for i in range(5 * 4, len(node_ids)):
        cur_mem = node_embs[i]

        pre_mem = node_embs[
            : i // 4 * 4
        ]

        edges = get_edges(
            cur_mem,
            i,
            pre_mem,
            ARGS.mem_threshold,
            ARGS.n_components,
        )

        all_edges += edges

    g = ig.Graph(
        n=len(node_ids),
        edges=all_edges,
        directed=False,
    )

    g.vs["name"] = node_ids

    return g


def retrieve_one(qid):
    hashes = question_sessions[qid]

    indices = [
        cache_hash_to_idx[h]
        for h in hashes
    ]

    raw = cache["raw"][indices]
    turn = cache["turn_mean"][indices]
    summary = cache["summary"][indices]
    keyword = cache["keyword"][indices]

    q_idx = cache_question_to_idx[qid]
    q_emb = cache["question_embeddings"][q_idx]

    n_sessions = len(hashes)

    assert raw.shape == (n_sessions, 768)
    assert turn.shape == (n_sessions, 768)
    assert summary.shape == (n_sessions, 768)
    assert keyword.shape == (n_sessions, 768)
    assert q_emb.shape == (768,)

    # -------------------------
    # Official graph
    # -------------------------

    g = build_official_graph(
        raw,
        turn,
        summary,
        keyword,
    )

    # -------------------------
    # Official entropy routing
    # -------------------------

    emb_list = [
        raw,
        turn,
        summary,
        keyword,
    ]

    router_weights = multi_granularity_routing(
        ARGS,
        q_emb,
        emb_list,
    )

    weighted_embs = [
        w * v
        for w, v in zip(
            router_weights,
            emb_list,
        )
    ]

    # Official interleaving:
    # sess1-session
    # sess1-turn
    # sess1-summary
    # sess1-keyword
    # sess2-session ...
    multi_gran_emb = []

    for i in range(n_sessions):
        for e in weighted_embs:
            multi_gran_emb.append(e[i])

    multi_gran_emb = torch.stack(
        multi_gran_emb,
        dim=0,
    )

    scores = (
        q_emb @ multi_gran_emb.T
    ).squeeze()

    # All PersonaMem instances have at least 16 nodes,
    # so official top-15 is valid.
    assert scores.numel() >= ARGS.num_seednodes

    topk_values, _ = torch.topk(
        scores,
        ARGS.num_seednodes,
    )

    seed_threshold = topk_values[-1]

    # EXACT official behavior:
    # values below threshold become zero.
    reset_scores = scores.clone()

    reset_scores[
        reset_scores < seed_threshold
    ] = 0

    # -------------------------
    # Official Personalized PageRank
    # -------------------------

    ppr_scores = run_ppr(
        g,
        reset_scores.numpy(),
        ARGS.damping,
    )

    # Official aggregation:
    # sum each group of four nodes.
    session_scores = [
        float(
            sum(
                ppr_scores[i:i+4]
            )
        )
        for i in range(
            0,
            len(ppr_scores),
            4,
        )
    ]

    rankings = sorted(
        range(n_sessions),
        key=lambda i: session_scores[i],
        reverse=True,
    )

    return {
        "hashes": hashes,
        "graph": g,
        "router_weights": router_weights,
        "session_scores": session_scores,
        "rankings": rankings,
        "seed_threshold": float(
            seed_threshold
        ),
    }


# ------------------------------------------------------------
# Main retrieval
# ------------------------------------------------------------

results = []

session_count_dist = Counter()
edge_count_dist = Counter()

zero_edge_questions = 0

for num_done, qid in enumerate(
    cache["question_ids"][:3],
    1,
):
    qid = str(qid)

    qmeta = question_by_id[qid]

    r = retrieve_one(qid)

    hashes = r["hashes"]
    rankings = r["rankings"]
    session_scores = r["session_scores"]

    n_sessions = len(hashes)
    n_edges = r["graph"].ecount()

    session_count_dist[n_sessions] += 1
    edge_count_dist[n_edges] += 1

    if n_edges == 0:
        zero_edge_questions += 1

    ranked_items = []

    for rank, idx in enumerate(
        rankings,
        1,
    ):
        h = hashes[idx]

        item = {
            "rank": rank,
            "session_id": f"session_{idx+1}",
            "session_hash": h,
            "score": session_scores[idx],
        }

        # Preserve every rank, but only store full text for QA Top-3.
        if rank <= 3:
            item["session"] = unique_by_hash[h]["session"]

        ranked_items.append(item)

    result = {
        "question_id": qid,
        "persona_id": qmeta["persona_id"],
        "question_type": qmeta["question_type"],
        "question": qmeta[
            "user_question_or_message"
        ],
        "correct_answer": qmeta["correct_answer"],
        "all_options": qmeta["all_options"],

        "num_visible_sessions": n_sessions,

        "graph": {
            "num_nodes": r["graph"].vcount(),
            "num_edges": n_edges,
        },

        "router_weights": {
            "session": float(
                r["router_weights"][0]
            ),
            "turn": float(
                r["router_weights"][1]
            ),
            "summary": float(
                r["router_weights"][2]
            ),
            "keyword": float(
                r["router_weights"][3]
            ),
        },

        "seed_threshold": r["seed_threshold"],

        "retrieval_results": {
            "ranked_items": ranked_items,
        },
    }

    results.append(result)

    if (
        num_done <= 5
        or num_done % 50 == 0
        or num_done == 2727
    ):
        print(
            f"[{num_done}/2727]"
            f" qid={qid}"
            f" sessions={n_sessions}"
            f" nodes={r['graph'].vcount()}"
            f" edges={n_edges}"
            f" top1={ranked_items[0]['session_id']}",
            flush=True,
        )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

with open(
    OUT_PATH,
    "w",
    encoding="utf-8",
) as f:
    for x in results:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


audit = {
    "num_questions": len(results),

    "parameters": {
        "num_seednodes": ARGS.num_seednodes,
        "mem_threshold": ARGS.mem_threshold,
        "n_components": ARGS.n_components,
        "damping": ARGS.damping,
        "temp": ARGS.temp,
    },

    "session_count_distribution": dict(
        sorted(session_count_dist.items())
    ),

    "zero_edge_questions": zero_edge_questions,

    "edge_count_distribution": dict(
        sorted(edge_count_dist.items())
    ),

    "notes": [
        (
            "Graph construction preserves official "
            "range(5*4, len(node_ids)) behavior."
        ),
        (
            "PersonaMem has no answer_session_ids; "
            "no retrieval Recall/NDCG was fabricated."
        ),
    ],
}

with open(
    AUDIT_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        audit,
        f,
        ensure_ascii=False,
        indent=2,
    )


print("\n===== FINAL =====", flush=True)
print("results:", len(results), flush=True)
print(
    "session_count_distribution:",
    dict(sorted(session_count_dist.items())),
    flush=True,
)
print(
    "zero_edge_questions:",
    zero_edge_questions,
    flush=True,
)
print(
    "saved:",
    OUT_PATH,
    flush=True,
)

print(
    "PERSONAMEM_MEMGAS_RETRIEVAL_OK",
    flush=True,
)
