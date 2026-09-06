import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util


BASELINE_DIR = (
    "/root/autodl-tmp/research/results/"
    "table3_gpt4o/baselines"
)
sys.path.insert(0, BASELINE_DIR)

from run_fewshot_cot_retrieval import (
    load_pool,
    retrieve_top3,
    build_prompt,
    parse_output,
)


def normalize_strategy(x):
    if not x:
        return None
    return " ".join(x.lower().strip().split())


def select_consistent_path(paths, model):
    """
    Reconstruction rule:
    1. Majority vote over predicted strategy.
    2. Among paths with majority strategy, choose semantic medoid response.
    3. If no strategy majority, semantic medoid over all responses.
    """

    valid = [
        x for x in paths
        if x.get("response")
    ]

    if not valid:
        raise RuntimeError("No valid paths")

    normalized = [
        normalize_strategy(x.get("pred_strategy"))
        for x in valid
    ]

    counter = Counter(
        x for x in normalized
        if x is not None
    )

    majority_strategy = None

    if counter:
        strategy, count = counter.most_common(1)[0]

        # 3 paths -> at least 2 constitute majority
        if count >= 2:
            majority_strategy = strategy

    if majority_strategy is not None:
        candidates = [
            p for p, s in zip(valid, normalized)
            if s == majority_strategy
        ]
    else:
        candidates = valid

    if len(candidates) == 1:
        return candidates[0], majority_strategy

    responses = [
        x["response"]
        for x in candidates
    ]

    emb = model.encode(
        responses,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    sims = util.cos_sim(emb, emb)

    # average similarity to other candidate answers
    mean_scores = sims.mean(dim=1)

    best = int(
        torch.argmax(mean_scores).item()
    )

    return candidates[best], majority_strategy


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default=(
            "/root/autodl-tmp/research/results/"
            "multiagentesc_qwen25_32b_100.json"
        )
    )

    ap.add_argument(
        "--pool",
        default=(
            "/root/autodl-tmp/research/results/"
            "table3_gpt4o/baselines/"
            "fewshot_pool267.jsonl"
        )
    )

    ap.add_argument(
        "--output",
        default=(
            "/root/autodl-tmp/research/results/"
            "table3_gpt4o/baselines/"
            "self_consistency_1210.jsonl"
        )
    )

    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--paths", type=int, default=3)

    args = ap.parse_args()

    if args.paths != 3:
        raise ValueError(
            "This reconstruction is frozen to 3 paths."
        )

    with open(
        args.input,
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    pool = load_pool(args.pool)

    print("usable few-shot pool =", len(pool))

    model = SentenceTransformer(
        "all-roberta-large-v1"
    )

    pool_embeddings = model.encode(
        [x["context"] for x in pool],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    client = OpenAI(
        base_url="http://127.0.0.1:11434/v1",
        api_key="NULL",
        timeout=600,
        max_retries=0,
    )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    completed = set()

    if output.exists():
        with output.open(
            encoding="utf-8"
        ) as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    x = json.loads(line)

                    if x.get("status") == "ok":
                        completed.add(
                            x["index"]
                        )

                except Exception:
                    pass

    end = (
        len(data)
        if args.limit is None
        else min(args.limit, len(data))
    )

    print("process =", end)
    print("paths per item =", args.paths)
    print("already completed =", len(completed))

    for i in range(end):
        if i in completed:
            continue

        item = data[i]

        examples = retrieve_top3(
            item["context"],
            pool,
            model,
            pool_embeddings,
        )

        prompt = build_prompt(
            item["context"],
            examples,
        )

        paths = []
        last_error = None

        for path_id in range(args.paths):
            path_result = None

            for attempt in range(1, 4):
                try:
                    r = client.chat.completions.create(
                        model="qwen2.5:32b",
                        temperature=0,
                        max_tokens=400,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                    )

                    raw = (
                        r.choices[0]
                        .message.content
                        .strip()
                    )

                    strategy, reasoning, response = (
                        parse_output(raw)
                    )

                    if not response:
                        raise ValueError(
                            "Response parse failed"
                        )

                    path_result = {
                        "path_id": path_id,
                        "pred_strategy":
                            strategy,
                        "reasoning":
                            reasoning,
                        "response":
                            response,
                        "raw_response":
                            raw,
                    }

                    break

                except Exception as e:
                    last_error = repr(e)

                    print(
                        f"[{i+1}/{end}] "
                        f"path {path_id+1}/3 "
                        f"attempt {attempt}/3 "
                        f"failed: {last_error}"
                    )

                    time.sleep(
                        attempt * 3
                    )

            if path_result is None:
                break

            paths.append(path_result)

        if len(paths) == args.paths:
            selected, majority_strategy = (
                select_consistent_path(
                    paths,
                    model,
                )
            )

            unique_responses = len(
                set(
                    x["response"]
                    for x in paths
                )
            )

            unique_strategies = len(
                set(
                    normalize_strategy(
                        x.get(
                            "pred_strategy"
                        )
                    )
                    for x in paths
                )
            )

            result = {
                "index": i,
                "status": "ok",
                "strategy":
                    item.get("strategy"),
                "reference":
                    item.get("reference"),
                "context":
                    item["context"],

                "response":
                    selected["response"],
                "pred_strategy":
                    selected[
                        "pred_strategy"
                    ],
                "reasoning":
                    selected["reasoning"],

                "paths":
                    paths,

                "path_count":
                    args.paths,
                "unique_path_responses":
                    unique_responses,
                "unique_path_strategies":
                    unique_strategies,
                "majority_strategy":
                    majority_strategy,

                "retrieved_examples": [
                    {
                        "pool_index":
                            x["pool_index"],
                        "source_dialog_idx":
                            x[
                                "source_dialog_idx"
                            ],
                        "strategy":
                            x["strategy"],
                        "similarity":
                            round(
                                x["similarity"],
                                6
                            ),
                    }
                    for x in examples
                ],

                "method":
                    "Self-consistency reconstruction",

                "model":
                    "qwen2.5:32b",
                "temperature":
                    0,

                "reconstruction": {
                    "base_prompt":
                        "Few-shot CoT Appendix Prompt 11 reconstruction",
                    "reasoning_paths":
                        3,
                    "strategy_selection":
                        "majority vote",
                    "response_selection":
                        "semantic medoid within majority strategy",
                    "semantic_model":
                        "all-roberta-large-v1",
                    "author_path_count_reported":
                        False,
                    "author_selection_rule_reported":
                        False,
                },
            }

        else:
            result = {
                "index": i,
                "status": "error",
                "context":
                    item["context"],
                "error":
                    last_error,
                "completed_paths":
                    len(paths),
            }

        with output.open(
            "a",
            encoding="utf-8"
        ) as f:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                )
                + "\n"
            )

        print(
            f"[{i+1}/{end}]",
            result["status"],
            "unique_paths=",
            result.get(
                "unique_path_responses"
            ),
            "strategy=",
            result.get(
                "pred_strategy"
            ),
            "response=",
            repr(
                result.get(
                    "response", ""
                )
            )[:80],
        )

    # ----- sanity check -----

    rows = []

    with output.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                x = json.loads(line)

                if (
                    x.get("status") == "ok"
                    and x["index"] < end
                ):
                    rows.append(x)

    responses = [
        x["response"]
        for x in rows
    ]

    strategies = [
        x.get("pred_strategy")
        for x in rows
    ]

    diverse_path_items = sum(
        x.get(
            "unique_path_responses", 0
        ) > 1
        for x in rows
    )

    print("\n=== SANITY CHECK ===")
    print("ok =", len(rows))
    print(
        "unique final responses =",
        len(set(responses))
    )
    print(
        "unique final strategies =",
        len(set(strategies))
    )
    print(
        "items with >1 unique path response =",
        diverse_path_items
    )

    if len(rows) >= 10:
        if len(set(responses)) <= 2:
            print(
                "SANITY = FAIL: response collapse"
            )
        elif len(set(strategies)) <= 1:
            print(
                "SANITY = FAIL: strategy collapse"
            )
        else:
            print("SANITY = PASS")


if __name__ == "__main__":
    main()
