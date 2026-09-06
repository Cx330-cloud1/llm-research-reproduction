import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

REPO = "/root/autodl-tmp/research/repos/MultiAgentESC-official"
sys.path.insert(0, REPO)

from strategy import strategy_definitions


def load_pool(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)

            if x.get("status") != "ok":
                continue

            # The formal target set comes from dataset[:100].
            # Exclude candidate examples originating from those same
            # source dialogues to avoid test leakage.
            if x["source_dialog_idx"] < 100:
                continue

            rows.append(x)

    return rows


def format_examples(examples):
    blocks = []

    for i, x in enumerate(examples, 1):
        blocks.append(
            f"""Example {i}
Context:
{x["context"]}

Strategy: {x["strategy"]}
Reasoning: {x["reasoning"]}
Response: {x["response"]}"""
        )

    return "\n\n".join(blocks)


def build_prompt(context, examples):
    return f"""### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'.

### Dialogue context
{context}

You should select an appropriate emotional support strategy first and then generate a strategy-constrained response.

{strategy_definitions}

Please ensure that you are absolutely fair and do not overly favor any particular strategy.
The following are some examples, all presented in the format of <context
strategy
reasoning
response>.

### Examples
{format_examples(examples)}

Your answer must include the following elements:
Strategy: the most appropriate strategy.
Reasoning: the reason why you choose this strategy.
Response: strategy-constrained response. Response must be fewer than 30 words.

Your answer must follow this format:
Strategy: [strategy]
Reasoning: [reasoning]
Response: [response]
"""


def parse_output(text):
    sm = re.search(
        r"Strategy:\s*(.+?)(?:\n|$)",
        text, re.I
    )

    rm = re.search(
        r"Reasoning:\s*(.*?)(?=\n\s*Response:|\Z)",
        text, re.I | re.S
    )

    pm = re.search(
        r"Response:\s*(.*)",
        text, re.I | re.S
    )

    return (
        sm.group(1).strip().strip("[]") if sm else None,
        rm.group(1).strip() if rm else None,
        pm.group(1).strip() if pm else None,
    )


def retrieve_top3(target, pool, model, pool_embeddings):
    emb = model.encode(
        target,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    scores = util.cos_sim(
        emb,
        pool_embeddings
    )[0]

    ranked = torch.argsort(
        scores,
        descending=True
    ).tolist()

    chosen = []
    used_dialogues = set()

    for idx in ranked:
        x = pool[idx]

        # avoid exact context copy
        if x["context"].strip() == target.strip():
            continue

        # require different source conversations
        if x["source_dialog_idx"] in used_dialogues:
            continue

        chosen.append({
            **x,
            "similarity": float(scores[idx]),
        })

        used_dialogues.add(x["source_dialog_idx"])

        if len(chosen) == 3:
            break

    if len(chosen) != 3:
        raise RuntimeError(
            f"Could only retrieve {len(chosen)} examples"
        )

    return chosen


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default="/root/autodl-tmp/research/results/multiagentesc_qwen25_32b_100.json"
    )

    ap.add_argument(
        "--pool",
        default="/root/autodl-tmp/research/results/table3_gpt4o/baselines/fewshot_pool267.jsonl"
    )

    ap.add_argument(
        "--output",
        default="/root/autodl-tmp/research/results/table3_gpt4o/baselines/fewshot_cot_retrieval_1210.jsonl"
    )

    ap.add_argument("--limit", type=int, default=None)

    args = ap.parse_args()

    data = json.load(
        open(args.input, encoding="utf-8")
    )

    pool = load_pool(args.pool)

    print("usable pool =", len(pool))

    model = SentenceTransformer(
        "all-roberta-large-v1"
    )

    pool_texts = [
        x["context"]
        for x in pool
    ]

    pool_embeddings = model.encode(
        pool_texts,
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

    completed = set()

    if output.exists():
        with output.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    x = json.loads(line)

                    if x.get("status") == "ok":
                        completed.add(x["index"])

    end = (
        len(data)
        if args.limit is None
        else min(args.limit, len(data))
    )

    print("process =", end)
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

        result = None
        last_error = None

        for attempt in range(1, 4):
            try:
                r = client.chat.completions.create(
                    model="qwen2.5:32b",
                    temperature=0,
                    max_tokens=400,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                )

                raw = (
                    r.choices[0]
                    .message.content
                    .strip()
                )

                pred_strategy, reasoning, response = (
                    parse_output(raw)
                )

                if not response:
                    raise ValueError(
                        "Response parse failed"
                    )

                example_responses = {
                    x["response"].strip()
                    for x in examples
                }

                result = {
                    "index": i,
                    "status": "ok",
                    "strategy": item.get("strategy"),
                    "reference": item.get("reference"),
                    "context": item["context"],
                    "response": response,
                    "pred_strategy": pred_strategy,
                    "reasoning": reasoning,
                    "raw_response": raw,
                    "retrieved_examples": [
                        {
                            "pool_index": x["pool_index"],
                            "source_dialog_idx":
                                x["source_dialog_idx"],
                            "strategy": x["strategy"],
                            "similarity":
                                round(x["similarity"], 6),
                        }
                        for x in examples
                    ],
                    "exact_example_copy":
                        response.strip()
                        in example_responses,
                    "method":
                        "Few-shot CoT retrieval reconstruction",
                    "model": "qwen2.5:32b",
                    "temperature": 0,
                    "reconstruction": {
                        "prompt":
                            "Appendix Prompt 11",
                        "retrieval_model":
                            "all-roberta-large-v1",
                        "top_k": 3,
                        "different_source_dialogues": True,
                        "exclude_dataset_first100_sources":
                            True,
                    },
                }

                break

            except Exception as e:
                last_error = repr(e)

                print(
                    f"[{i+1}/{end}] "
                    f"attempt {attempt}/3 failed: "
                    f"{last_error}"
                )

                time.sleep(3 * attempt)

        if result is None:
            result = {
                "index": i,
                "status": "error",
                "context": item["context"],
                "error": last_error,
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
            result.get("pred_strategy"),
            repr(result.get("response", ""))[:90],
        )

    # automatic sanity check
    rows = []

    with output.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x = json.loads(line)
                if x.get("status") == "ok":
                    rows.append(x)

    considered = [
        x for x in rows
        if x["index"] < end
    ]

    responses = [
        x["response"]
        for x in considered
    ]

    strategies = [
        x.get("pred_strategy")
        for x in considered
    ]

    copies = sum(
        bool(x.get("exact_example_copy"))
        for x in considered
    )

    print("\n=== SANITY CHECK ===")
    print("ok =", len(considered))
    print("unique responses =", len(set(responses)))
    print("unique strategies =", len(set(strategies)))
    print("exact example copies =", copies)

    if len(considered) >= 10:
        if len(set(responses)) <= 2:
            print("SANITY = FAIL: response collapse")
        elif len(set(strategies)) <= 1:
            print("SANITY = FAIL: strategy collapse")
        elif copies / len(considered) > 0.30:
            print("SANITY = FAIL: excessive example copying")
        else:
            print("SANITY = PASS")


if __name__ == "__main__":
    main()
