import os
import re
import ast
import sys
import json
import shutil
import argparse
import tempfile
import importlib.util
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import SamplingParams

# ============================================================
# Fixed paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HELA_ROOT = Path(
    os.environ.get("HELA_ROOT", str(PROJECT_ROOT))
)

RF_ROOT = Path(
    os.environ.get(
        "PERSONAMEM_DATA_ROOT",
        str(PROJECT_ROOT / "data" / "personamem_data"),
    )
)

QUESTION_PATH = Path(
    os.environ.get(
        "PERSONAMEM_QUESTION_PATH",
        str(RF_ROOT / "data/questions_1M.csv"),
    )
)

CONTEXT_PATH = Path(
    os.environ.get(
        "PERSONAMEM_CONTEXT_PATH",
        str(RF_ROOT / "data/shared_contexts_1M.jsonl"),
    )
)

# ============================================================
# Unified local experiment settings
# ============================================================

os.environ["HEBBIAN_LOCAL_QWEN"] = "1"
os.environ.setdefault(
    "HEBBIAN_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# PersonaMem has no real timestamps.
# Do NOT fabricate temporal ranking signal.
os.environ["HEBBIAN_USE_TIME_DECAY"] = "false"

os.environ.setdefault("HEBBIAN_USE_KEYWORD_MATCH", "true")
os.environ.setdefault("HEBBIAN_LEARNING_RATE", "0.02")
os.environ.setdefault("HEBBIAN_DECAY_RATE", "0.995")
os.environ.setdefault("HEBBIAN_ACTIVATION_ALPHA", "0.1")
os.environ.setdefault("HEBBIAN_SPREADING_THRESHOLD", "0.4")
os.environ.setdefault("HEBBIAN_MAX_FLIPPED", "3")
os.environ.setdefault("HEBBIAN_KEYWORD_WEIGHT", "0.7")

# ============================================================
# HeLa imports
# ============================================================

sys.path.insert(0, str(HELA_ROOT))

from hela_mem.hebbian_memory import HebbianMemoryGraph
from hela_mem.hebbian_retriever import HebbianRetriever
from hela_mem.local_qwen_bridge import get_llm


# ============================================================
# Load RF-Mem PersonaMem utilities without copying dataset logic
# ============================================================

spec = importlib.util.spec_from_file_location(
    "rf_personamem_utils",
    RF_ROOT / "utils.py",
)
rf_utils = importlib.util.module_from_spec(spec)

# RF utils may import modules relative to its directory.
sys.path.insert(0, str(RF_ROOT))
spec.loader.exec_module(rf_utils)

load_rows_with_context = rf_utils.load_rows_with_context


# ============================================================
# Read RF-Mem's exact INSTRUCTION constant from source.
# Avoid importing main_batch.py because it has execution setup.
# ============================================================

def load_rf_instruction():
    source = (RF_ROOT / "main_batch.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
            )

            for target in targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "INSTRUCTION"
                ):
                    value = ast.literal_eval(node.value)
                    return value

    raise RuntimeError("Could not find INSTRUCTION in RF main_batch.py")


INSTRUCTION = load_rf_instruction()


# ============================================================
# Same answer parser as RF-Mem qwen3_8b.py
# ============================================================

def extract_answer(predicted_answer: str, correct_answer: str):

    def _extract_only_options(text: str) -> set[str]:
        text = text.lower()

        in_parens = re.findall(r"\(([a-d])\)", text)
        if in_parens:
            return set(in_parens)

        return set(re.findall(r"\b([a-d])\b", text))

    correct = correct_answer.lower().strip("() ")

    full_response = predicted_answer

    predicted_answer = predicted_answer.strip()

    if "<final_answer>" in predicted_answer:
        predicted_answer = (
            predicted_answer
            .split("<final_answer>")[-1]
            .strip()
        )

    if predicted_answer.endswith("</final_answer>"):
        predicted_answer = (
            predicted_answer[
                :-len("</final_answer>")
            ].strip()
        )

    pred_options = _extract_only_options(predicted_answer)

    if pred_options == {correct}:
        return True, predicted_answer

    response_options = _extract_only_options(full_response)

    if response_options == {correct}:
        return True, predicted_answer

    return False, predicted_answer


# ============================================================
# PersonaMem -> HeLa episodic turns
# ============================================================

def strip_prefix(text, prefix):
    text = str(text).strip()

    if text.lower().startswith(prefix.lower()):
        text = text[len(prefix):].lstrip()

    return text


def context_to_turns(context):
    """
    Match HeLa's official LongMemEval encoding semantics:

    - skip system
    - only complete user -> assistant pairs
    - ignore trailing unmatched user message
    """

    turns = []

    i = 0

    while i < len(context):

        msg = context[i]

        if msg.get("role") != "user":
            i += 1
            continue

        if (
            i + 1 < len(context)
            and context[i + 1].get("role") == "assistant"
        ):
            user_text = strip_prefix(
                msg.get("content", ""),
                "User:",
            )

            assistant_text = strip_prefix(
                context[i + 1].get("content", ""),
                "Assistant:",
            )

            turns.append(
                (user_text, assistant_text)
            )

            i += 2

        else:
            # trailing / unmatched user:
            # same principle as official parser
            i += 1

    return turns


# ============================================================
# HeLa retrieval context
# ============================================================

def format_retrieval(results):
    blocks = []

    for res in results:
        node = res["node"]
        score = res["score"]

        source_label = (
            "Direct Match"
            if res.get("base_score", 0) > 0.6
            else "Associative Memory"
        )

        blocks.append(
            f"[{source_label} | Relevancy: {score:.2f}]\n"
            f"Content: {node['content']}"
        )

    return blocks


# ============================================================
# Same Qwen prompt semantics as patched RF qwen3_8b.py:
#
# if history is List[str]:
#     prompt = "\n".join(history) + prompt
#     messages = [{"role": "user", "content": prompt}]
# ============================================================

def build_generation_text(llm, prompt, history):
    tokenizer = llm.get_tokenizer()

    combined_prompt = "\n".join(history) + prompt

    messages = [
        {
            "role": "user",
            "content": combined_prompt,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_answers(llm, prompts, histories):
    texts = [
        build_generation_text(llm, p, h)
        for p, h in zip(prompts, histories)
    ]

    params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        repetition_penalty=1.05,
        max_tokens=32,
        seed=42,
    )

    outputs = llm.generate(
        texts,
        params,
        use_tqdm=True,
    )

    return [
        x.outputs[0].text.strip()
        for x in outputs
    ]


# ============================================================
# Simple reporting using exact per-item binary score
# ============================================================

def report(df):

    total = len(df)
    correct = int(df["score"].sum())
    wrong = total - correct
    accuracy = correct / total if total else 0.0

    print("\n=== Overall Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Wrong: {wrong}")

    print("\n=== Topics Results ===")

    topic = (
        df.groupby("topic")["score"]
        .agg(["mean", "count", "sum"])
        .rename(
            columns={
                "mean": "Accuracy",
                "count": "Samples",
                "sum": "Correct",
            }
        )
        .sort_values(
            ["Accuracy", "Samples"],
            ascending=[False, False],
        )
    )

    print(topic)

    print("\n=== Question Types Results ===")

    qt = (
        df.groupby("question_type")["score"]
        .agg(["mean", "count", "sum"])
        .rename(
            columns={
                "mean": "Accuracy",
                "count": "Samples",
                "sum": "Correct",
            }
        )
        .sort_values(
            ["Accuracy", "Samples"],
            ascending=[False, False],
        )
    )

    print(qt)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_items",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--output",
        default="results_hela_personamem_sanity5.csv",
    )

    args = parser.parse_args()

    # One shared Qwen instance:
    # keyword extraction + final answers use this same engine.
    llm = get_llm()

    rows = []
    prompts = []
    histories = []

    temp_root = tempfile.mkdtemp(
        prefix="hela_personamem_"
    )

    try:
        iterator = load_rows_with_context(
            str(QUESTION_PATH),
            str(CONTEXT_PATH),
        )

        for idx, (row, context) in enumerate(iterator):

            if idx >= args.num_items:
                break

            qid = row["question_id"]
            question = row["user_question_or_message"]
            correct_answer = row["correct_answer"]
            all_options = row["all_options"]

            # Loader already performs:
            # context[:end_index_in_shared_context]
            end_idx = int(
                row["end_index_in_shared_context"]
            )

            assert len(context) == end_idx, (
                qid,
                len(context),
                end_idx,
            )

            turns = context_to_turns(context)

            mem_path = os.path.join(
                temp_root,
                f"{qid}_hebbian.json",
            )

            graph = HebbianMemoryGraph(
                file_path=mem_path,
            )

            retriever = HebbianRetriever(
                graph,
                profile_memory=None,
            )

            # PersonaMem has no real timestamps.
            # Since time decay is disabled, use a constant solely
            # to satisfy HeLa node schema.
            fixed_timestamp = "2026-01-01 00:00:00"

            print(
                f"\n[{idx + 1}/{args.num_items}] "
                f"{qid}: encoding {len(turns)} turns"
            )

            for user_text, assistant_text in turns:
                retriever.process_conversation_turn(
                    user_text,
                    assistant_text,
                    timestamp=fixed_timestamp,
                )

            results = graph.retrieve(
                question,
                top_k=args.top_k,
            )

            history = format_retrieval(results)

            prompt = (
                question
                + "\n\n"
                + INSTRUCTION
                + "\n\n"
                + all_options
            )

            histories.append(history)
            prompts.append(prompt)

            rows.append(
                {
                    "score": False,
                    "persona_id": row["persona_id"],
                    "question_id": qid,
                    "question": question,
                    "question_type": row["question_type"],
                    "topic": row["topic"],
                    "correct_answer": correct_answer,
                    "model_response": "",
                    "predicted_answer": "",
                    "num_turns": len(turns),
                    "retrieved_count": len(results),
                    "end_index_in_shared_context": end_idx,
                }
            )

        print(
            "\nEncoding/retrieval finished:",
            len(rows),
            "samples",
        )

        responses = generate_answers(
            llm,
            prompts,
            histories,
        )

        for i, response in enumerate(responses):

            score, predicted = extract_answer(
                response,
                rows[i]["correct_answer"],
            )

            rows[i]["score"] = bool(score)
            rows[i]["model_response"] = response
            rows[i]["predicted_answer"] = predicted

            print("\n------------------------------")
            print("Q:", rows[i]["question"])
            print("Response:", response)
            print("Predicted:", predicted)
            print("Correct:", rows[i]["correct_answer"])
            print("Score:", score)

        df = pd.DataFrame(rows)

        df.to_csv(
            args.output,
            index=False,
            encoding="utf-8-sig",
        )

        json_path = str(
            Path(args.output).with_suffix(".json")
        )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                rows,
                f,
                ensure_ascii=False,
                indent=2,
            )

        report(df)

        print("\nSaved:")
        print(args.output)
        print(json_path)

    finally:
        shutil.rmtree(
            temp_root,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
