import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path("/root/autodl-tmp/paper_repro/MemGAS")
MODEL_PATH = Path("/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
DATA_PATH = ROOT / "data/process_data/longmemeval_s.json"
INPUT_DIR = ROOT / "generation_logs/longmemeval_s_qwen7b_ctx4k"
OUTPUT_DIR = ROOT / "repro_logs/longmemeval_s_qwen_judge"

METHODS = [
    "session_level",
    "turn_level",
    "summary_level",
    "keyword_level",
    "hybrid_level",
    "memgas",
]

MODEL_NAME = "Qwen2.5-7B-Instruct"
PROTOCOL = "LongMemEval official task-specific judge prompts; local Qwen judge"


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(
                    f"Invalid JSON: {path}:{lineno}: {exc}"
                )
    return rows


def write_jsonl_line(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    f.flush()


def build_prompt(task, question, answer, response, abstention=False):
    # Prompt templates follow the official LongMemEval evaluate_qa.py.
    if abstention:
        template = """I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {}

Explanation: {}

Model Response: {}

Does the model correctly identify the question as unanswerable? Answer yes or no only."""
        return template.format(question, answer, response)

    if task in [
        "single-session-user",
        "single-session-assistant",
        "multi-session",
    ]:
        template = """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {}

Correct Answer: {}

Model Response: {}

Is the model response correct? Answer yes or no only."""
    elif task == "temporal-reasoning":
        template = """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days.
If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

Question: {}

Correct Answer: {}

Model Response: {}

Is the model response correct? Answer yes or no only."""
    elif task == "knowledge-update":
        template = """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {}

Correct Answer: {}

Model Response: {}

Is the model response correct? Answer yes or no only."""
    elif task == "single-session-preference":
        template = """I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: {}

Rubric: {}

Model Response: {}

Is the model response correct? Answer yes or no only."""
    else:
        raise ValueError(f"Unsupported question type: {task}")

    return template.format(question, answer, response)


def parse_label(raw):
    # Use the first yes/no verdict, ignoring capitalization and punctuation.
    match = re.search(r"\b(yes|no)\b", raw.strip().lower())
    if not match:
        return None
    return match.group(1) == "yes"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=METHODS,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum NEW judgments per method in this run; for smoke tests.",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Local model not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    refs = {}

    for conv in data:
        qid = conv["conversation_id"]
        if len(conv.get("qa", [])) != 1:
            raise ValueError(f"Expected one QA per record: {qid}")

        qa = conv["qa"][0]
        refs[qid] = {
            "question_id": qid,
            "question_type": qa["question_type"],
            "question": qa["question"],
            "answer": str(qa["answer"]),
            "question_date": qa.get("question_date", ""),
        }

    if len(refs) != 500:
        raise ValueError(f"Expected 500 unique references, got {len(refs)}")

    print("===== CONFIG =====")
    print("references:", len(refs))
    print("methods:", args.methods)
    print("model:", MODEL_PATH)
    print("judge:", MODEL_NAME)
    print("output:", OUTPUT_DIR)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    device = model.get_input_embeddings().weight.device
    print("model_device:", device)
    print("GPU loaded; starting evaluation")

    @torch.inference_mode()
    def judge(prompt):
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(
            chat,
            return_tensors="pt",
            truncation=False,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        context_limit = getattr(
            model.config, "max_position_embeddings", 32768
        )
        if encoded["input_ids"].shape[1] + 12 > context_limit:
            raise ValueError(
                "Judge prompt exceeds model context; refusing to truncate."
            )

        input_len = encoded["input_ids"].shape[1]
        output = model.generate(
            **encoded,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        raw = tokenizer.decode(
            output[0, input_len:],
            skip_special_tokens=True,
        ).strip()

        del encoded, output
        return raw, parse_label(raw)

    for method in args.methods:
        input_path = INPUT_DIR / (
            f"longmemeval_s-contriever-{method}"
            "-filter-qwen2.5-7b-topk_3.jsonl"
        )
        output_path = OUTPUT_DIR / (
            f"longmemeval_s-{method}-qwen2.5-7b-judge.jsonl"
        )

        if not input_path.exists():
            raise FileNotFoundError(f"Missing QA output: {input_path}")

        source_rows = read_jsonl(input_path)
        if len(source_rows) != 500:
            raise ValueError(
                f"{method}: expected 500 source rows, got {len(source_rows)}"
            )

        source_ids = [r.get("question_id") for r in source_rows]
        if len(set(source_ids)) != 500 or set(source_ids) != set(refs):
            raise ValueError(f"{method}: question IDs do not match references")

        # Keep only valid prior judgments when resuming.
        existing = {}
        if output_path.exists():
            for row in read_jsonl(output_path):
                if (
                    row.get("question_id") in refs
                    and isinstance(row.get("judge_label"), bool)
                ):
                    existing[row["question_id"]] = row

        # Compact the checkpoint: remove invalid or duplicate rows.
        with output_path.open("w", encoding="utf-8") as f:
            for row in existing.values():
                write_jsonl_line(f, row)

        new_count = 0
        print(f"\n===== {method} =====")
        print(f"already_valid={len(existing)}/500")

        with output_path.open("a", encoding="utf-8") as out:
            pending = [
                row for row in source_rows
                if row["question_id"] not in existing
            ]

            if args.limit is not None:
                pending = pending[:args.limit]

            for src in tqdm(pending, desc=f"Judge {method}", unit="qa"):
                qid = src["question_id"]
                ref = refs[qid]
                hypothesis = src.get("hypothesis")

                if not isinstance(hypothesis, str) or not hypothesis.strip():
                    raise ValueError(f"{method}: empty hypothesis for {qid}")

                prompt = build_prompt(
                    task=ref["question_type"],
                    question=ref["question"],
                    answer=ref["answer"],
                    response=hypothesis,
                    abstention="_abs" in qid,
                )

                raw, label = judge(prompt)

                # Retry once if the model did not return a parseable verdict.
                if label is None:
                    raw, label = judge(
                        prompt + "\n\nReturn exactly one word: yes or no."
                    )

                result = {
                    "question_id": qid,
                    "question_type": ref["question_type"],
                    "question": ref["question"],
                    "question_date": ref["question_date"],
                    "gold_answer": ref["answer"],
                    "hypothesis": hypothesis,
                    "method": method,
                    "topk": src.get("topk", 3),
                    "judge_model": MODEL_NAME,
                    "judge_protocol": PROTOCOL,
                    "judge_raw": raw,
                    "judge_label": label,
                    "abstention_question": "_abs" in qid,
                    "truncated_session_ids": src.get(
                        "truncated_session_ids", []
                    ),
                }

                write_jsonl_line(out, result)
                new_count += 1

                if label is None:
                    print(f"\n[INVALID VERDICT] {method} {qid}: {raw!r}")

        print(f"new judgments attempted: {new_count}")
        print(f"checkpoint: {output_path}")

    # Rebuild a consistent metrics summary from saved judgments.
    summaries = {}

    for method in METHODS:
        path = OUTPUT_DIR / (
            f"longmemeval_s-{method}-qwen2.5-7b-judge.jsonl"
        )
        if not path.exists():
            continue

        # Latest row per ID wins, then only parseable labels are scored.
        by_id = {}
        for row in read_jsonl(path):
            by_id[row["question_id"]] = row

        valid = [
            row for row in by_id.values()
            if isinstance(row.get("judge_label"), bool)
        ]

        by_type = defaultdict(list)
        for row in valid:
            by_type[row["question_type"]].append(
                int(row["judge_label"])
            )

        correct = sum(int(row["judge_label"]) for row in valid)
        summaries[method] = {
            "judge_model": MODEL_NAME,
            "protocol": PROTOCOL,
            "n_expected": 500,
            "n_judged": len(valid),
            "n_correct": correct,
            "accuracy": correct / len(valid) if valid else None,
            "complete": len(valid) == 500,
            "by_question_type": {
                qtype: {
                    "n": len(values),
                    "correct": sum(values),
                    "accuracy": sum(values) / len(values),
                }
                for qtype, values in sorted(by_type.items())
            },
        }

    summary_path = OUTPUT_DIR / "metrics_summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== CURRENT QA JUDGE METRICS =====")
    for method, summary in summaries.items():
        accuracy = summary["accuracy"]
        accuracy_text = (
            f"{accuracy * 100:.2f}%"
            if accuracy is not None else "N/A"
        )
        print(
            f"{method:16s} "
            f"{summary['n_judged']:3d}/500 "
            f"accuracy={accuracy_text} "
            f"complete={summary['complete']}"
        )

    print("summary:", summary_path)
    print("LONGMEMEVAL_QWEN_JUDGE_RUN_OK")


if __name__ == "__main__":
    main()
