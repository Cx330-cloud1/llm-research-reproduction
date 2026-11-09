import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE = Path("/root/autodl-tmp/paper_repro/adapters/memgas_personamem_1m")

MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"

RETRIEVAL_PATH = BASE / "personamem_memgas_retrieval_1m_smoke.jsonl"
MEMORY_GEN_PATH = BASE / "session_generations_qwen7b.jsonl"

FILTER_CACHE = BASE / "personamem_memgas_filter_qwen7b_smoke.jsonl"
ANSWER_CACHE = BASE / "personamem_memgas_answers_qwen7b_smoke.jsonl"
METRICS_PATH = BASE / "personamem_memgas_metrics_qwen7b_smoke.json"


# ------------------------------------------------------------
# Official MemGAS filtering prompt
# Compatibility adaptation:
# PersonaMem has no real session/question dates, so date fields
# are omitted rather than fabricated.
# ------------------------------------------------------------

PROMPT_MULTIGRAN = """
You are an intelligent dialog bot. You will be shown History Dialogs and corresponding multi-granular information.
Filter the History Dialogs, summaries, and keywords to extract only the parts directly relevant to the Question. Preserve original tokens, do not paraphrase. Remove irrelevant turns, redundant info, and non-essential details.

History Dialogs: {retrieved_texts}

Question: {question}
Answer:
"""


# ------------------------------------------------------------
# MemGAS final-generation framing +
# PersonaMem-v1 official MCQ instruction/interface.
# ------------------------------------------------------------

PERSONAMEM_INSTRUCTION = (
    "Find the most appropriate model response and give your final "
    "answer (a), (b), (c), or (d) after the special token "
    "<final_answer>."
)

PROMPT_FINAL = """
You are an intelligent dialog bot. You will be shown History Dialogs. Please read, memorize, and understand the given Dialogs, then generate one concise, coherent and helpful response for the Question.

History Dialogs: {filtered_text}

Question: {question}

{instruction}

{all_options}
"""


def load_jsonl(path):
    rows = []

    if not path.exists():
        return rows

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


print("===== LOAD DATA =====", flush=True)

retrieval_rows = load_jsonl(RETRIEVAL_PATH)
memory_rows = load_jsonl(MEMORY_GEN_PATH)

assert len(retrieval_rows) == 3
assert len(memory_rows) == 2090

memory_by_hash = {}

for x in memory_rows:
    h = x["hash"]
    assert h not in memory_by_hash
    memory_by_hash[h] = x

print("retrieval questions:", len(retrieval_rows), flush=True)
print("memory generations:", len(memory_by_hash), flush=True)


# ------------------------------------------------------------
# Load tokenizer/model
# ------------------------------------------------------------

print("\n===== LOAD QWEN =====", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={
        0: "18GiB",
        "cpu": "24GiB",
    },
    low_cpu_mem_usage=True,
    local_files_only=True,
)

model.eval()

print("hf_device_map:", model.hf_device_map, flush=True)
print(
    "GPU_ALLOCATED_GB:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    flush=True,
)


# ------------------------------------------------------------
# Previously verified memory optimization:
# HuggingFace native generate() remains unchanged;
# only the unnecessary historical lm_head positions are removed.
#
# This optimization was separately verified:
# TOKEN_IDS_IDENTICAL=True.
# ------------------------------------------------------------

def _keep_last_hidden_only(module, args):
    hidden_states = args[0]
    return (hidden_states[:, -1:, :],)


_lm_head_hook = model.lm_head.register_forward_pre_hook(
    _keep_last_hidden_only
)


@torch.inference_mode()
def generate(prompt, max_new_tokens):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Memory sections are budgeted before prompt assembly.
    # Never silently truncate task instructions or the question.
    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    # Entire model is currently on cuda:0.
    device = next(model.parameters()).device

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    input_len = inputs["input_ids"].shape[1]

    if input_len > 4096:
        raise ValueError(
            f"Prompt exceeds 4096 tokens: {input_len}. "
            "Reduce memory-section budgets before generation."
        )

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    generated_ids = output[0, input_len:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    del inputs
    del output
    del generated_ids

    torch.cuda.empty_cache()

    return response, input_len


# ------------------------------------------------------------
def clip_text_tokens(text, max_tokens):
    text = str(text)
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if len(ids) <= max_tokens:
        return text

    head_n = max_tokens // 2
    tail_n = max_tokens - head_n

    head = tokenizer.decode(
        ids[:head_n],
        skip_special_tokens=True,
    )
    tail = tokenizer.decode(
        ids[-tail_n:],
        skip_special_tokens=True,
    )

    return (
        head
        + "\n[...middle omitted for context budget...]\n"
        + tail
    )


# PersonaMem-v1 official-style answer extraction
# ------------------------------------------------------------

def _extract_only_options(text):
    text = text.lower()

    in_parens = re.findall(
        r"\(([a-d])\)",
        text,
    )

    if in_parens:
        return set(in_parens)

    return set(
        re.findall(
            r"\b([a-d])\b",
            text,
        )
    )


def extract_answer(predicted_answer, correct_answer):
    """
    Mirrors PersonaMem v1 Evaluation.extract_answer().
    """

    correct = (
        str(correct_answer)
        .lower()
        .strip("() ")
    )

    full_response = predicted_answer or ""

    predicted = full_response.strip()

    if "<final_answer>" in predicted:
        predicted = (
            predicted
            .split("<final_answer>")[-1]
            .strip()
        )

    if predicted.endswith("</final_answer>"):
        predicted = predicted[
            :-len("</final_answer>")
        ].strip()

    pred_options = _extract_only_options(
        predicted
    )

    if pred_options == {correct}:
        return True, predicted, sorted(pred_options)

    # PersonaMem official fallback:
    # inspect whole response if needed.
    response_options = _extract_only_options(
        full_response
    )

    if response_options == {correct}:
        return True, predicted, sorted(response_options)

    return False, predicted, sorted(pred_options)


# ------------------------------------------------------------
# Stage 1: MemGAS top-3 multigranularity filtering
# ------------------------------------------------------------

existing_filter = {
    str(x["question_id"]): x
    for x in load_jsonl(FILTER_CACHE)
}

print(
    "\nfilter already done:",
    len(existing_filter),
    flush=True,
)


memory_clip_cache = {}

with open(
    FILTER_CACHE,
    "a",
    encoding="utf-8",
) as out_f:

    for idx, sample in enumerate(
        retrieval_rows,
        1,
    ):
        qid = str(sample["question_id"])

        if qid in existing_filter:
            continue

        retrieved_texts = ""

        top3 = sample[
            "retrieval_results"
        ]["ranked_items"][:3]

        assert len(top3) == 3

        for item in top3:
            h = item["session_hash"]

            assert h in memory_by_hash

            memory = memory_by_hash[h]

            if h not in memory_clip_cache:
                memory_clip_cache[h] = (
                    clip_text_tokens(
                        str(item["session"]), 650
                    ),
                    clip_text_tokens(
                        memory["summary"], 150
                    ),
                    clip_text_tokens(
                        memory["keywords"], 60
                    ),
                )

            (
                session_text,
                summary_text,
                keyword_text,
            ) = memory_clip_cache[h]

            retrieved_texts += (
                "\nSession Content:\n"
                f"{session_text}\n\n"
                "Session Summary:\n"
                f"{summary_text}\n"
                "Session Keyword:\n"
                f"{keyword_text}\n"
            )

        prompt = PROMPT_MULTIGRAN.format(
            retrieved_texts=retrieved_texts,
            question=sample["question"],
        )

        response, prompt_tokens = generate(
            prompt,
            max_new_tokens=512,
        )

        record = {
            "question_id": qid,
            "question_type": sample["question_type"],
            "prompt_tokens": prompt_tokens,
            "response": response,
        }

        out_f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )
        out_f.flush()

        existing_filter[qid] = record

        print(
            f"[FILTER {idx}/3]"
            f" qid={qid}"
            f" prompt_tokens={prompt_tokens}"
            f" output_chars={len(response)}",
            flush=True,
        )


assert len(existing_filter) == 3

print(
    "PERSONAMEM_MEMGAS_FILTER_DONE",
    flush=True,
)


# ------------------------------------------------------------
# Stage 2: PersonaMem-compatible MCQ answer generation
#
# Compatibility fix:
# Official generation_multigran.py accidentally repeats the
# SAME filtered response once for every top-k session.
# We include it exactly once.
# ------------------------------------------------------------

existing_answers = {
    str(x["question_id"]): x
    for x in load_jsonl(ANSWER_CACHE)
}

print(
    "\nanswers already done:",
    len(existing_answers),
    flush=True,
)


with open(
    ANSWER_CACHE,
    "a",
    encoding="utf-8",
) as out_f:

    for idx, sample in enumerate(
        retrieval_rows,
        1,
    ):
        qid = str(sample["question_id"])

        if qid in existing_answers:
            continue

        filtered_text = existing_filter[qid][
            "response"
        ]

        prompt = PROMPT_FINAL.format(
            filtered_text=filtered_text,
            question=sample["question"],
            instruction=PERSONAMEM_INSTRUCTION,
            all_options=sample["all_options"],
        )

        response, prompt_tokens = generate(
            prompt,
            max_new_tokens=512,
        )

        correct, parsed, options_found = (
            extract_answer(
                response,
                sample["correct_answer"],
            )
        )

        record = {
            "question_id": qid,
            "persona_id": sample["persona_id"],
            "question_type": sample["question_type"],
            "question": sample["question"],
            "all_options": sample["all_options"],
            "correct_answer": sample["correct_answer"],

            "prompt_tokens": prompt_tokens,
            "response": response,

            "parsed_answer": parsed,
            "options_found": options_found,
            "score": bool(correct),
        }

        out_f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )
        out_f.flush()

        existing_answers[qid] = record

        print(
            f"[ANSWER {idx}/3]"
            f" qid={qid}"
            f" score={int(correct)}"
            f" found={options_found}",
            flush=True,
        )


assert len(existing_answers) == 3

print(
    "PERSONAMEM_MEMGAS_ANSWER_DONE",
    flush=True,
)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

answers = [
    existing_answers[
        str(x["question_id"])
    ]
    for x in retrieval_rows
]

total = len(answers)
correct = sum(
    int(x["score"])
    for x in answers
)

overall_acc = correct / total


by_type = defaultdict(
    lambda: {
        "total": 0,
        "correct": 0,
    }
)

unparsed = 0

for x in answers:
    qt = x["question_type"]

    by_type[qt]["total"] += 1
    by_type[qt]["correct"] += int(
        x["score"]
    )

    if not x["options_found"]:
        unparsed += 1


per_type = {}

for qt in sorted(by_type):
    n = by_type[qt]["total"]
    c = by_type[qt]["correct"]

    per_type[qt] = {
        "total": n,
        "correct": c,
        "accuracy": c / n,
    }


metrics = {
    "benchmark": "PersonaMem-1M",

    "method": "MemGAS",

    "retrieval_parameters": {
        "topk_sessions": 3,
        "num_seednodes": 15,
        "mem_threshold": 30,
        "n_components": 2,
        "damping": 0.1,
        "temp": 0.2,
    },

    "generator": {
        "model": "Qwen2.5-7B-Instruct",
        "filter_max_new_tokens": 4000,
        "answer_max_new_tokens": 512,
        "do_sample": False,
    },

    "compatibility_notes": [
        "gpt-4o-mini replaced with Qwen2.5-7B-Instruct",
        "PersonaMem has no real session/question date; unavailable date fields omitted",
        "official generation_multigran.py duplicates the same filtered response topk times in stage 2; this benchmark includes the filtered response once",
        "PersonaMem-v1 MCQ answer interface and extraction logic used for final ACC",
    ],

    "total": total,
    "correct": correct,
    "overall_accuracy": overall_acc,
    "unparsed": unparsed,

    "per_question_type": per_type,
}


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metrics,
        f,
        ensure_ascii=False,
        indent=2,
    )


print("\n===== FINAL METRICS =====", flush=True)

print(
    f"overall: {correct}/{total}"
    f" = {overall_acc * 100:.2f}%",
    flush=True,
)

print(
    "unparsed:",
    unparsed,
    flush=True,
)

for qt, x in per_type.items():
    print(
        f"{qt}:"
        f" {x['correct']}/{x['total']}"
        f" = {x['accuracy'] * 100:.2f}%",
        flush=True,
    )

print(
    "metrics saved:",
    METRICS_PATH,
    flush=True,
)

print(
    "PERSONAMEM_MEMGAS_QA_OK",
    flush=True,
)
