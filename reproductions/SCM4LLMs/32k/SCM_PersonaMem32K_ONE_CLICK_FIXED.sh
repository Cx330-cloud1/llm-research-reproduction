#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/root/autodl-tmp/paper_repro/adapters/scm_personamem
PY=/root/autodl-tmp/envs/memgas/bin/python

cd "$WORKDIR"
mkdir -p repro_logs

cat > run_scm_controller_qwen.py <<'PY'
import json
from pathlib import Path
import torch
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
DATA = Path("personamem_scm_32k.jsonl")
UNIQUE = Path("unique_scm_memory_units.jsonl")
SUMMARIES = Path("scm_summaries_qwen7b.jsonl")
TOPK = Path("scm_top6_retrieval.jsonl")
OUT = Path("scm_controller_qwen7b.jsonl")
MAX_HISTORY_TOKENS = 2500
MAX_PRE_TURN_TOKENS = 500

try:
    enc = tiktoken.encoding_for_model("text-davinci-003")
except Exception:
    enc = tiktoken.get_encoding("p50k_base")

def toklen(text):
    return len(enc.encode(text))

JUDGE_PROMPT = """给定[对话内容]和[用户问题]，请回答指令问题。

[对话内容]:
```
{content}
```

[用户问题]:
```
{query}
```

指令问题:
```
根据[对话内容]是否能够回答[用户问题]？如果能请回答 `(A)是`，否则请回答 `(B)否`。
```

现在请开始回答，输出必须严格按照以下格式输出：

[答案]: 最终的答案是：(A)是 / (B)否
"""

questions = {}
with open(DATA, encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        questions[x["question_id"]] = x

unique = {}
with open(UNIQUE, encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        unique[x["unit_hash"]] = x

summaries = {}
with open(SUMMARIES, encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        summaries[x["unit_hash"]] = x["summary"]

retrieval = []
with open(TOPK, encoding="utf-8") as f:
    for line in f:
        retrieval.append(json.loads(line))

targets = [x for x in retrieval if x["needs_controller"]]
print("CONTROLLER_TARGETS:", len(targets))
assert len(targets) == 2

done = {}
if OUT.exists():
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                x = json.loads(line)
                done[x["question_id"]] = x

remaining = [x for x in targets if x["question_id"] not in done]
print("ALREADY_DONE:", len(done))
print("REMAINING:", len(remaining))

if not remaining:
    print("SCM_CONTROLLER_CACHE_OK")
    raise SystemExit(0)

tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "18GiB", "cpu": "24GiB"},
    low_cpu_mem_usage=True,
    local_files_only=True,
    trust_remote_code=True,
)
model.eval()
print("HF_DEVICE_MAP:", getattr(model, "hf_device_map", None))

def lm_head_pre_hook(module, args):
    hidden = args[0]
    if hidden.ndim == 3 and hidden.shape[1] > 1:
        hidden = hidden[:, -1:, :]
    return (hidden,)

hook = model.lm_head.register_forward_pre_hook(lm_head_pre_hook)

def generate(prompt):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inp = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.inference_mode():
        out = model.generate(
            **inp,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0, inp["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

def binary_answer(content, query):
    output = generate(JUDGE_PROMPT.format(content=content, query=query))
    return "(B)" not in output, output

def previous_len(previous_hash):
    raw = unique[previous_hash]["raw_text"]
    if toklen(raw) < MAX_PRE_TURN_TOKENS:
        return toklen(raw)
    return toklen(summaries[previous_hash])

def is_too_long(selected_hashes, flags, previous_hash):
    total = 0
    for h, flag in zip(selected_hashes, flags):
        if flag == "drop":
            continue
        if flag == "summary":
            total += toklen(summaries[h])
        else:
            total += toklen(unique[h]["raw_text"])
    total += previous_len(previous_hash)
    return total > MAX_HISTORY_TOKENS, total

with open(OUT, "a", encoding="utf-8") as fout:
    for n, r in enumerate(remaining, 1):
        qid = r["question_id"]
        query = questions[qid]["question"]
        hashes = r["controller_top_hashes"]
        previous_hash = r["previous_hash"]
        flags = ["raw"] * len(hashes)
        decisions = []
        too_long, total = is_too_long(hashes, flags, previous_hash)
        assert too_long
        first_round = True
        rounds = 0

        while too_long:
            rounds += 1
            for i, h in enumerate(hashes):
                if first_round:
                    if i < len(hashes) - 1:
                        raw_ok, raw_output = binary_answer(unique[h]["raw_text"], query)
                        if not raw_ok:
                            choice = "drop"
                            decisions.append({"round": rounds, "position": i, "unit_hash": h, "stage": "raw", "model_output": raw_output, "choice": choice})
                        else:
                            summ_ok, summ_output = binary_answer(summaries[h], query)
                            choice = "summary" if summ_ok else "raw"
                            decisions.append({"round": rounds, "position": i, "unit_hash": h, "stage": "raw_then_summary", "raw_model_output": raw_output, "summary_model_output": summ_output, "choice": choice})
                    else:
                        choice = "raw"
                        decisions.append({"round": rounds, "position": i, "unit_hash": h, "stage": "forced_highest_similarity", "choice": choice})
                else:
                    choice = flags[i]
                    if choice == "raw":
                        choice = "summary"
                    decisions.append({"round": rounds, "position": i, "unit_hash": h, "stage": "subsequent_round", "choice": choice})

                flags[i] = choice
                if not first_round:
                    too_long, total = is_too_long(hashes, flags, previous_hash)
                    if not too_long:
                        break

            too_long, total = is_too_long(hashes, flags, previous_hash)
            if first_round:
                first_round = False
            if rounds > 10:
                raise RuntimeError(f"controller did not converge: {qid}")

        flag_by_hash = dict(zip(hashes, flags))
        chronological_hashes = r["chronological_top_hashes"]
        pieces = []
        for h in chronological_hashes:
            flag = flag_by_hash[h]
            if flag == "drop":
                continue
            pieces.append(summaries[h] if flag == "summary" else unique[h]["raw_text"])

        retrieve_history_text = "\n\n".join(pieces)
        if retrieve_history_text:
            retrieve_history_text += "\n\n"

        record = {
            "question_id": qid,
            "flags_controller_order": flags,
            "controller_hashes": hashes,
            "chronological_hashes": chronological_hashes,
            "final_budget_tokens": total,
            "rounds": rounds,
            "decisions": decisions,
            "retrieve_history_text": retrieve_history_text,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[CONTROLLER] {n}/{len(remaining)} qid={qid} rounds={rounds} flags={flags} tokens={total}", flush=True)

hook.remove()
final = {}
with open(OUT, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            x = json.loads(line)
            final[x["question_id"]] = x
missing = {x["question_id"] for x in targets} - set(final)
print("FINAL_CONTROLLER_RECORDS:", len(final))
print("MISSING:", len(missing))
assert len(final) == 2
assert not missing
print("SCM_CONTROLLER_CACHE_OK")
PY

cat > run_scm_personamem_qa.py <<'PY'
import ast
import json
import re
from collections import Counter
from pathlib import Path
import torch
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
DATA = Path("personamem_scm_32k.jsonl")
UNIQUE = Path("unique_scm_memory_units.jsonl")
SUMMARIES = Path("scm_summaries_qwen7b.jsonl")
TOPK = Path("scm_top6_retrieval.jsonl")
CONTROLLER = Path("scm_controller_qwen7b.jsonl")
OUT = Path("personamem_scm_answers_qwen7b.jsonl")
METRICS = Path("personamem_scm_metrics_qwen7b.json")
MAX_PRE_TURN_TOKENS = 500

try:
    enc = tiktoken.encoding_for_model("text-davinci-003")
except Exception:
    enc = tiktoken.get_encoding("p50k_base")

def toklen(text):
    return len(enc.encode(text))

EN_HISTORY_AGENT_PROMPT = """The following is a conversation between a user and an AI assistant. Please answer the current question based on the history of the conversation:

Related conversation history:

{history_turn_text}

Previous conversation:

{pre_turn_text}

###

User: {input}

Assistant: """

EN_NO_HISTORY_AGENT_PROMPT = """The following is a conversation between a user and an AI assistant. Please answer the current question based on the history of the conversation:

Previous conversation:

{pre_turn_text}

###

User: {input}

Assistant: """

questions = []
with open(DATA, encoding="utf-8") as f:
    for line in f:
        questions.append(json.loads(line))
assert len(questions) == 589

unique = {}
with open(UNIQUE, encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        unique[x["unit_hash"]] = x

summaries = {}
with open(SUMMARIES, encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        summaries[x["unit_hash"]] = x["summary"]

retrieval = {}
with open(TOPK, encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        retrieval[x["question_id"]] = x
assert len(retrieval) == 589

controller = {}
with open(CONTROLLER, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            x = json.loads(line)
            controller[x["question_id"]] = x

controller_targets = [qid for qid, r in retrieval.items() if r["needs_controller"]]
assert len(controller_targets) == 2
missing_controller = set(controller_targets) - set(controller)
assert not missing_controller, f"Missing controller cache: {missing_controller}"

def parse_options(value):
    if isinstance(value, list):
        return value
    try:
        x = json.loads(value)
        if isinstance(x, list):
            return x
    except Exception:
        pass
    try:
        x = ast.literal_eval(value)
        if isinstance(x, list):
            return x
    except Exception:
        pass
    raise ValueError(f"Cannot parse all_options: {value[:200]}")

def build_mcq_input(q):
    options = parse_options(q["all_options"])
    option_text = "\n".join(str(x) for x in options)
    return (
        f"{q['question']}\n\n"
        "Choose the single best answer from the options below.\n\n"
        f"{option_text}\n\n"
        "Your final answer must contain exactly one option label from (a), (b), (c), or (d).\n"
        "End your response using exactly this format:\n"
        "<final_answer>(a)</final_answer>\n"
        "Replace (a) with the correct option label."
    )

def get_previous_text(previous_hash):
    raw = unique[previous_hash]["raw_text"]
    if toklen(raw) < MAX_PRE_TURN_TOKENS:
        return raw, "raw"
    return summaries[previous_hash], "summary"

def get_retrieved_history(qid):
    r = retrieval[qid]
    if r["needs_controller"]:
        return controller[qid]["retrieve_history_text"], "controller"
    pieces = [unique[h]["raw_text"] for h in r["chronological_top_hashes"]]
    text = "\n\n".join(pieces)
    if text:
        text += "\n\n"
    return text, "raw_top6"

def build_final_prompt(q):
    qid = q["question_id"]
    r = retrieval[qid]
    previous_text, previous_mode = get_previous_text(r["previous_hash"])
    retrieved_text, retrieval_mode = get_retrieved_history(qid)
    mcq_input = build_mcq_input(q)
    if retrieved_text:
        prompt = EN_HISTORY_AGENT_PROMPT.format(
            history_turn_text=retrieved_text,
            pre_turn_text=previous_text,
            input=mcq_input,
        )
    else:
        prompt = EN_NO_HISTORY_AGENT_PROMPT.format(
            pre_turn_text=previous_text,
            input=mcq_input,
        )
    return prompt, {
        "retrieval_mode": retrieval_mode,
        "previous_mode": previous_mode,
        "previous_hash": r["previous_hash"],
    }

OPTION_RE = re.compile(r"\(([abcdABCD])\)")

def normalize_option(x):
    if not x:
        return None
    x = x.lower()
    return f"({x})" if x in {"a", "b", "c", "d"} else None

def parse_answer(text):
    if text is None:
        return None, [], "empty"
    lower = text.lower()
    marker = "<final_answer>"
    region = text
    source = "full_response"
    pos = lower.rfind(marker)
    if pos >= 0:
        region = text[pos + len(marker):]
        end = region.lower().find("</final_answer>")
        if end >= 0:
            region = region[:end]
        source = "final_answer_region"
    found = [normalize_option(m.group(1)) for m in OPTION_RE.finditer(region)]
    found = [x for x in found if x is not None]
    unique_found = list(dict.fromkeys(found))
    if len(unique_found) == 1:
        return unique_found[0], unique_found, source
    if source != "full_response":
        found_all = [normalize_option(m.group(1)) for m in OPTION_RE.finditer(text)]
        found_all = [x for x in found_all if x is not None]
        unique_all = list(dict.fromkeys(found_all))
        if len(unique_all) == 1:
            return unique_all[0], unique_all, "whole_response_fallback"
        return None, unique_all, "ambiguous"
    return None, unique_found, "ambiguous"

done = {}
if OUT.exists():
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                x = json.loads(line)
                done[x["question_id"]] = x

remaining = [q for q in questions if q["question_id"] not in done]
print("QUESTIONS:", len(questions))
print("ALREADY_DONE:", len(done))
print("REMAINING:", len(remaining))

tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)

qwen_prompt_lengths = []
for q in questions:
    prompt, _ = build_final_prompt(q)
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer(chat_text, add_special_tokens=False)["input_ids"]
    qwen_prompt_lengths.append(len(ids))

vals = sorted(qwen_prompt_lengths)
def percentile(p):
    return vals[int(round((len(vals) - 1) * p))]

print("PROMPT_TOKENS:")
print("  min:", vals[0])
print("  median:", percentile(0.50))
print("  p95:", percentile(0.95))
print("  p99:", percentile(0.99))
print("  max:", vals[-1])

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "18GiB", "cpu": "24GiB"},
    low_cpu_mem_usage=True,
    local_files_only=True,
    trust_remote_code=True,
)
model.eval()
print("HF_DEVICE_MAP:", getattr(model, "hf_device_map", None))

def lm_head_pre_hook(module, args):
    hidden = args[0]
    if hidden.ndim == 3 and hidden.shape[1] > 1:
        hidden = hidden[:, -1:, :]
    return (hidden,)

hook = model.lm_head.register_forward_pre_hook(lm_head_pre_hook)

def generate(prompt):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inp = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.inference_mode():
        out = model.generate(
            **inp,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0, inp["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

with open(OUT, "a", encoding="utf-8") as fout:
    for i, q in enumerate(remaining, 1):
        qid = q["question_id"]
        prompt, memory_meta = build_final_prompt(q)
        response = generate(prompt)
        parsed, options_found, parser_source = parse_answer(response)
        gold = q["correct_answer"].strip().lower()
        correct = parsed == gold
        record = {
            "question_id": qid,
            "question_type": q["question_type"],
            "question": q["question"],
            "correct_answer": gold,
            "parsed_answer": parsed,
            "correct": correct,
            "options_found": options_found,
            "parser_source": parser_source,
            "response": response,
            "memory_meta": memory_meta,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[ANSWER] {i}/{len(remaining)} qid={qid} gold={gold} pred={parsed} correct={correct} found={options_found}", flush=True)

hook.remove()
print("PERSONAMEM_SCM_ANSWER_DONE")

records = []
with open(OUT, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
assert len(records) == 589

correct_total = sum(1 for x in records if x["correct"])
unparsed = sum(1 for x in records if x["parsed_answer"] is None)
type_total = Counter()
type_correct = Counter()
type_unparsed = Counter()
for x in records:
    t = x["question_type"]
    type_total[t] += 1
    if x["correct"]:
        type_correct[t] += 1
    if x["parsed_answer"] is None:
        type_unparsed[t] += 1

overall_acc = correct_total / len(records)
per_type = {}
for t in sorted(type_total):
    total = type_total[t]
    correct = type_correct[t]
    per_type[t] = {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "unparsed": type_unparsed[t],
    }

metrics = {
    "method": "SCM",
    "dataset": "PersonaMem-32K",
    "setting": "compatibility",
    "text_llm": "Qwen2.5-7B-Instruct",
    "embedding": "Contriever",
    "similar_top_k": 6,
    "max_history_tokens": 2500,
    "max_previous_turn_tokens": 500,
    "questions": len(records),
    "correct": correct_total,
    "overall_accuracy": overall_acc,
    "unparsed": unparsed,
    "controller_triggered": len(controller_targets),
    "per_question_type": per_type,
}

with open(METRICS, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print()
print("===== FINAL METRICS =====")
print(f"overall: {correct_total}/{len(records)} = {overall_acc * 100:.2f}%")
print("unparsed:", unparsed)
for t in sorted(type_total):
    correct = type_correct[t]
    total = type_total[t]
    print(f"{t}: {correct}/{total} = {correct / total * 100:.2f}%")
print("metrics saved:", METRICS.resolve())
print("PERSONAMEM_SCM_QA_OK")
PY

export OMP_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo '===== VERIFY INPUTS ====='
for f in personamem_scm_32k.jsonl unique_scm_memory_units.jsonl scm_summaries_qwen7b.jsonl scm_top6_retrieval.jsonl; do
  test -s "$f" || { echo "MISSING $f"; exit 1; }
done

echo '===== COMPILE ====='
"$PY" -m py_compile run_scm_controller_qwen.py
"$PY" -m py_compile run_scm_personamem_qa.py
echo 'PY_COMPILE_OK'

echo '===== CONTROLLER ====='
"$PY" run_scm_controller_qwen.py

echo '===== FINAL QA ====='
"$PY" run_scm_personamem_qa.py
