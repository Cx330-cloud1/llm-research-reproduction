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
