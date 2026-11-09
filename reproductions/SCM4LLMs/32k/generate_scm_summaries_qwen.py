import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"

UNIQUE = Path("unique_scm_memory_units.jsonl")
OUT = Path("scm_summaries_qwen7b.jsonl")

PROMPT = """Below is a conversation between a user and an AI assistant. Please write a summary for each of them in one sentence and list them in separate paragraphs, while trying to preserve the key information of the user’s question and the assistant’s answer as much as possible.

conversation content:

{input}

Summary:
"""


def load_done():
    done = {}
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                x = json.loads(line)
                done[x["unit_hash"]] = x
    return done


units = []
with open(UNIQUE, encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        units.append(x)

done = load_done()

# First save all official-short-path units without any LLM call.
with open(OUT, "a", encoding="utf-8") as fout:
    for x in units:
        uid = x["unit_hash"]

        if uid in done:
            continue

        if not x["needs_summary"]:
            rec = {
                "unit_hash": uid,
                "summary": x["raw_text"],
                "source": "raw_under_300",
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done[uid] = rec

print(
    f"TOTAL={len(units)} "
    f"DONE={len(done)} "
    f"NEED_GENERATION={sum(1 for x in units if x['needs_summary'] and x['unit_hash'] not in done)}"
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
    trust_remote_code=True,
)

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


# ------------------------------------------------------------
# Memory-saving hook:
# keep native model.generate(), but lm_head only sees last hidden token.
# This was already equivalence-tested in the MemGAS reproduction.
# ------------------------------------------------------------
def lm_head_pre_hook(module, args):
    hidden = args[0]
    if hidden.ndim == 3 and hidden.shape[1] > 1:
        hidden = hidden[:, -1:, :]
    return (hidden,)

hook = model.lm_head.register_forward_pre_hook(lm_head_pre_hook)


def generate(prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output[0, inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
    ).strip()


remaining = [
    x for x in units
    if x["needs_summary"]
    and x["unit_hash"] not in done
]

with open(OUT, "a", encoding="utf-8") as fout:
    for i, x in enumerate(remaining, 1):
        uid = x["unit_hash"]

        prompt = PROMPT.format(
            input=x["raw_text"]
        )

        summary = generate(prompt)

        rec = {
            "unit_hash": uid,
            "summary": summary,
            "source": "qwen2.5-7b",
        }

        fout.write(
            json.dumps(rec, ensure_ascii=False)
            + "\n"
        )
        fout.flush()

        print(
            f"[SUMMARY] {i}/{len(remaining)} "
            f"hash={uid[:10]} "
            f"chars={len(summary)}",
            flush=True,
        )


hook.remove()

# final validation
final = load_done()

missing = {
    x["unit_hash"] for x in units
} - set(final)

print("FINAL_RECORDS:", len(final))
print("MISSING:", len(missing))

assert len(final) == len(units)
assert not missing

print("SCM_SUMMARY_CACHE_OK")
