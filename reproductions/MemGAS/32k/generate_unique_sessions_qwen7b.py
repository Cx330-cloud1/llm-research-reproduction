import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path("/root/autodl-tmp/paper_repro")
MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"

INPUT = ROOT / "adapters/memgas_personamem/unique_sessions.jsonl"
OUTPUT = ROOT / "adapters/memgas_personamem/session_generations_qwen7b.jsonl"

SUMMARY_PROMPT = (
    "Below is an user-AI assistant dialogue memory. "
    "Please summarize the following dialogue as concisely as possible "
    "in a short paragraph, extracting the main themes and key information.\n"
)

KEYWORD_PROMPT = (
    "Below is an user-AI assistant dialogue memory. "
    "Please extract the most relevant keywords, separated by semicolon.\n"
)

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

# Memory-only optimization.
# Hugging Face generate() only consumes the final-position logits
# during greedy decoding. Keeping only the final hidden position before
# lm_head avoids allocating [sequence_length, vocabulary_size] logits.
def _keep_last_hidden_only(module, args):
    hidden_states = args[0]
    return (hidden_states[:, -1:, :],)

_lm_head_hook = model.lm_head.register_forward_pre_hook(
    _keep_last_hidden_only
)


def generate(instruction, entry):
    prompt = f"{instruction}\n\n{entry}\n\nYour answer:"

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
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


sessions = []
with INPUT.open(encoding="utf-8") as f:
    for line in f:
        sessions.append(json.loads(line))

done = {}

if OUTPUT.exists():
    with OUTPUT.open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            done[x["hash"]] = x

print("unique_sessions:", len(sessions))
print("already_done:", len(done))

with OUTPUT.open("a", encoding="utf-8") as out:
    for i, x in enumerate(sessions, 1):
        h = x["hash"]

        if h in done:
            print(f"[{i}/{len(sessions)}] SKIP {h[:12]}")
            continue

        print(f"[{i}/{len(sessions)}] {h[:12]}")

        summary = generate(
            SUMMARY_PROMPT,
            x["text"],
        )

        keywords = generate(
            KEYWORD_PROMPT,
            x["text"],
        )

        row = {
            "hash": h,
            "summary": summary,
            "keywords": keywords,
        }

        out.write(
            json.dumps(row, ensure_ascii=False) + "\n"
        )
        out.flush()

print("PERSONAMEM_SESSION_GENERATION_DONE")
