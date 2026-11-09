import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
DATA_PATH = Path("data/process_data/locomo10.json")
OUT_DIR = Path("multi_granularity_logs")
OUT_DIR.mkdir(exist_ok=True)

SUMMARY_PATH = OUT_DIR / "locomo10-summary_level.jsonl"
KEYWORD_PATH = OUT_DIR / "locomo10-keyword_level.jsonl"

SUMMARY_PROMPT = (
    "Below is an user-user dialogue memory. "
    "Please summarize the following dialogue as concisely as possible "
    "in a short paragraph, extracting the main themes and key information.\n"
)

KEYWORD_PROMPT = (
    "Below is an user-user dialogue memory. "
    "Please extract the most relevant keywords, separated by semicolon.\n"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    local_files_only=True,
)
model.eval()


def load_done(path):
    done = {}
    if not path.exists():
        return done

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.update(json.loads(line))
    return done


def generate(instruction, entry):
    # Exactly preserves the MemGAS user prompt text.
    prompt = f"{instruction}\n\n{entry}\n\nYour answer:"

    messages = [{"role": "user", "content": prompt}]

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

    new_tokens = output[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()


data = json.load(open(DATA_PATH, encoding="utf-8"))

sessions = []

for conv in data:
    conv_id = conv["conversation_id"]

    for sessid, sess in zip(
        conv["sessions_ids"],
        conv["sessions"],
    ):
        sid = f"convid-{str(conv_id)}-sessid-{sessid}"

        sessions.append(
            (sid, "\n\n".join(sess))
        )

print("TOTAL_SESSIONS:", len(sessions))


def run_level(path, instruction, level):
    done = load_done(path)

    print(f"{level}: existing={len(done)}")

    with path.open("a", encoding="utf-8") as f:
        for idx, (sid, entry) in enumerate(sessions, 1):

            if sid in done:
                print(f"[{idx}/{len(sessions)}] SKIP {sid}")
                continue

            print(f"[{idx}/{len(sessions)}] {level} {sid}")

            result = generate(
                instruction,
                entry,
            )

            f.write(
                json.dumps(
                    {sid: result},
                    ensure_ascii=False,
                ) + "\n"
            )

            # Immediate persistence for safe resume.
            f.flush()

            print(result)


run_level(
    SUMMARY_PATH,
    SUMMARY_PROMPT,
    "summary",
)

run_level(
    KEYWORD_PATH,
    KEYWORD_PROMPT,
    "keyword",
)

print("QWEN7B_FULL_GENERATION_DONE")
