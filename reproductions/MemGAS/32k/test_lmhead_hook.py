import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"

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

prompt = """Below is an user-AI assistant dialogue memory. Please summarize the following dialogue as concisely as possible in a short paragraph, extracting the main themes and key information.

User: I have been learning piano recently.
Assistant: That's wonderful. What kind of music do you enjoy?
User: Mostly jazz and classical music.

Your answer:"""

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

# -------------------------
# Original generate()
# -------------------------
with torch.no_grad():
    normal = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

normal_new = normal[:, inputs["input_ids"].shape[1]:]


# -------------------------
# Low-memory hook:
# preserve model.generate(),
# only slice lm_head input
# -------------------------
def keep_last_hidden_only(module, args):
    hidden_states = args[0]
    return (hidden_states[:, -1:, :],)

handle = model.lm_head.register_forward_pre_hook(
    keep_last_hidden_only
)

with torch.no_grad():
    hooked = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

handle.remove()

hooked_new = hooked[:, inputs["input_ids"].shape[1]:]

print(
    "TOKEN_IDS_IDENTICAL:",
    torch.equal(normal_new, hooked_new)
)

print(
    "NORMAL:",
    tokenizer.decode(
        normal_new[0],
        skip_special_tokens=True,
    )
)

print(
    "HOOKED:",
    tokenizer.decode(
        hooked_new[0],
        skip_special_tokens=True,
    )
)

assert torch.equal(normal_new, hooked_new)

print("LMHEAD_HOOK_EQUIVALENCE_OK")
