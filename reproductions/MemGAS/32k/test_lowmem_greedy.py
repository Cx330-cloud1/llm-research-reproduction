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


def lowmem_greedy(input_ids, attention_mask, max_new_tokens=64):
    generated = []
    past_key_values = None
    cur_input_ids = input_ids

    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = {eos_ids}
    else:
        eos_ids = set(eos_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model.model(
                input_ids=cur_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            # CRITICAL:
            # only the LAST hidden state goes through the LM head.
            last_hidden = outputs.last_hidden_state[:, -1:, :]
            logits = model.lm_head(last_hidden).float()[:, -1, :]

            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True,
            )

            generated.append(next_token)

            if next_token.item() in eos_ids:
                break

            past_key_values = outputs.past_key_values
            cur_input_ids = next_token

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.size(0), 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )

    return torch.cat(generated, dim=1)


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

with torch.no_grad():
    normal = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

normal_new = normal[:, inputs["input_ids"].shape[1]:]

lowmem_new = lowmem_greedy(
    inputs["input_ids"],
    inputs["attention_mask"],
    max_new_tokens=64,
)

print(
    "TOKEN_IDS_IDENTICAL:",
    torch.equal(normal_new, lowmem_new)
)

print(
    "NORMAL:",
    tokenizer.decode(normal_new[0], skip_special_tokens=True)
)

print(
    "LOWMEM:",
    tokenizer.decode(lowmem_new[0], skip_special_tokens=True)
)

assert torch.equal(normal_new, lowmem_new)

print("LOWMEM_GREEDY_EQUIVALENCE_OK")
