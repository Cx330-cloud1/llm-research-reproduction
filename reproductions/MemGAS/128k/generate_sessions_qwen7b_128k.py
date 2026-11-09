import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc
import json
import argparse
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)

MODEL_PATH = Path("/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
INPUT = Path("unique_sessions.jsonl")
OUTPUT = Path("session_generations_qwen7b.jsonl")

DEFAULT_LIMIT = 32768
YARN_LIMIT = 131072
MAX_NEW_TOKENS = 500

SUMMARY_PROMPT = (
    "Below is an user-AI assistant dialogue memory. "
    "Please summarize the following dialogue as concisely as possible "
    "in a short paragraph, extracting the main themes and key information.\n"
)

KEYWORD_PROMPT = (
    "Below is an user-AI assistant dialogue memory. "
    "Please extract the most relevant keywords, separated by semicolon.\n"
)


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)


def session_text(x):
    s = x["session"]
    assert isinstance(s, list)
    assert s
    assert all(isinstance(v, str) for v in s)

    # Same representation used by completed 32K Qwen generation.
    return "\n\n".join(s)


def build_chat(instruction, entry):
    prompt = f"{instruction}\n\n{entry}\n\nYour answer:"

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def count_prompt(instruction, entry):
    text = build_chat(instruction, entry)

    return len(
        tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
    )


def load_model(use_yarn):
    config = AutoConfig.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )

    if use_yarn:
        config.rope_scaling = {
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "type": "yarn",
        }
        config.max_position_embeddings = YARN_LIMIT

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        config=config,
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

    # Avoid irrelevant sampling warnings.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    # Same memory optimization used successfully previously:
    # during greedy decoding only final-position logits are consumed.
    def keep_last_hidden_only(module, args):
        hidden = args[0]
        return (hidden[:, -1:, :],)

    hook = model.lm_head.register_forward_pre_hook(
        keep_last_hidden_only
    )

    return model, hook


@torch.inference_mode()
def generate(model, instruction, entry, max_context):
    text = build_chat(instruction, entry)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    prompt_tokens = inputs["input_ids"].shape[1]

    if prompt_tokens + MAX_NEW_TOKENS > max_context:
        raise ValueError(
            f"context overflow: prompt={prompt_tokens} "
            f"new={MAX_NEW_TOKENS} limit={max_context}"
        )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_len = inputs["input_ids"].shape[1]

    generated = output[0, input_len:]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return answer, prompt_tokens


def unload(model, hook):
    hook.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--max-short",
        type=int,
        default=None,
        help="maximum NEW standard-context sessions",
    )

    ap.add_argument(
        "--max-long",
        type=int,
        default=None,
        help="maximum NEW YaRN sessions",
    )

    args = ap.parse_args()

    sessions = load_jsonl(INPUT)

    assert len(sessions) == 1249, len(sessions)

    hashes = [x["hash"] for x in sessions]

    assert len(hashes) == len(set(hashes))

    done = {}

    if OUTPUT.exists():
        for x in load_jsonl(OUTPUT):
            h = x["hash"]

            if h in done:
                raise RuntimeError(f"duplicate output hash: {h}")

            done[h] = x

    assert set(done).issubset(set(hashes))

    short = []
    long = []

    print("===== CLASSIFY SESSIONS =====", flush=True)

    for i, x in enumerate(sessions, 1):
        entry = session_text(x)

        # Summary prompt is longer than keyword prompt,
        # so use it for conservative routing.
        n = count_prompt(
            SUMMARY_PROMPT,
            entry,
        )

        x2 = dict(x)
        x2["_summary_tokens"] = n

        if n + MAX_NEW_TOKENS <= DEFAULT_LIMIT:
            short.append(x2)
        else:
            long.append(x2)

        if i % 250 == 0:
            print(f"classified {i}/{len(sessions)}", flush=True)

    print()
    print("total:", len(sessions))
    print("standard:", len(short))
    print("yarn4:", len(long))
    print("already_done:", len(done))

    assert len(short) + len(long) == 1249
    print(f"Sessions requiring YaRN: {len(long)}", flush=True)

    def run_phase(rows, mode, use_yarn, limit):
        pending = [
            x for x in rows
            if x["hash"] not in done
        ]

        if limit is not None:
            pending = pending[:limit]

        print()
        print(f"===== PHASE {mode} =====")
        print("pending_this_run:", len(pending))

        if not pending:
            return

        max_context = (
            YARN_LIMIT
            if use_yarn
            else DEFAULT_LIMIT
        )

        model, hook = load_model(use_yarn)

        print(
            f"{mode} model loaded; "
            f"context_limit={max_context}",
            flush=True,
        )

        with OUTPUT.open("a", encoding="utf-8") as out:
            for j, x in enumerate(pending, 1):
                h = x["hash"]
                entry = session_text(x)

                print(
                    f"[{mode} {j}/{len(pending)}] "
                    f"{h[:12]} "
                    f"audited_tokens={x['_summary_tokens']}",
                    flush=True,
                )

                summary, st = generate(
                    model,
                    SUMMARY_PROMPT,
                    entry,
                    max_context,
                )

                keywords, kt = generate(
                    model,
                    KEYWORD_PROMPT,
                    entry,
                    max_context,
                )

                if not summary:
                    raise RuntimeError(
                        f"empty summary: {h}"
                    )

                if not keywords:
                    raise RuntimeError(
                        f"empty keywords: {h}"
                    )

                row = {
                    "hash": h,
                    "summary": summary,
                    "keywords": keywords,
                    "generation_mode": mode,
                    "summary_prompt_tokens": st,
                    "keyword_prompt_tokens": kt,
                    "max_new_tokens": MAX_NEW_TOKENS,
                }

                out.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    ) + "\n"
                )
                out.flush()

                done[h] = row

        unload(model, hook)

    # Standard Qwen for short/normal sessions.
    run_phase(
        short,
        mode="standard",
        use_yarn=False,
        limit=args.max_short,
    )

    # Official Qwen YaRN only for >32K sessions.
    run_phase(
        long,
        mode="yarn4",
        use_yarn=True,
        limit=args.max_long,
    )

    final_rows = load_jsonl(OUTPUT)

    final_by_hash = {
        x["hash"]: x
        for x in final_rows
    }

    if len(final_by_hash) != len(final_rows):
        raise RuntimeError("duplicate hashes in final output")

    standard_done = sum(
        x.get("generation_mode") == "standard"
        for x in final_rows
    )

    yarn_done = sum(
        x.get("generation_mode") == "yarn4"
        for x in final_rows
    )

    print()
    print("===== CURRENT OUTPUT AUDIT =====")
    print("records:", len(final_rows))
    print("unique:", len(final_by_hash))
    print("standard_done:", standard_done)
    print("yarn4_done:", yarn_done)
    print("remaining:", len(sessions) - len(final_by_hash))

    if len(final_by_hash) == len(sessions):
        assert set(final_by_hash) == set(hashes)

        assert all(
            isinstance(x["summary"], str)
            and x["summary"].strip()
            for x in final_rows
        )

        assert all(
            isinstance(x["keywords"], str)
            and x["keywords"].strip()
            for x in final_rows
        )

        print("MEMGAS_1M_SESSION_GENERATION_OK")
    else:
        print("MEMGAS_1M_SESSION_GENERATION_PARTIAL_OK")


if __name__ == "__main__":
    main()
