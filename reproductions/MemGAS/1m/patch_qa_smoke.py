from pathlib import Path
import shutil

p = Path("run_personamem_memgas_qa_1m_smoke.py")
src = p.read_text(encoding="utf-8")

backup = p.with_name(
    p.name + ".before_promptfix.bak"
)

if not backup.exists():
    shutil.copy2(p, backup)

def replace_once(old, new, label):
    global src
    count = src.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected 1 match, got {count}. "
            "No changes have been written."
        )
    src = src.replace(old, new, 1)
    print("PATCHED:", label)

# 1. Add token-budgeted clipping for individual memory sections.
marker = "# PersonaMem-v1 official-style answer extraction"

helper = r'''def clip_text_tokens(text, max_tokens):
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


'''

replace_once(
    marker,
    helper + marker,
    "add memory-section token clipping",
)

# 2. Cache clipped memory sections across questions.
stage_marker = r'''with open(
    FILTER_CACHE,
    "a",
    encoding="utf-8",
) as out_f:'''

replace_once(
    stage_marker,
    "memory_clip_cache = {}\n\n" + stage_marker,
    "add clipped-memory cache",
)

# 3. Bound each retrieved memory independently.
old_assembly = r'''            # Official MemGAS inserts the Python list object
            # through an f-string; str(list) preserves that behavior.
            session_text = str(
                item["session"]
            )

            retrieved_texts += (
                "\nSession Content:\n"
                f"{session_text}\n\n"
                "Session Summary:\n"
                f"{memory['summary']}\n"
                "Session Keyword:\n"
                f"{memory['keywords']}\n"
            )'''

new_assembly = r'''            if h not in memory_clip_cache:
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
            )'''

replace_once(
    old_assembly,
    new_assembly,
    "budget each retrieved memory",
)

# 4. Remove silent left truncation of the entire prompt.
old_tokenizer = r'''    # Smoke-test safeguard: bound very long memory prompts.
    # Keep the end of the prompt, including the question.
    tokenizer.truncation_side = "left"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=4096,
    )'''

new_tokenizer = r'''    # Memory sections are budgeted before prompt assembly.
    # Never silently truncate task instructions or the question.
    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )'''

replace_once(
    old_tokenizer,
    new_tokenizer,
    "remove whole-prompt truncation",
)

old_length = r'''    input_len = inputs["input_ids"].shape[1]

    output = model.generate('''

new_length = r'''    input_len = inputs["input_ids"].shape[1]

    if input_len > 4096:
        raise ValueError(
            f"Prompt exceeds 4096 tokens: {input_len}. "
            "Reduce memory-section budgets before generation."
        )

    output = model.generate('''

replace_once(
    old_length,
    new_length,
    "add explicit prompt-length guard",
)

p.write_text(src, encoding="utf-8")
print("QA_SMOKE_PATCH_OK")
print("Backup:", backup)
