src = open(
    "run_personamem_memgas_qa.py",
    encoding="utf-8",
).read()

# Only load definitions/model; stop before Stage 1 loop.
src = src.split(
    "# ------------------------------------------------------------\n# Stage 1:"
)[0]

exec(src)

sample = retrieval_rows[0]

retrieved_texts = ""

for item in sample["retrieval_results"]["ranked_items"][:3]:
    h = item["session_hash"]
    m = memory_by_hash[h]

    retrieved_texts += (
        "\nSession Content:\n"
        f"{str(item['session'])}\n\n"
        "Session Summary:\n"
        f"{m['summary']}\n"
        "Session Keyword:\n"
        f"{m['keywords']}\n"
    )

prompt = PROMPT_MULTIGRAN.format(
    retrieved_texts=retrieved_texts,
    question=sample["question"],
)

filtered, n = generate(
    prompt,
    max_new_tokens=4000,
)

print("filter_prompt_tokens:", n)
print("filtered_chars:", len(filtered))

final_prompt = PROMPT_FINAL.format(
    filtered_text=filtered,
    question=sample["question"],
    instruction=PERSONAMEM_INSTRUCTION,
    all_options=sample["all_options"],
)

answer, n2 = generate(
    final_prompt,
    max_new_tokens=512,
)

score, parsed, found = extract_answer(
    answer,
    sample["correct_answer"],
)

print("answer_prompt_tokens:", n2)
print("response:", answer)
print("parsed:", parsed)
print("options_found:", found)
print("correct_answer:", sample["correct_answer"])
print("score:", score)

print("PERSONAMEM_MEMGAS_QA_SMOKE_OK")
