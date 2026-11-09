# SCM4LLMs Reproducibility Issues

Repository:
https://github.com/wbbeyourself/SCM4LLMs

Official baseline commit:
26216044f5d4caccc0303d86667b491ead11d878

Security/compatibility branch:
modern-api-adapter

## SCM-API-001 — Hard-coded third-party API endpoint

Original `env.py` forces:

https://api.chatanywhere.com.cn/v1

through `OPENAI_API_BASE`.

Impact:
A real OpenAI API key used with the unmodified repository may be sent to
a third-party OpenAI-compatible endpoint.

Reproduction policy:
Never place a real OpenAI API key into the unmodified repository.

The `modern-api-adapter` branch removes the forced endpoint without
changing SCM memory/retrieval algorithms.

---

## SCM-ENV-001 — Unbounded legacy OpenAI SDK dependency

`requirements.txt` specifies:

openai>=0.27.0

but the source relies on legacy interfaces:

- openai.Embedding.create
- openai.Completion.create
- openai.ChatCompletion.create

The reproduction environment therefore pins:

openai==0.27.0

This is an environment compatibility decision, not an algorithm change.

---

## SCM-CODE-001 — Non-terminating controller loop

`get_related_turn()` can enter a non-terminating loop when:

1. selected memories exceed `MAX_HISTORY_TOKENS`;
2. controller compression reduces candidates to summary/drop states;
3. summaries still exceed the token budget;
4. later loop iterations no longer reduce any token lengths.

Dynamic reproduction:

A synthetic over-budget test was run under:

timeout 5s python ...

Observed:

timeout_exit=124

The process failed to return within 5 seconds and was terminated by
GNU `timeout`.

Impact:
Certain long-memory inputs can make SCM retrieval hang indefinitely.

Policy:
Do not silently patch this behavior in the original reproduction path.
Any termination guard must be implemented and reported as an adaptation.

---

## SCM-CODE-002 — Naive-memory character/token mismatch

In the naive-memory path, when a retrieved turn is larger than the
history budget, truncation uses Python character slicing:

cur_text[:(MAX_HISTORY_TOKENS - 500)]

although the budget is defined and checked in tokens.

Impact:
Actual context length can differ from the configured token budget,
especially across languages.

---

## SCM-EVAL-001 — Incomplete automatic evaluation pipeline

The public repository contains:

- published markdown results;
- annotation data;
- human annotation scripts;
- result-to-markdown utilities.

A complete end-to-end automatic QA/judge evaluation pipeline matching
the paper experiments was not found in the audited repository.

Published result files must not be represented as newly reproduced
results.

---

## SCM-MODEL-001 — Historical model dependency

The original implementation depends on:

- text-embedding-ada-002
- text-davinci-003
- gpt-3.5-turbo

No official API call cache was found in the cloned repository.

Therefore:
restoring the Python environment does not imply that the original
model-based experiment can still be rerun identically today.

Any modern-model experiment must be reported separately from the
original-paper reproduction.

---

# Dynamically verified SCM behavior

The following behaviors were executed successfully in the local
reproduction environment without external API calls:

- `Turn` and `ChatBot.history`;
- previous-turn handling;
- cosine similarity;
- top-k long-term retrieval;
- exclusion of previous turn from long-term retrieval;
- controller outputs: raw / summary / drop;
- controller is not invoked while under the token budget;
- controller is invoked after exceeding the token budget;
- the highest-similarity selected candidate is protected as raw during
  the first compression round;
- `naive_memory` performs semantic retrieval without using the
  raw/summary/drop controller.

Environment used for offline validation:

- Python 3.8.20
- openai 0.27.0
- numpy 1.24.4
- tiktoken 0.7.0
- chardet 5.2.0
- torch 2.1.2+cpu
