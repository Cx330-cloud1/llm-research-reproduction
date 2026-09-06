import argparse
import json
import re
import time
from pathlib import Path
from openai import OpenAI


def load_jsonl(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            if x.get("status") == "ok":
                rows[x["index"]] = x
    return rows


def parse_feedback(text):
    m = re.search(
        r"Feedback:\s*(.*)",
        text,
        flags=re.I | re.S
    )
    return m.group(1).strip() if m else text.strip()


def parse_response(text):
    m = re.search(
        r"Response:\s*(.*)",
        text,
        flags=re.I | re.S
    )
    return m.group(1).strip() if m else None


def feedback_prompt(context, initial_response):
    return f"""### Instruction
You are a psychological counseling expert.

You will be provided with a dialogue context between an 'Assistant' and a 'User', together with a candidate Assistant response.

### Dialogue context
{context}

### Candidate response
{initial_response}

Please provide concise and constructive feedback on how the candidate response could better support the user.

Consider:
- whether it accurately understands the user's situation;
- whether it provides appropriate emotional support;
- whether it is empathetic and natural;
- whether useful suggestions should be provided;
- whether the response is concise and relevant.

Do not generate the revised response yet.

Your answer must follow this format:
Feedback: [feedback]
"""


def refine_prompt(context, initial_response, feedback):
    return f"""### Instruction
You are a psychological counseling expert.

You will be provided with a dialogue context, an initial Assistant response, and feedback on that response.

### Dialogue context
{context}

### Initial response
{initial_response}

### Feedback
{feedback}

Please refine the initial response according to the feedback.

The refined response should:
- directly address the user's current situation;
- provide appropriate emotional support;
- sound natural and empathetic;
- incorporate useful suggestions when appropriate;
- contain strictly fewer than 30 words.
Before answering, count the words in the refined response.
The final Response must contain between 1 and 29 words.
If your draft contains 30 or more words, shorten it before output.

Do not explain your revision.

Your answer must follow this format:
Response: [response]
"""


def call_model(client, prompt, max_tokens):
    last_error = None

    for attempt in range(1, 4):
        try:
            r = client.chat.completions.create(
                model="qwen2.5:32b",
                temperature=0,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )

            return (
                r.choices[0]
                .message.content
                .strip()
            )

        except Exception as e:
            last_error = repr(e)
            print(
                f"attempt {attempt}/3 failed:",
                last_error
            )
            time.sleep(attempt * 3)

    raise RuntimeError(last_error)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default=(
            "/root/autodl-tmp/research/results/"
            "multiagentesc_qwen25_32b_100.json"
        )
    )

    ap.add_argument(
        "--zero-shot",
        default=(
            "/root/autodl-tmp/research/results/"
            "table3_gpt4o/baselines/"
            "zero_shot_1210.jsonl"
        )
    )

    ap.add_argument(
        "--output",
        default=(
            "/root/autodl-tmp/research/results/"
            "table3_gpt4o/baselines/"
            "self_refine_1210.jsonl"
        )
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None
    )

    args = ap.parse_args()

    with open(
        args.input,
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    zero = load_jsonl(args.zero_shot)

    if len(zero) != len(data):
        raise RuntimeError(
            f"Zero-shot alignment failure: "
            f"{len(zero)} vs {len(data)}"
        )

    for i, item in enumerate(data):
        z = zero.get(i)

        if z is None:
            raise RuntimeError(
                f"Missing zero-shot index {i}"
            )

        if (
            z["context"].strip()
            != item["context"].strip()
        ):
            raise RuntimeError(
                f"Context mismatch at index {i}"
            )

    print(
        "alignment check = PASS:",
        len(data),
        "records"
    )

    client = OpenAI(
        base_url="http://127.0.0.1:11434/v1",
        api_key="NULL",
        timeout=600,
        max_retries=0,
    )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    completed = set()

    if output.exists():
        with output.open(
            encoding="utf-8"
        ) as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    x = json.loads(line)

                    if x.get("status") == "ok":
                        completed.add(
                            x["index"]
                        )

                except Exception:
                    pass

    end = (
        len(data)
        if args.limit is None
        else min(
            args.limit,
            len(data)
        )
    )

    print("process =", end)
    print(
        "already completed =",
        len(completed)
    )

    start_all = time.time()

    for i in range(end):
        if i in completed:
            continue

        start = time.time()

        item = data[i]
        initial_response = (
            zero[i]["response"]
        )

        result = None
        last_error = None

        try:
            # ---- Stage 1: self-feedback ----

            raw_feedback = call_model(
                client,
                feedback_prompt(
                    item["context"],
                    initial_response
                ),
                max_tokens=250,
            )

            feedback = parse_feedback(
                raw_feedback
            )

            if not feedback:
                raise ValueError(
                    "Empty feedback"
                )

            # ---- Stage 2: refinement ----

            raw_refined = call_model(
                client,
                refine_prompt(
                    item["context"],
                    initial_response,
                    feedback,
                ),
                max_tokens=100,
            )

            refined_response = (
                parse_response(
                    raw_refined
                )
            )

            if not refined_response:
                raise ValueError(
                    "Could not parse refined Response"
                )

            word_count = len(
                refined_response.split()
            )

            # Output-format enforcement only.
            # If the refined response violates the paper's <30-word
            # requirement, ask the same model to shorten it without
            # changing its meaning.
            length_correction_rounds = 0

            while word_count >= 30 and length_correction_rounds < 2:
                length_correction_rounds += 1

                length_prompt = f"""You are given an emotional-support response.

Dialogue context:
{item["context"]}

Current response:
{refined_response}

Rewrite ONLY the response so that:
- its meaning and emotional-support intent are preserved;
- no new information is added;
- it contains between 1 and 29 words.

Output exactly:
Response: [response]
"""

                raw_length_fix = call_model(
                    client,
                    length_prompt,
                    max_tokens=80,
                )

                fixed_response = parse_response(
                    raw_length_fix
                )

                if not fixed_response:
                    raise ValueError(
                        "Could not parse length-corrected Response"
                    )

                refined_response = fixed_response
                word_count = len(
                    refined_response.split()
                )

            if word_count >= 30:
                raise ValueError(
                    f"Length constraint still violated: {word_count} words"
                )

            result = {
                "index": i,
                "status": "ok",

                "strategy":
                    item.get("strategy"),
                "reference":
                    item.get("reference"),
                "context":
                    item["context"],

                "initial_response":
                    initial_response,

                "feedback":
                    feedback,

                "response":
                    refined_response,

                "raw_feedback":
                    raw_feedback,
                "raw_refined_response":
                    raw_refined,

                "changed_from_initial":
                    refined_response.strip()
                    != initial_response.strip(),

                "word_count":
                    word_count,

                "under_30_words":
                    word_count < 30,

                "length_correction_rounds":
                    length_correction_rounds,

                "elapsed_seconds":
                    round(
                        time.time()
                        - start,
                        3
                    ),

                "method":
                    "Self-Refine reconstruction",

                "model":
                    "qwen2.5:32b",

                "temperature":
                    0,

                "reconstruction": {
                    "initial_response":
                        "frozen Zero-shot Prompt 8 output",
                    "feedback_rounds":
                        1,
                    "refinement_rounds":
                        1,
                    "feedback_prompt":
                        "task-specific reconstruction",
                    "refinement_prompt":
                        "task-specific reconstruction",
                    "author_iteration_count_reported":
                        False,
                    "author_self_refine_prompts_reported":
                        False,
                },
            }

        except Exception as e:
            last_error = repr(e)

            result = {
                "index": i,
                "status": "error",
                "context":
                    item["context"],
                "initial_response":
                    initial_response,
                "error":
                    last_error,
            }

        with output.open(
            "a",
            encoding="utf-8"
        ) as f:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                )
                + "\n"
            )

        print(
            f"[{i+1}/{end}]",
            result["status"],
            "changed=",
            result.get(
                "changed_from_initial"
            ),
            "words=",
            result.get(
                "word_count"
            ),
            "time=",
            result.get(
                "elapsed_seconds"
            ),
            "response=",
            repr(
                result.get(
                    "response",
                    ""
                )
            )[:90],
            flush=True,
        )

    # ---------- sanity ----------

    rows = []

    with output.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)

            if (
                x.get("status") == "ok"
                and x["index"] < end
            ):
                rows.append(x)

    unique_responses = len(
        set(
            x["response"]
            for x in rows
        )
    )

    changed = sum(
        x.get(
            "changed_from_initial",
            False
        )
        for x in rows
    )

    under30 = sum(
        x.get(
            "under_30_words",
            False
        )
        for x in rows
    )

    avg_time = (
        sum(
            x.get(
                "elapsed_seconds",
                0
            )
            for x in rows
        )
        / len(rows)
        if rows else 0
    )

    print(
        "\n=== SANITY CHECK ==="
    )
    print(
        "ok =",
        len(rows)
    )
    print(
        "unique refined responses =",
        unique_responses
    )
    print(
        "changed from initial =",
        changed
    )
    print(
        "under 30 words =",
        under30
    )
    print(
        "average seconds/item =",
        round(avg_time, 3)
    )

    if len(rows) >= 10:
        if unique_responses <= 2:
            print(
                "SANITY = FAIL: response collapse"
            )

        elif changed <= 2:
            print(
                "SANITY = FAIL: refinement inactive"
            )

        elif under30 < (
            len(rows) * 0.8
        ):
            print(
                "SANITY = FAIL: excessive length violations"
            )

        else:
            print(
                "SANITY = PASS"
            )


if __name__ == "__main__":
    main()
