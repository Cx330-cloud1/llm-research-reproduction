import argparse
import json
import re
import time
from pathlib import Path
from collections import defaultdict

from openai import OpenAI


BASE = Path("/root/autodl-tmp/research/results")
B = BASE / "table3_gpt4o/baselines"

FILES = {
    "Zero-shot": B / "zero_shot_1210.jsonl",
    "Few-shot CoT": B / "fewshot_cot_retrieval_1210.jsonl",
    "Self-consistency": B / "self_consistency_1210.jsonl",
    "Self-Refine": B / "self_refine_1210.jsonl",
}

OURS = BASE / "multiagentesc_qwen25_32b_100.json"

METHOD_ORDER = [
    "Zero-shot",
    "Few-shot CoT",
    "Self-consistency",
    "Self-Refine",
    "MultiAgentESC",
]

LETTERS = ["A", "B", "C", "D", "E"]

DIMENSIONS = [
    "Fluency",
    "Identification",
    "Comforting",
    "Suggestion",
    "Overall",
]


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


def build_prompt(context, responses):
    return f"""You are an impartial evaluator with expertise in psychology and linguistics.

Below is a dialogue context followed by five anonymous candidate responses A-E.

Evaluate each candidate independently from 1 to 5 on:

Fluency:
Naturalness, coherence, fluency, and contextual fit.

Identification:
How accurately the response understands the user's situation and underlying problem.

Comforting:
Empathy, reassurance, emotional comfort, and supportiveness.

Suggestion:
Usefulness and practicality of suggestions. Do not penalize a response if suggestions are not appropriate at this stage.

Overall:
Overall effectiveness as emotional support.

Important:
- Judge content, not response position.
- Do not favor longer responses.
- Do not infer which system generated each response.
- Scores must be integers from 1 to 5.
- Every dimension MUST contain exactly five scores corresponding to A, B, C, D, E.

Dialogue:
{context}

Candidate A:
{responses["A"]}

Candidate B:
{responses["B"]}

Candidate C:
{responses["C"]}

Candidate D:
{responses["D"]}

Candidate E:
{responses["E"]}

Return ONLY valid JSON, with no markdown and no text before or after it.

Use exactly this structure:

{{
  "Fluency": {{
    "scores": [1, 1, 1, 1, 1],
    "reason": "brief reason"
  }},
  "Identification": {{
    "scores": [1, 1, 1, 1, 1],
    "reason": "brief reason"
  }},
  "Comforting": {{
    "scores": [1, 1, 1, 1, 1],
    "reason": "brief reason"
  }},
  "Suggestion": {{
    "scores": [1, 1, 1, 1, 1],
    "reason": "brief reason"
  }},
  "Overall": {{
    "scores": [1, 1, 1, 1, 1],
    "reason": "brief reason"
  }}
}}

Replace the example scores with your actual evaluation.
"""


def extract_json(text):
    text = text.strip()

    # Remove accidental markdown fences.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I
    )
    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")

    return json.loads(
        text[start:end + 1]
    )


def validate_scores(obj):
    parsed = {}

    for dim in DIMENSIONS:
        if dim not in obj:
            raise ValueError(
                f"Missing dimension: {dim}"
            )

        scores = obj[dim].get("scores")

        if not isinstance(scores, list):
            raise ValueError(
                f"{dim}: scores is not a list"
            )

        if len(scores) != 5:
            raise ValueError(
                f"{dim}: expected 5 scores, got {scores}"
            )

        clean = []

        for x in scores:
            try:
                x = int(x)
            except Exception:
                raise ValueError(
                    f"{dim}: invalid score {x}"
                )

            if x < 1 or x > 5:
                raise ValueError(
                    f"{dim}: out-of-range score {x}"
                )

            clean.append(x)

        parsed[dim] = {
            letter: score
            for letter, score
            in zip(LETTERS, clean)
        }

    return parsed


def call_judge(client, prompt):
    last_error = None

    for attempt in range(1, 4):
        try:
            r = client.chat.completions.create(
                model="mistral-small:24b",
                temperature=0,
                max_tokens=700,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            raw = (
                r.choices[0]
                .message.content
                .strip()
            )

            obj = extract_json(raw)
            scores = validate_scores(obj)

            return raw, scores

        except Exception as e:
            last_error = repr(e)

            print(
                f"judge attempt {attempt}/3 failed:",
                last_error,
                flush=True
            )

            time.sleep(attempt * 2)

    raise RuntimeError(last_error)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--output",
        default=str(
            BASE /
            "table3_gpt4o/"
            "mistral_proxy_judge_balanced_1210.jsonl"
        )
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None
    )

    args = ap.parse_args()

    with open(
        OURS,
        encoding="utf-8"
    ) as f:
        ours = json.load(f)

    baselines = {
        name: load_jsonl(path)
        for name, path
        in FILES.items()
    }

    n = len(ours)

    assert n == 1210

    for name, rows in baselines.items():
        assert len(rows) == n, (
            f"{name}: {len(rows)} != {n}"
        )

        for i in range(n):
            assert (
                rows[i]["context"].strip()
                == ours[i]["context"].strip()
            ), (
                f"Context mismatch: "
                f"{name} index={i}"
            )

    print(
        "ALIGNMENT = PASS:",
        n,
        "records",
        flush=True
    )

    method_data = {
        **baselines,
        "MultiAgentESC": {
            i: x
            for i, x in enumerate(ours)
        }
    }

    client = OpenAI(
        base_url="http://127.0.0.1:11434/v1",
        api_key="NULL",
        timeout=600,
        max_retries=0,
    )

    out = Path(args.output)
    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    completed = set()

    if out.exists():
        with out.open(
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
        n
        if args.limit is None
        else min(args.limit, n)
    )

    print("process =", end)
    print(
        "already completed =",
        len(completed),
        flush=True
    )

    for i in range(end):
        if i in completed:
            continue

        started = time.time()

        # Perfectly balanced cyclic position assignment.
        shift = i % 5

        sample_order = (
            METHOD_ORDER[shift:]
            + METHOD_ORDER[:shift]
        )

        responses = {}
        method_map = {}

        for letter, method in zip(
            LETTERS,
            sample_order
        ):
            responses[letter] = (
                method_data[method][i]
                .get("response", "")
                .strip()
            )

            method_map[letter] = method

        prompt = build_prompt(
            ours[i]["context"],
            responses
        )

        try:
            raw, scores = call_judge(
                client,
                prompt
            )

            result = {
                "index": i,
                "status": "ok",
                "context":
                    ours[i]["context"],

                "method_map":
                    method_map,

                "scores":
                    scores,

                "raw_judgement":
                    raw,

                "judge":
                    "mistral-small:24b",

                "temperature":
                    0,

                "position_design":
                    "balanced cyclic",

                "elapsed_seconds":
                    round(
                        time.time() - started,
                        3
                    ),

                "evaluation_type":
                    "local proxy judge",

                "protocol":
                    "paper five-dimensional judge protocol; JSON output implementation",
            }

        except Exception as e:
            result = {
                "index": i,
                "status": "error",
                "context":
                    ours[i]["context"],
                "error":
                    repr(e),
            }

        with out.open(
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
            "time=",
            result.get(
                "elapsed_seconds"
            ),
            flush=True
        )

    rows = {}

    with out.open(
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
                rows[x["index"]] = x

    sums = defaultdict(
        lambda: defaultdict(float)
    )

    counts = defaultdict(
        lambda: defaultdict(int)
    )

    position_counts = defaultdict(
        lambda: defaultdict(int)
    )

    for x in rows.values():
        mapping = x["method_map"]

        for letter, method in mapping.items():
            position_counts[method][letter] += 1

        for dim in DIMENSIONS:
            for letter in LETTERS:
                method = mapping[letter]

                sums[method][dim] += (
                    x["scores"][dim][letter]
                )

                counts[method][dim] += 1

    print(
        "\n=== PROXY JUDGE SUMMARY ==="
    )

    print(
        "unique ok =",
        len(rows)
    )

    print(
        "missing indexes =",
        sorted(
            set(range(end))
            - set(rows)
        )[:20]
    )

    if rows:
        elapsed = [
            x["elapsed_seconds"]
            for x in rows.values()
            if "elapsed_seconds" in x
        ]

        print(
            "average seconds/item =",
            round(
                sum(elapsed) / len(elapsed),
                3
            ) if elapsed else 0
        )

        print(
            "timed records =",
            len(elapsed)
        )

    print()

    print(
        "Method\t"
        + "\t".join(DIMENSIONS)
    )

    for method in METHOD_ORDER:
        values = []

        for dim in DIMENSIONS:
            c = counts[method][dim]

            mean = (
                sums[method][dim] / c
                if c else 0
            )

            values.append(
                f"{mean:.3f}"
            )

        print(
            method
            + "\t"
            + "\t".join(values)
        )

    print(
        "\n=== POSITION COUNTS ==="
    )

    for method in METHOD_ORDER:
        print(
            method,
            {
                letter:
                    position_counts[method][letter]
                for letter in LETTERS
            }
        )


if __name__ == "__main__":
    main()
