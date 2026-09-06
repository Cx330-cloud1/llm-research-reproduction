import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from openai import OpenAI


ROOT = Path("/root/autodl-tmp/research/results")
B = ROOT / "table3_gpt4o/baselines"

OURS_PATH = ROOT / "multiagentesc_qwen25_32b_100.json"

BASELINE_FILES = {
    "Zero-shot": B / "zero_shot_1210.jsonl",
    "Few-shot CoT": B / "fewshot_cot_retrieval_1210.jsonl",
    "Self-consistency": B / "self_consistency_1210.jsonl",
    "Self-Refine": B / "self_refine_1210.jsonl",
}

BASELINES = list(BASELINE_FILES)

DIMENSIONS = [
    "Fluency",
    "Identification",
    "Comforting",
    "Suggestion",
    "Overall",
]

ANNOTATORS = [
    {
        "id": "A",
        "profile": (
            "Prioritize accurate identification of the help-seeker's "
            "problem, psychological appropriateness, and useful support."
        ),
    },
    {
        "id": "B",
        "profile": (
            "Prioritize empathy, validation, emotional comfort, "
            "naturalness, and whether the help-seeker feels understood."
        ),
    },
    {
        "id": "C",
        "profile": (
            "Use a balanced perspective considering understanding, "
            "comfort, suggestions, fluency, and overall usefulness."
        ),
    },
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


def extract_json(text):
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end <= start:
        raise ValueError("No JSON object found")

    return json.loads(text[start:end + 1])


def validate(obj):
    result = {}

    for i in range(1, 5):
        cid = f"C{i}"

        if cid not in obj:
            raise ValueError(f"Missing {cid}")

        result[cid] = {}

        for dim in DIMENSIONS:
            value = str(
                obj[cid].get(dim, "")
            ).strip().lower()

            if value == "a":
                result[cid][dim] = "A"

            elif value == "b":
                result[cid][dim] = "B"

            elif value in {
                "tie",
                "equal",
                "same",
            }:
                result[cid][dim] = "Tie"

            else:
                raise ValueError(
                    f"{cid}/{dim}: invalid value {value}"
                )

    return result


def build_prompt(context, comparisons, profile):
    blocks = []

    for cid, x in comparisons.items():
        blocks.append(
            f"""### {cid}
Candidate A:
{x["A"]}

Candidate B:
{x["B"]}"""
        )

    pairs = "\n\n".join(blocks)

    return f"""You are simulating an independent evaluator of emotional-support conversations.

This is a simulated evaluation, not a real human-subject experiment.

Evaluator perspective:
{profile}

Assume the role of the help-seeker.

Dialogue context:
{context}

Below are four independent pairwise comparisons.
For each comparison, choose A, B, or Tie on every dimension.

Fluency:
Which response is more coherent, natural, and smooth?

Identification:
Which response better understands and identifies the help-seeker's problem?

Comforting:
Which response provides better empathy, reassurance, and emotional comfort?

Suggestion:
Which response gives more useful and appropriate suggestions?
Do not reward advice merely because more advice is given.

Overall:
Which response provides more effective emotional support overall?

Important:
- Judge each comparison independently.
- Do not infer the source model.
- Do not prefer A or B because of position.
- Do not prefer longer responses merely for being longer.

{pairs}

Return ONLY valid JSON:

{{
  "C1": {{
    "Fluency": "A",
    "Identification": "A",
    "Comforting": "A",
    "Suggestion": "A",
    "Overall": "A"
  }},
  "C2": {{
    "Fluency": "A",
    "Identification": "A",
    "Comforting": "A",
    "Suggestion": "A",
    "Overall": "A"
  }},
  "C3": {{
    "Fluency": "A",
    "Identification": "A",
    "Comforting": "A",
    "Suggestion": "A",
    "Overall": "A"
  }},
  "C4": {{
    "Fluency": "A",
    "Identification": "A",
    "Comforting": "A",
    "Suggestion": "A",
    "Overall": "A"
  }}
}}

Replace every value with A, B, or Tie.
"""


def majority(votes):
    c = Counter(votes)

    value, count = c.most_common(1)[0]

    if count >= 2:
        return value

    return "Tie"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--limit",
        type=int,
        default=100,
    )

    ap.add_argument(
        "--output",
        default=str(
            ROOT /
            "table2_simulated/"
            "simulated_human_eval_v2_100.jsonl"
        ),
    )

    args = ap.parse_args()

    ours = json.load(
        open(
            OURS_PATH,
            encoding="utf-8"
        )
    )

    baselines = {
        name: load_jsonl(path)
        for name, path
        in BASELINE_FILES.items()
    }

    assert len(ours) == 1210

    for name, rows in baselines.items():
        assert len(rows) == 1210, (
            name,
            len(rows),
        )

        for i in range(1210):
            assert (
                rows[i]["context"].strip()
                == ours[i]["context"].strip()
            ), f"context mismatch {name}/{i}"

    print(
        "ALIGNMENT = PASS: 1210 response-level contexts"
    )

    # Frozen reconstructed sampling rule.
    rng = random.Random(42)

    sampled_indexes = rng.sample(
        range(1210),
        100,
    )

    limit = min(
        args.limit,
        100,
    )

    sampled_indexes = (
        sampled_indexes[:limit]
    )

    print(
        "sampled cases =",
        len(sampled_indexes)
    )

    print(
        "sample indexes =",
        sampled_indexes,
    )

    out = Path(args.output)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = set()

    if out.exists():
        with out.open(
            encoding="utf-8"
        ) as f:
            for line in f:
                if not line.strip():
                    continue

                x = json.loads(line)

                if x.get("status") == "ok":
                    completed.add(
                        (
                            x["sample_index"],
                            x["annotator_id"],
                        )
                    )

    print(
        "already completed =",
        len(completed)
    )

    client = OpenAI(
        base_url="http://127.0.0.1:11434/v1",
        api_key="NULL",
        timeout=600,
        max_retries=0,
    )

    for sample_no, idx in enumerate(
        sampled_indexes
    ):
        context = ours[idx]["context"]
        ours_response = ours[idx]["response"]

        for ann_i, ann in enumerate(
            ANNOTATORS
        ):
            key = (
                idx,
                ann["id"],
            )

            if key in completed:
                continue

            comparisons = {}
            comparison_map = {}

            for b_i, baseline in enumerate(
                BASELINES
            ):
                cid = f"C{b_i + 1}"

                baseline_response = (
                    baselines[baseline][idx]
                    ["response"]
                )

                # Deterministic balanced side assignment.
                ours_is_a = (
                    (
                        sample_no
                        + b_i
                        + ann_i
                    )
                    % 2
                    == 0
                )

                if ours_is_a:
                    comparisons[cid] = {
                        "A": ours_response,
                        "B": baseline_response,
                    }

                    ours_position = "A"

                else:
                    comparisons[cid] = {
                        "A": baseline_response,
                        "B": ours_response,
                    }

                    ours_position = "B"

                comparison_map[cid] = {
                    "baseline": baseline,
                    "ours_position":
                        ours_position,
                }

            prompt = build_prompt(
                context,
                comparisons,
                ann["profile"],
            )

            result = None
            last_error = None
            started = time.time()

            for attempt in range(1, 4):
                try:
                    r = (
                        client.chat.completions
                        .create(
                            model="mistral-small:24b",
                            temperature=0,
                            max_tokens=500,
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                        )
                    )

                    raw = (
                        r.choices[0]
                        .message.content
                        .strip()
                    )

                    choices = validate(
                        extract_json(raw)
                    )

                    decisions = {}

                    for cid, dims in (
                        choices.items()
                    ):
                        ours_position = (
                            comparison_map[cid]
                            ["ours_position"]
                        )

                        decisions[cid] = {}

                        for dim, choice in (
                            dims.items()
                        ):
                            if choice == "Tie":
                                decision = "Tie"

                            elif (
                                choice
                                == ours_position
                            ):
                                decision = "Win"

                            else:
                                decision = "Lose"

                            decisions[cid][dim] = (
                                decision
                            )

                    result = {
                        "sample_no":
                            sample_no,

                        "sample_index":
                            idx,

                        "annotator_id":
                            ann["id"],

                        "status":
                            "ok",

                        "comparison_map":
                            comparison_map,

                        "raw_choices":
                            choices,

                        "decisions_relative_to_multiagentesc":
                            decisions,

                        "raw_judgement":
                            raw,

                        "judge_model":
                            "mistral-small:24b",

                        "simulation":
                            True,

                        "sampling_rule":
                            "random.Random(42).sample(range(1210), 100)",

                        "sample_unit":
                            "response-level dialogue context",

                        "elapsed_seconds":
                            round(
                                time.time()
                                - started,
                                3
                            ),
                    }

                    break

                except Exception as e:
                    last_error = repr(e)

                    print(
                        f"sample {sample_no} "
                        f"annotator {ann['id']} "
                        f"attempt {attempt}/3 failed: "
                        f"{last_error}",
                        flush=True,
                    )

                    time.sleep(
                        attempt * 2
                    )

            if result is None:
                result = {
                    "sample_no":
                        sample_no,

                    "sample_index":
                        idx,

                    "annotator_id":
                        ann["id"],

                    "status":
                        "error",

                    "error":
                        last_error,
                }

            with out.open(
                "a",
                encoding="utf-8"
            ) as f:
                f.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            print(
                f"[{sample_no+1}/{limit}]",
                ann["id"],
                result["status"],
                result.get(
                    "elapsed_seconds"
                ),
                flush=True,
            )

    # Latest successful record per
    # sample/annotator pair.
    ok = {}

    with out.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)

            if x.get("status") == "ok":
                key = (
                    x["sample_index"],
                    x["annotator_id"],
                )

                if (
                    x["sample_index"]
                    in sampled_indexes
                ):
                    ok[key] = x

    expected = (
        len(sampled_indexes)
        * 3
    )

    print(
        "\n=== SIMULATION CHECK ==="
    )

    print(
        "unique ok =",
        len(ok)
    )

    print(
        "expected =",
        expected
    )

    missing = []

    for idx in sampled_indexes:
        for ann in ANNOTATORS:
            if (
                idx,
                ann["id"],
            ) not in ok:
                missing.append(
                    (
                        idx,
                        ann["id"],
                    )
                )

    print(
        "missing =",
        missing
    )

    table = defaultdict(
        lambda: defaultdict(
            Counter
        )
    )

    unanimous = 0
    total_vote_sets = 0

    for idx in sampled_indexes:
        ann_rows = [
            ok.get(
                (
                    idx,
                    ann["id"],
                )
            )
            for ann in ANNOTATORS
        ]

        if any(
            x is None
            for x in ann_rows
        ):
            continue

        for b_i, baseline in enumerate(
            BASELINES
        ):
            cid = f"C{b_i + 1}"

            for dim in DIMENSIONS:
                votes = [
                    x[
                        "decisions_relative_to_multiagentesc"
                    ][cid][dim]
                    for x in ann_rows
                ]

                if len(set(votes)) == 1:
                    unanimous += 1

                total_vote_sets += 1

                decision = majority(votes)

                table[
                    baseline
                ][dim][decision] += 1

    print(
        "unanimous rate =",
        round(
            unanimous / total_vote_sets,
            4
        )
        if total_vote_sets
        else 0
    )

    print(
        "\n=== SIMULATED TABLE 2 ==="
    )

    for baseline in BASELINES:
        print(
            "\nMultiAgentESC vs",
            baseline
        )

        print(
            "Dimension\tWin\tLose\tTie"
        )

        for dim in DIMENSIONS:
            c = table[
                baseline
            ][dim]

            print(
                f"{dim}\t"
                f"{c['Win']}\t"
                f"{c['Lose']}\t"
                f"{c['Tie']}"
            )


if __name__ == "__main__":
    main()
