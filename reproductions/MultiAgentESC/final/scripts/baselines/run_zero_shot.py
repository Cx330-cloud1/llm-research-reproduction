import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI


PROMPT = """### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'.
Your task is to play a role as 'Assistant' and generate a response based on the given dialogue context.

### Dialogue context
{context}

Your answer must be fewer than 30 words and must follow this format:
Response: [response]
"""


def clean_response(text):
    m = re.search(r"Response:\s*(.*)", text, flags=re.I | re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/root/autodl-tmp/research/results/multiagentesc_qwen25_32b_100.json",
    )
    parser.add_argument(
        "--output",
        default="/root/autodl-tmp/research/results/table3_gpt4o/baselines/zero_shot_1210.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data = json.load(open(args.input, encoding="utf-8"))

    end = len(data) if args.limit is None else min(args.limit, len(data))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    completed = set()

    if output.exists():
        with output.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    x = json.loads(line)
                    if x.get("status") == "ok":
                        completed.add(x["index"])
                except Exception:
                    pass

    client = OpenAI(
        base_url="http://127.0.0.1:11434/v1",
        api_key="NULL",
        timeout=600,
        max_retries=0,
    )

    print("Total source records:", len(data))
    print("Will process:", end)
    print("Already completed:", len(completed))

    for i in range(end):
        if i in completed:
            print(f"[{i+1}/{end}] skip")
            continue

        item = data[i]
        prompt = PROMPT.format(context=item["context"])

        result = None
        last_error = None

        for attempt in range(1, 4):
            try:
                r = client.chat.completions.create(
                    model="qwen2.5:32b",
                    temperature=0,
                    max_tokens=100,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )

                raw = r.choices[0].message.content

                result = {
                    "index": i,
                    "status": "ok",
                    "strategy": item.get("strategy"),
                    "reference": item.get("reference"),
                    "context": item["context"],
                    "response": clean_response(raw),
                    "raw_response": raw,
                    "method": "Zero-shot",
                    "model": "qwen2.5:32b",
                    "temperature": 0,
                }
                break

            except Exception as e:
                last_error = repr(e)
                print(
                    f"[{i+1}/{end}] attempt {attempt}/3 failed:",
                    last_error,
                )
                time.sleep(attempt * 3)

        if result is None:
            result = {
                "index": i,
                "status": "error",
                "context": item["context"],
                "error": last_error,
                "method": "Zero-shot",
            }

        with output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(
            f"[{i+1}/{end}]",
            result["status"],
            repr(result.get("response", ""))[:100],
        )


if __name__ == "__main__":
    main()
