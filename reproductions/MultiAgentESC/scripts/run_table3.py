from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # keeps offline tests usable before dependencies are installed
    def tqdm(iterable, **_):
        return iterable

from .common import LLMClient, MockClient, append_jsonl, load_config, read_jsonl, slug
from .dataset_utils import final_target_per_dialogue


METHODS = ["Zero-shot", "Few-shot CoT", "Self-consistency", "Self-Refine", "MultiAgentESC (Ours)"]
DIMENSIONS = ["Fluency", "Identification", "Comforting", "Suggestion", "Overall"]
LETTERS = list("ABCDE")


def load_outputs(results: Path, backbone: str, allow_mock: bool = False) -> dict[str, dict[str, dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in (results / "generations").glob("*/*.jsonl"):
        for row in read_jsonl(path):
            if (
                row.get("backbone") == backbone
                and row.get("status") == "ok"
                and (allow_mock or not row.get("is_mock"))
                and row.get("variant", "full") == "full"
            ):
                grouped[row["method"]].append(row)
    return {method: final_target_per_dialogue(rows) for method, rows in grouped.items()}


def judge_prompt(context: str, responses: dict[str, str]) -> str:
    aspects = """Fluency (1-5): fluency and contextual coherence.
Identification (1-5): effectiveness in identifying the user's issues.
Comforting (1-5): reassurance and empathy.
Suggestion (1-5): useful and practical recommendations.
Overall (1-5): overall emotional support."""
    options = "\n".join(f"{letter}: {responses[letter]}" for letter in LETTERS)
    return f"""# Role
You are a judge with a background in psychology and linguistics.
# Task
Score five responses on five aspects. Avoid position, length, and assistant-name bias.
# Aspects
{aspects}
# Dialogue history
{context}
# Responses
{options}
Output strictly:
Fluency: [A_rating], [B_rating], [C_rating], [D_rating], [E_rating]; reasons
Identification: [A_rating], [B_rating], [C_rating], [D_rating], [E_rating]; reasons
Comforting: [A_rating], [B_rating], [C_rating], [D_rating], [E_rating]; reasons
Suggestion: [A_rating], [B_rating], [C_rating], [D_rating], [E_rating]; reasons
Overall: [A_rating], [B_rating], [C_rating], [D_rating], [E_rating]; reasons"""


def parse_scores(text: str) -> dict[str, list[float]]:
    parsed = {}
    for dimension in DIMENSIONS:
        match = re.search(rf"(?im)^\s*{dimension}\s*:\s*([^;\n]+)", text)
        if not match:
            raise ValueError(f"missing {dimension}")
        numbers = [float(value) for value in re.findall(r"(?<!\d)([1-5](?:\.\d+)?)(?!\d)", match.group(1))]
        if len(numbers) != 5:
            raise ValueError(f"{dimension}: expected 5 ratings, got {numbers}")
        parsed[dimension] = numbers
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.full.json"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--backbone")
    parser.add_argument("--max-dialogues", type=int, default=100)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    backbone = args.backbone or config.get("evaluation_backbone") or next(iter(config["models"]))
    outputs = load_outputs(args.results, backbone, allow_mock=args.mock)
    missing = [method for method in METHODS if method not in outputs]
    if missing:
        raise SystemExit(f"Missing Table 3 generations: {missing}")
    common = sorted(set.intersection(*(set(outputs[m]) for m in METHODS)))[:args.max_dialogues]
    if len(common) < args.max_dialogues:
        print(f"WARNING: only {len(common)} common dialogues")

    judge = MockClient("MOCK-JUDGE") if args.mock else LLMClient(
        config["judge"], config["judge"]["model"], args.results / "logs" / "table3_judge.jsonl", int(config.get("seed", 42))
    )
    judge_model = "MOCK-JUDGE" if args.mock else str(config["judge"]["model"])
    raw_name = "table3_mock_raw.jsonl" if args.mock else f"table3_{slug(judge_model)}_raw.jsonl"
    raw_path = args.results / raw_name
    completed = {
        row["dialogue_id"]
        for row in read_jsonl(raw_path)
        if row.get("status") == "ok" and row.get("judge_model") == judge_model
    }
    rng = random.Random(int(config.get("seed", 42)))
    for dialogue_id in tqdm(common, desc="Table 3 judge"):
        order = METHODS.copy(); rng.shuffle(order)
        if dialogue_id in completed:
            continue
        mapping = dict(zip(LETTERS, order))
        responses = {letter: outputs[method][dialogue_id]["prediction"] for letter, method in mapping.items()}
        context = outputs[METHODS[0]][dialogue_id]["context"]
        error = None
        parse_retries = 1 if args.mock else int(config["judge"].get("parse_retries", 3))
        for attempt in range(1, parse_retries + 1):
            try:
                prompt = judge_prompt(context, responses)
                if attempt > 1:
                    prompt += "\nYour previous format was invalid. Output exactly five score lines, each containing exactly five ratings from 1 to 5."
                result = judge.complete(prompt, temperature=0.0, max_tokens=1200, tag="table3_local_llm_judge")
                scores = parse_scores(result.text)
                row = {
                    "dialogue_id": dialogue_id,
                    "mapping": mapping,
                    "scores": scores,
                    "raw": result.text,
                    "status": "ok",
                    "is_mock": args.mock,
                    "judge_model": judge_model,
                    "judge_type": "mock" if args.mock else config["judge"].get("type", "local-llm-approximation"),
                    "parse_attempts": attempt,
                }
                break
            except Exception as exc:
                error = exc
        else:
            row = {
                "dialogue_id": dialogue_id,
                "mapping": mapping,
                "status": "error",
                "error": repr(error),
                "is_mock": args.mock,
                "judge_model": judge_model,
                "judge_type": "mock" if args.mock else config["judge"].get("type", "local-llm-approximation"),
                "parse_attempts": parse_retries,
            }
        append_jsonl(raw_path, row)

    if args.mock:
        print(raw_path)
        print("MOCK output is only a pipeline test and is not aggregated as an experiment.")
        return
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in read_jsonl(raw_path):
        if row.get("status") != "ok" or row.get("is_mock"):
            continue
        for dimension, values in row["scores"].items():
            for letter, value in zip(LETTERS, values):
                totals[(row["mapping"][letter], dimension)].append(float(value))
    output = args.results / "table3_local.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "fluency", "identification", "comforting", "suggestion", "overall", "n", "source", "judge_model"])
        for method in METHODS:
            values = [totals[(method, dimension)] for dimension in DIMENSIONS]
            n = min((len(item) for item in values), default=0)
            means = [sum(item) / len(item) if item else float("nan") for item in values]
            writer.writerow([method, *[f"{value:.4f}" for value in means], n, "local-llm-judge-approximate", judge_model])
    print(output)


if __name__ == "__main__":
    main()
