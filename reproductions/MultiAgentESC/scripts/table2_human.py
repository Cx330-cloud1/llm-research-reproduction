from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .common import read_jsonl
from .dataset_utils import final_target_per_dialogue
from .metrics import exact_sign_test


BASELINES = ["Zero-shot", "Few-shot CoT", "Self-consistency", "Self-Refine"]
OURS = "MultiAgentESC (Ours)"
DIMENSIONS = ["Fluency", "Identification", "Comforting", "Suggestion", "Overall"]


def load_method_rows(results: Path, backbone: str, allow_mock: bool = False) -> dict[str, dict[str, dict]]:
    methods: dict[str, list[dict]] = defaultdict(list)
    for path in (results / "generations").glob("*/*.jsonl"):
        for row in read_jsonl(path):
            if row.get("backbone") == backbone and row.get("status") == "ok" and (allow_mock or not row.get("is_mock")) and row.get("variant", "full") == "full":
                methods[row["method"]].append(row)
    return {method: final_target_per_dialogue(rows) for method, rows in methods.items()}


def prepare(args: argparse.Namespace) -> None:
    data = load_method_rows(args.results, args.backbone, args.allow_mock)
    missing = [method for method in [*BASELINES, OURS] if method not in data]
    if missing:
        raise SystemExit(f"Missing generations for: {missing}")
    common = sorted(set.intersection(*(set(data[method]) for method in [*BASELINES, OURS])))[:100]
    if len(common) < 100:
        print(f"WARNING: only {len(common)} common dialogues are available; paper used 100")
    rng = random.Random(args.seed)
    tasks, mapping = [], []
    item = 0
    for baseline in BASELINES:
        for dialogue_id in common:
            ours_row, base_row = data[OURS][dialogue_id], data[baseline][dialogue_id]
            for dimension in DIMENSIONS:
                item += 1
                ours_is_a = bool(rng.getrandbits(1))
                a = ours_row["prediction"] if ours_is_a else base_row["prediction"]
                b = base_row["prediction"] if ours_is_a else ours_row["prediction"]
                task_id = f"H{item:05d}"
                tasks.append({
                    "task_id": task_id,
                    "dialogue_id": dialogue_id,
                    "dimension": dimension,
                    "context": ours_row["context"],
                    "response_A": a,
                    "response_B": b,
                    "annotator_1": "",
                    "annotator_2": "",
                    "annotator_3": "",
                })
                mapping.append({"task_id": task_id, "baseline": baseline, "A": OURS if ours_is_a else baseline, "B": baseline if ours_is_a else OURS})
    args.results.mkdir(parents=True, exist_ok=True)
    form = args.results / "table2_blind_annotation_form.csv"
    with form.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tasks[0]))
        writer.writeheader(); writer.writerows(tasks)
    private = args.results / "table2_private_mapping.json"
    private.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(form)
    print(private)


def aggregate(args: argparse.Namespace) -> None:
    mapping = {row["task_id"]: row for row in json.loads(args.mapping.read_text(encoding="utf-8"))}
    totals: dict[tuple[str, str], Counter] = defaultdict(Counter)
    unresolved = 0
    with args.form.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            votes = [row.get(f"annotator_{i}", "").strip().upper() for i in range(1, 4)]
            if any(vote not in {"A", "B", "TIE"} for vote in votes):
                unresolved += 1
                continue
            counts = Counter(votes)
            choice, count = counts.most_common(1)[0]
            if count == 1:
                choice = "TIE"
            info = mapping[row["task_id"]]
            key = (info["baseline"], row["dimension"])
            if choice == "TIE":
                totals[key]["tie"] += 1
            elif info[choice] == OURS:
                totals[key]["win"] += 1
            else:
                totals[key]["lose"] += 1
    output = args.results / "table2_local.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["baseline", "dimension", "win", "lose", "tie", "p_value_sign_test", "n", "source"])
        for baseline in BASELINES:
            for dimension in DIMENSIONS:
                c = totals[(baseline, dimension)]
                n = c["win"] + c["lose"] + c["tie"]
                writer.writerow([baseline, dimension, c["win"], c["lose"], c["tie"], f"{exact_sign_test(c['win'], c['lose']):.6f}", n, "local-human-evaluation"])
    print(output)
    if unresolved:
        print(f"WARNING: {unresolved} rows were incomplete and excluded")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--results", type=Path, default=Path("results"))
    prep.add_argument("--backbone", default="Qwen2.5-32b")
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--allow-mock", action="store_true")
    agg = sub.add_parser("aggregate")
    agg.add_argument("--results", type=Path, default=Path("results"))
    agg.add_argument("--form", type=Path, default=Path("results/table2_blind_annotation_form.csv"))
    agg.add_argument("--mapping", type=Path, default=Path("results/table2_private_mapping.json"))
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else aggregate(args)


if __name__ == "__main__":
    main()
