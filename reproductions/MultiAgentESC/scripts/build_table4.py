from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from .common import read_jsonl
from .metrics import automatic_metrics


METRICS = ("D-1", "D-2", "B-1", "B-2", "R-L")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--paper", type=Path, default=Path("data/paper/table4.csv"))
    parser.add_argument("--backbone", default="Qwen2.5-32b")
    parser.add_argument("--allow-mock", action="store_true")
    args = parser.parse_args()
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in (args.results / "generations").glob("*/*.jsonl"):
        for row in read_jsonl(path):
            if row.get("backbone") != args.backbone or row.get("method") != "MultiAgentESC (Ours)":
                continue
            if row.get("status") == "ok" and (not row.get("is_mock") or args.allow_mock):
                groups[row.get("variant", "full")].append(row)
    if not groups:
        raise SystemExit("No Table 4 generations found.")
    display = {
        "full": "MultiAgentESC (Ours)",
        "w/o dialogue analysis": "w/o dialogue analysis",
        "w/o experience": "w/o experience",
        "w/o group discussion": "w/o group discussion",
    }
    paper = {}
    with args.paper.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            paper[row["variant"]] = row
    output = args.results / "table4_local_and_comparison.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        fields = ["variant", "n"] + [part for metric in METRICS for part in (f"paper_{metric}", f"local_{metric}", f"abs_diff_{metric}")]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, rows in sorted(groups.items()):
            name = display.get(variant, variant)
            values = automatic_metrics([r["prediction"] for r in rows], [r["reference"] for r in rows])
            source = paper.get(name, {})
            record = {"variant": name, "n": len(rows)}
            for metric in METRICS:
                p = float(source[metric]) if source.get(metric) else None
                record[f"paper_{metric}"] = "" if p is None else f"{p:.4f}"
                record[f"local_{metric}"] = f"{values[metric]:.4f}"
                record[f"abs_diff_{metric}"] = "" if p is None else f"{abs(values[metric]-p):.4f}"
            writer.writerow(record)
    print(output)


if __name__ == "__main__":
    main()
