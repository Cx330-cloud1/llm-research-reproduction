from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from .common import read_jsonl
from .metrics import automatic_metrics


METRICS = ("D-1", "D-2", "B-1", "B-2", "B-3", "F1", "R-L")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--paper", type=Path, default=Path("data/paper/table1.csv"))
    parser.add_argument("--allow-mock", action="store_true")
    args = parser.parse_args()
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in (args.results / "generations").glob("*/*.jsonl"):
        if "__w-o-" in path.stem:
            continue
        for row in read_jsonl(path):
            if row.get("status") != "ok" or (row.get("is_mock") and not args.allow_mock):
                continue
            if row.get("variant", "full") == "full":
                groups[(row["backbone"], row["method"])].append(row)
    if not groups:
        raise SystemExit("No eligible real generations found. Run scripts.run_generations first.")

    args.results.mkdir(parents=True, exist_ok=True)
    local_path = args.results / "table1_local.csv"
    local_rows = []
    with local_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "backbone", *METRICS, "n", "source"])
        for (backbone, method), rows in sorted(groups.items()):
            values = automatic_metrics([r["prediction"] for r in rows], [r["reference"] for r in rows])
            record = {"method": method, "backbone": backbone, **values, "n": len(rows), "source": "local-measured"}
            local_rows.append(record)
            writer.writerow([method, backbone, *[f"{values[m]:.4f}" for m in METRICS], len(rows), "local-measured"])

    paper = {}
    with args.paper.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            paper[(row["backbone"], row["method"])] = row
    compare_path = args.results / "table1_comparison.csv"
    with compare_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["backbone", "method", "n"] + [part for metric in METRICS for part in (f"paper_{metric}", f"local_{metric}", f"abs_diff_{metric}")]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in local_rows:
            source = paper.get((row["backbone"], row["method"]), {})
            output = {"backbone": row["backbone"], "method": row["method"], "n": row["n"]}
            for metric in METRICS:
                p = float(source[metric]) if source.get(metric) else None
                output[f"paper_{metric}"] = "" if p is None else f"{p:.4f}"
                output[f"local_{metric}"] = f"{row[metric]:.4f}"
                output[f"abs_diff_{metric}"] = "" if p is None else f"{abs(row[metric]-p):.4f}"
            writer.writerow(output)
    print(local_path)
    print(compare_path)


if __name__ == "__main__":
    main()

