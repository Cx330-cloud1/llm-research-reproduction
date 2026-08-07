from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(module: str, *arguments: str) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Table 1/3/4 and prepare Table 2 human evaluation")
    parser.add_argument("--config", default="config.full.json")
    parser.add_argument("--results", default="results")
    parser.add_argument("--skip-table3", action="store_true")
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--mock", action="store_true", help="Run an offline plumbing test; outputs are excluded from experiments")
    args = parser.parse_args()
    config = str(Path(args.config))
    with Path(args.config).open(encoding="utf-8") as handle:
        config_data = json.load(handle)
    evaluation_backbone = config_data.get("evaluation_backbone") or next(iter(config_data["models"]))
    effective_results = str(Path(args.results) / "mock-not-for-reporting") if args.mock else args.results
    result_args = ["--results", effective_results]
    generation_limit = [] if args.max_targets is None else ["--max-targets", str(args.max_targets)]
    mock_flag = ["--mock"] if args.mock else []
    allow_mock = ["--allow-mock"] if args.mock else []

    run("scripts.run_generations", "--config", config, "--table", "table1", "--output-root", args.results, *generation_limit, *mock_flag)
    run("scripts.build_table1", *result_args, *allow_mock)
    run("scripts.table2_human", "prepare", *result_args, "--backbone", evaluation_backbone, *allow_mock)
    if not args.skip_table3:
        run("scripts.run_table3", "--config", config, *result_args, "--backbone", evaluation_backbone, *mock_flag)
    run(
        "scripts.run_generations", "--config", config, "--table", "table4", "--models", evaluation_backbone,
        "--output-root", args.results, *generation_limit, *mock_flag,
    )
    run("scripts.build_table4", *result_args, "--backbone", evaluation_backbone, *allow_mock)
    run("scripts.compare_all", *result_args)
    print("\nTable 1/3/4 pipeline finished. Table 2 still requires three human annotators.")


if __name__ == "__main__":
    main()
