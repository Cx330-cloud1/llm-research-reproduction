from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # keeps mock/offline validation usable before installation
    def tqdm(iterable, **_):
        return iterable

from .common import LLMClient, MockClient, append_jsonl, completed_ids, load_config, resolve_path, slug
from .dataset_utils import Retriever, build_experiences, extract_targets, few_shot_examples, load_dialogues
from .methods import ALL_METHODS, MethodRunner


class NullRetriever:
    def search(self, _query: str, _k: int = 10) -> list[dict[str, str]]:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample-level outputs for Table 1 or Table 4")
    parser.add_argument("--config", type=Path, default=Path("config.smoke.json"))
    parser.add_argument("--table", choices=("table1", "table4"), default="table1")
    parser.add_argument("--models", nargs="*", help="Backbone labels from config; default: all")
    parser.add_argument("--methods", nargs="*", choices=ALL_METHODS, help="Table 1 methods; default: all")
    parser.add_argument("--variants", nargs="*", default=["full", "w/o dialogue analysis", "w/o experience", "w/o group discussion"])
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--mock", action="store_true", help="Offline pipeline test only; never use as an experiment")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dialogues = load_dialogues(resolve_path(config, config["dataset"]))
    targets = extract_targets(
        dialogues,
        int(config["dialogue_limit"]),
        str(config.get("selection", "first")),
        int(config.get("seed", 42)),
    )
    if args.max_targets is not None:
        targets = targets[:args.max_targets]
    examples = few_shot_examples(dialogues)
    experiences = build_experiences(dialogues, start=100)

    methods = args.methods or ALL_METHODS
    variants = ["full"] if args.table == "table1" else args.variants
    if args.table == "table4":
        methods = ["MultiAgentESC (Ours)"]
    model_labels = args.models or list(config["models"])
    output_root = args.output_root
    if args.mock:
        output_root = output_root / "mock-not-for-reporting"

    needs_retrieval = methods == ["MultiAgentESC (Ours)"] or "MultiAgentESC (Ours)" in methods
    retriever = Retriever(
        experiences,
        config["retrieval"]["backend"],
        config["retrieval"]["model"],
    ) if needs_retrieval else NullRetriever()

    manifest = {
        "created_unix": time.time(),
        "config": str(args.config.resolve()),
        "table": args.table,
        "models": model_labels,
        "methods": methods,
        "variants": variants,
        "targets": len(targets),
        "mock": args.mock,
        "selection": config.get("selection", "first"),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{args.table}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for model_label in model_labels:
        if model_label not in config["models"]:
            raise ValueError(f"model label not in config: {model_label}")
        model_name = config["models"][model_label]
        log_path = output_root / "logs" / f"{slug(model_label)}.jsonl"
        client = MockClient(model_name) if args.mock else LLMClient(config["provider"], model_name, log_path, int(config.get("seed", 42)))
        runner = MethodRunner(client, config, examples, retriever)
        for method in methods:
            for variant in variants:
                suffix = "" if variant == "full" else f"__{slug(variant)}"
                output = output_root / "generations" / slug(model_label) / f"{slug(method)}{suffix}.jsonl"
                if args.overwrite and output.exists():
                    output.unlink()
                done = completed_ids(output)
                progress = tqdm(targets, desc=f"{model_label} | {method} | {variant}")
                for target in progress:
                    if target.target_id in done:
                        continue
                    base = target.as_dict()
                    try:
                        generated = runner.run(method, base, variant)
                        row = {
                            **base,
                            **generated,
                            "backbone": model_label,
                            "model_name": model_name,
                            "method": method,
                            "status": "ok",
                            "is_mock": args.mock,
                            "seed": config.get("seed", 42),
                            "temperature": config["generation"].get("temperature", 0.0),
                        }
                    except Exception as exc:
                        row = {
                            **base,
                            "backbone": model_label,
                            "model_name": model_name,
                            "method": method,
                            "variant": variant,
                            "status": "error",
                            "error": repr(exc),
                            "is_mock": args.mock,
                        }
                    append_jsonl(output, row)
    print(output_root.resolve())


if __name__ == "__main__":
    main()
