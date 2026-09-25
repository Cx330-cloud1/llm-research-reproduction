from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .io_utils import (
        read_jsonl,
        sha256_json,
        write_jsonl_atomic,
    )
except ImportError:
    from scripts.io_utils import (
        read_jsonl,
        sha256_json,
        write_jsonl_atomic,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLLECTION_SPLITS = {
    "dev_cal": "valid",
    "pilot100": "test",
    "smoke20": "test",
}


def selected_collections(collection: str) -> list[str]:
    """Return the collections selected for model inference."""
    if collection == "all":
        return list(COLLECTION_SPLITS)

    if collection not in COLLECTION_SPLITS:
        raise ValueError(f"unknown collection: {collection}")

    return [collection]


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def context_signature(model_context: Mapping[str, Any]) -> str:
    """Return a deterministic signature for all model-visible inputs."""
    return sha256_json(
        {
            "dialogue_history": model_context["dialogue_history"],
            "strategy_history": str(
                model_context["strategy_history"]
            ),
            "speaker_turn": str(model_context["speaker_turn"]),
        }
    )


def join_preprocessed(
    base_rows: Sequence[Mapping[str, Any]],
    preprocessed_rows: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """
    Join manifest rows to EmoDynamiX-preprocessed rows.

    Ambiguous and missing matches fail closed.
    """
    preprocessed_by_signature: dict[
        str,
        Mapping[str, Any],
    ] = {}

    for row in preprocessed_rows:
        signature = context_signature(row)

        if signature in preprocessed_by_signature:
            raise ValueError(
                f"duplicate preprocessed signature: {signature}"
            )

        preprocessed_by_signature[signature] = row

    joined: list[
        tuple[Mapping[str, Any], Mapping[str, Any]]
    ] = []
    seen_base_signatures: set[str] = set()

    for base_row in base_rows:
        sample_id = str(base_row.get("sample_id", "<unknown>"))
        signature = context_signature(base_row["model_context"])

        if signature in seen_base_signatures:
            raise ValueError(
                f"duplicate base signature: {signature}"
            )

        seen_base_signatures.add(signature)

        if signature not in preprocessed_by_signature:
            raise ValueError(
                "missing preprocessed signature "
                f"for sample_id={sample_id}: {signature}"
            )

        joined.append(
            (
                base_row,
                preprocessed_by_signature[signature],
            )
        )

    return joined


def verified_label_order(
    strategy2id: Mapping[str, int],
    expected: Sequence[str],
) -> list[str]:
    """Verify that checkpoint IDs match the frozen label order."""
    actual = [
        label
        for label, _strategy_id in sorted(
            strategy2id.items(),
            key=lambda item: int(item[1]),
        )
    ]
    expected_list = list(expected)

    if actual != expected_list:
        raise ValueError(
            "label order mismatch: "
            f"expected={expected_list}, actual={actual}"
        )

    return actual


def resolve_checkpoint_file(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()

    if checkpoint.is_file():
        return checkpoint

    if checkpoint.is_dir():
        candidates = [
            checkpoint / "pytorch_model.bin",
            checkpoint / "model.pt",
            checkpoint / "model.pth",
        ]

        for candidate in candidates:
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        f"checkpoint file not found: {checkpoint}"
    )


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle_rows(path: str | Path) -> list[dict[str, Any]]:
    # Kept inside the runtime path so --help remains CPU-safe.
    import pickle

    with Path(path).open("rb") as handle:
        value = pickle.load(handle)

    if not isinstance(value, list):
        raise TypeError(
            f"preprocessed pickle must contain a list: {path}"
        )

    if not all(isinstance(row, dict) for row in value):
        raise TypeError(
            f"preprocessed pickle contains non-dict rows: {path}"
        )

    return value


def write_json_atomic(
    path: str | Path,
    value: Mapping[str, Any],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )
    temporary_path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def build_output_row(
    sample_id: str,
    logits: Sequence[float],
    label_order: Sequence[str],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    raw_logits = [float(value) for value in logits]

    if len(raw_logits) != 8:
        raise ValueError(
            f"expected 8 strategy logits for {sample_id}, "
            f"got {len(raw_logits)}"
        )

    if not all(math.isfinite(value) for value in raw_logits):
        raise ValueError(
            f"non-finite strategy logits for {sample_id}"
        )

    return {
        "sample_id": sample_id,
        "strategy_logits": raw_logits,
        "label_order": list(label_order),
        "status": "ok",
        "checkpoint_sha256": checkpoint_sha256,
    }


def collate_preprocessed(
    rows: Sequence[Mapping[str, Any]],
    torch_module: Any,
) -> dict[str, Any]:
    erc_tensors = [
        torch_module.as_tensor(row["erc_logits"])
        for row in rows
    ]

    return {
        "dialogue_history": [
            row["dialogue_history"] for row in rows
        ],
        "strategy_history": [
            row["strategy_history"] for row in rows
        ],
        "speaker_turn": [
            row["speaker_turn"] for row in rows
        ],
        "parsed_dialogue": [
            row["parsed_dialogue"] for row in rows
        ],
        "erc_logits": torch_module.cat(
            erc_tensors,
            dim=0,
        ),
    }


def output_is_complete(
    path: Path,
    base_rows: Sequence[Mapping[str, Any]],
    label_order: Sequence[str],
    checkpoint_sha256: str,
) -> bool:
    if not path.is_file():
        return False

    rows = read_jsonl(path)

    if len(rows) != len(base_rows):
        raise ValueError(
            f"incomplete existing output: {path}"
        )

    expected_ids = [
        str(row["sample_id"]) for row in base_rows
    ]
    actual_ids = [
        str(row.get("sample_id")) for row in rows
    ]

    if actual_ids != expected_ids:
        raise ValueError(
            f"existing output sample_id mismatch: {path}"
        )

    for row in rows:
        if row.get("status") != "ok":
            raise ValueError(
                f"existing output contains failure rows: {path}"
            )

        if row.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError(
                f"existing output checkpoint mismatch: {path}"
            )

        if row.get("label_order") != list(label_order):
            raise ValueError(
                f"existing output label order mismatch: {path}"
            )

        logits = row.get("strategy_logits")

        if not isinstance(logits, list) or len(logits) != 8:
            raise ValueError(
                f"existing output has invalid logits: {path}"
            )

    return True


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export eight-class raw EmoDynamiX strategy logits."
        )
    )
    parser.add_argument(
        "--emodynamix-root",
        required=True,
        help="Path to the EmoDynamiX-v2 repository.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Checkpoint file or a checkpoint directory "
            "containing pytorch_model.bin."
        ),
    )
    parser.add_argument(
        "--strategies",
        required=True,
        help="Path to data/esconv/strategies.json.",
    )
    parser.add_argument(
        "--valid-pkl",
        required=True,
        help="Path to the preprocessed valid.pkl.",
    )
    parser.add_argument(
        "--test-pkl",
        required=True,
        help="Path to the preprocessed test.pkl.",
    )
    parser.add_argument(
        "--manifest-root",
        required=True,
        help="Path containing dev_cal, pilot100 and smoke20.",
    )
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT / "configs" / "experiment.json"
        ),
        help="Frozen experiment configuration.",
    )
    parser.add_argument(
        "--output-root",
        help="Destination for strategy_results.raw.jsonl.",
    )
    parser.add_argument(
        "--collection",
        choices=["smoke20", "pilot100", "dev_cal", "all"],
        default="all",
        help="Collection to export; default exports all collections.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size; default reproduces official test.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate assets and CUDA without loading the model.",
    )
    return parser


def load_model(
    emodynamix_root: Path,
    checkpoint_file: Path,
    torch_module: Any,
) -> Any:
    class ModelArgs:
        dataset = "esconv-preprocessed"
        exclude_others = 0
        erc_temperature = 0.5
        erc_mixed = 1
        hg_dim = 512

    root_string = str(emodynamix_root)

    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    previous_directory = Path.cwd()

    try:
        os.chdir(emodynamix_root)

        from modules.roberta import (
            RobertaHeterogeneousGraph,
        )

        model = RobertaHeterogeneousGraph(
            ModelArgs(),
            lightmode=True,
        )
        model.load(str(checkpoint_file))
        model.to(torch_module.device("cuda"))
        model.eval()
    finally:
        os.chdir(previous_directory)

    return model


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    if not args.preflight_only and not args.output_root:
        parser.error(
            "--output-root is required unless "
            "--preflight-only is used"
        )

    emodynamix_root = Path(
        args.emodynamix_root
    ).expanduser().resolve()
    strategies_path = Path(
        args.strategies
    ).expanduser().resolve()
    valid_pkl_path = Path(
        args.valid_pkl
    ).expanduser().resolve()
    test_pkl_path = Path(
        args.test_pkl
    ).expanduser().resolve()
    manifest_root = Path(
        args.manifest_root
    ).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    try:
        checkpoint_file = resolve_checkpoint_file(
            args.checkpoint
        )
    except FileNotFoundError as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "missing_assets": [str(error)],
                },
                ensure_ascii=False,
            )
        )
        return 2

    manifest_paths = {
        collection: (
            manifest_root
            / collection
            / "base_manifest.jsonl"
        )
        for collection in COLLECTION_SPLITS
    }

    required_files = {
        "checkpoint": checkpoint_file,
        "strategies": strategies_path,
        "valid_pkl": valid_pkl_path,
        "test_pkl": test_pkl_path,
        "config": config_path,
        **{
            f"manifest_{collection}": path
            for collection, path in manifest_paths.items()
        },
    }

    missing_assets = []

    if not emodynamix_root.is_dir():
        missing_assets.append(str(emodynamix_root))

    for path in required_files.values():
        if not path.is_file():
            missing_assets.append(str(path))

    if missing_assets:
        print(
            json.dumps(
                {
                    "status": "error",
                    "missing_assets": missing_assets,
                },
                ensure_ascii=False,
            )
        )
        return 2

    experiment_config = load_json(config_path)
    strategy2id = load_json(strategies_path)

    label_order = verified_label_order(
        strategy2id,
        experiment_config["label_order"],
    )

    # Imported only after argparse and path validation.
    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_name = (
        torch.cuda.get_device_name(0)
        if cuda_available
        else None
    )

    asset_sha256 = {
        name: sha256_file(path)
        for name, path in required_files.items()
    }

    preflight = {
        "status": "ok" if cuda_available else "error",
        "cuda_available": cuda_available,
        "cuda_device_name": device_name,
        "label_order": label_order,
        "asset_sha256": asset_sha256,
        "paths": {
            "emodynamix_root": str(emodynamix_root),
            **{
                name: str(path)
                for name, path in required_files.items()
            },
        },
    }

    print(
        json.dumps(
            preflight,
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    if not cuda_available:
        return 2

    if args.preflight_only:
        return 0

    output_root = Path(
        args.output_root
    ).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    write_json_atomic(
        output_root / "preflight.json",
        preflight,
    )

    collections = selected_collections(args.collection)

    base_rows_by_collection = {
        collection: read_jsonl(manifest_paths[collection])
        for collection in collections
    }

    output_paths = {
        collection: (
            output_root
            / collection
            / "strategy_results.raw.jsonl"
        )
        for collection in collections
    }

    pending_collections = []

    for collection, base_rows in (
        base_rows_by_collection.items()
    ):
        output_path = output_paths[collection]

        if output_is_complete(
            output_path,
            base_rows,
            label_order,
            asset_sha256["checkpoint"],
        ):
            print(
                json.dumps(
                    {
                        "collection": collection,
                        "status": "skipped_complete",
                        "rows": len(base_rows),
                    }
                ),
                flush=True,
            )
        else:
            pending_collections.append(collection)

    if not pending_collections:
        return 0

    required_splits = {
        COLLECTION_SPLITS[collection]
        for collection in pending_collections
    }
    preprocessed_by_split = {}

    if "valid" in required_splits:
        preprocessed_by_split["valid"] = load_pickle_rows(
            valid_pkl_path
        )

    if "test" in required_splits:
        preprocessed_by_split["test"] = load_pickle_rows(
            test_pkl_path
        )

    joined_by_collection = {}

    for collection in pending_collections:
        split = COLLECTION_SPLITS[collection]
        joined_by_collection[collection] = join_preprocessed(
            base_rows_by_collection[collection],
            preprocessed_by_split[split],
        )

    model = load_model(
        emodynamix_root,
        checkpoint_file,
        torch,
    )

    completed_counts = {}

    for collection in pending_collections:
        joined = joined_by_collection[collection]
        output_rows = []

        for start in range(0, len(joined), args.batch_size):
            batch_pairs = joined[
                start : start + args.batch_size
            ]
            prepared_rows = [
                pair[1] for pair in batch_pairs
            ]
            model_batch = collate_preprocessed(
                prepared_rows,
                torch,
            )

            with torch.no_grad():
                outputs = model(model_batch)

            batch_logits = (
                outputs["logits"]
                .detach()
                .cpu()
                .tolist()
            )

            if len(batch_logits) != len(batch_pairs):
                raise ValueError(
                    "model output batch size mismatch: "
                    f"expected={len(batch_pairs)}, "
                    f"actual={len(batch_logits)}"
                )

            for pair, logits in zip(
                batch_pairs,
                batch_logits,
                strict=True,
            ):
                base_row = pair[0]
                output_rows.append(
                    build_output_row(
                        sample_id=str(
                            base_row["sample_id"]
                        ),
                        logits=logits,
                        label_order=label_order,
                        checkpoint_sha256=(
                            asset_sha256["checkpoint"]
                        ),
                    )
                )

        write_jsonl_atomic(
            output_paths[collection],
            output_rows,
        )
        completed_counts[collection] = len(output_rows)

        print(
            json.dumps(
                {
                    "collection": collection,
                    "status": "ok",
                    "rows": len(output_rows),
                    "output": str(
                        output_paths[collection]
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "completed": completed_counts,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())