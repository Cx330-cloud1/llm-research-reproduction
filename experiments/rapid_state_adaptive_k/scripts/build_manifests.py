from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .io_utils import (
    sha256_json,
    write_jsonl_atomic,
)


def normalize_text(value: Any) -> str:
    return " ".join(
        str(value).split()
    )


def official_split_indices(
    dialogues: list[dict[str, Any]],
    seed: int = 13,
) -> dict[str, list[int]]:
    indexed = list(
        range(len(dialogues))
    )
    random.Random(seed).shuffle(
        indexed
    )

    valid_size = int(
        0.15 * len(indexed)
    )
    test_size = int(
        0.15 * len(indexed)
    )

    return {
        "valid": indexed[:valid_size],
        "test": indexed[
            valid_size:
            valid_size + test_size
        ],
        "train": indexed[
            valid_size + test_size:
        ],
    }


def extract_strategy_targets(
    dialogue: dict[str, Any],
    dialogue_index: int,
    split: str,
    label_order: list[str],
) -> list[
    tuple[
        dict[str, Any],
        dict[str, Any],
    ]
]:
    turns = dialogue["dialog"]

    utterances = ["<START>"]
    speakers = ["None"]
    strategies = [-1]

    pairs: list[
        tuple[
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []

    for raw_index, turn in enumerate(
        turns
    ):
        text = normalize_text(
            turn.get("content", "")
        )
        annotation = (
            turn.get("annotation")
            or {}
        )
        strategy = annotation.get(
            "strategy"
        )

        if (
            turn.get("speaker")
            == "supporter"
            and strategy
        ):
            if strategy not in label_order:
                raise ValueError(
                    "unknown strategy: "
                    f"{strategy}"
                )

            model_index = raw_index + 1
            start = max(
                0,
                model_index - 5,
            )

            model_context = {
                "dialogue_history": (
                    " </s> ".join(
                        utterances[
                            start:model_index
                        ]
                    )
                ),
                "strategy_history": str(
                    strategies[
                        start:model_index
                    ]
                ),
                "speaker_turn": " ".join(
                    speakers[
                        start:model_index
                    ]
                ),
            }

            generation_context = (
                "\n".join(
                    (
                        "User"
                        if item["speaker"]
                        == "seeker"
                        else "Assistant"
                    )
                    + ": "
                    + normalize_text(
                        item.get(
                            "content",
                            "",
                        )
                    )
                    for item in turns[
                        :raw_index
                    ]
                )
            )

            sample_id = (
                f"{split}"
                f"-d{dialogue_index:04d}"
                f"-u{raw_index:03d}"
            )

            base_without_hash = {
                "schema_version": "1.0",
                "sample_id": sample_id,
                "split": split,
                "dialogue_id": (
                    f"d{dialogue_index:04d}"
                ),
                "target_turn_id": (
                    f"u{raw_index:03d}"
                ),
                "visible_turn_ids": [
                    f"u{i:03d}"
                    for i in range(
                        raw_index
                    )
                ],
                "model_context": (
                    model_context
                ),
                "generation_context": (
                    generation_context
                ),
            }

            base = {
                **base_without_hash,
                "context_hash": (
                    sha256_json(
                        base_without_hash
                    )
                ),
            }

            sidecar = {
                "sample_id": sample_id,
                "gold_strategy": (
                    strategy
                ),
                "reference_response": (
                    text
                ),
                "original_dialogue_index": (
                    dialogue_index
                ),
                "target_turn_index": (
                    raw_index
                ),
            }

            pairs.append(
                (base, sidecar)
            )

        utterances.append(
            text
        )
        speakers.append(
            str(
                turn.get(
                    "speaker",
                    "",
                )
            )
        )
        strategies.append(
            label_order.index(strategy)
            if strategy in label_order
            else -1
        )

    return pairs


def last_target_per_dialogue(
    pairs: list[
        tuple[
            dict[str, Any],
            dict[str, Any],
        ]
    ],
) -> list[
    tuple[
        dict[str, Any],
        dict[str, Any],
    ]
]:
    latest: dict[
        str,
        tuple[
            dict[str, Any],
            dict[str, Any],
        ],
    ] = {}

    for base, sidecar in pairs:
        dialogue_id = str(
            base["dialogue_id"]
        )
        previous = latest.get(
            dialogue_id
        )

        if (
            previous is None
            or int(
                sidecar[
                    "target_turn_index"
                ]
            )
            > int(
                previous[1][
                    "target_turn_index"
                ]
            )
        ):
            latest[dialogue_id] = (
                base,
                sidecar,
            )

    return [
        latest[dialogue_id]
        for dialogue_id in sorted(
            latest
        )
    ]


def proportional_quotas(
    counts: dict[str, int],
    size: int,
) -> dict[str, int]:
    total = sum(
        counts.values()
    )

    if size < 0:
        raise ValueError(
            "sample size must be "
            "non-negative"
        )

    if size > total:
        raise ValueError(
            f"sample size {size} exceeds "
            f"population {total}"
        )

    if total == 0:
        if size == 0:
            return {
                key: 0
                for key in counts
            }

        raise ValueError(
            "cannot sample from an "
            "empty population"
        )

    exact = {
        key: (
            counts[key]
            * size
            / total
        )
        for key in counts
    }

    quotas = {
        key: min(
            counts[key],
            int(exact[key]),
        )
        for key in counts
    }

    remaining = (
        size
        - sum(quotas.values())
    )

    order = sorted(
        counts,
        key=lambda key: (
            -(
                exact[key]
                - int(exact[key])
            ),
            key,
        ),
    )

    for key in order:
        if remaining == 0:
            break

        if quotas[key] < counts[key]:
            quotas[key] += 1
            remaining -= 1

    if remaining:
        raise ValueError(
            "unable to allocate "
            "stratified quotas"
        )

    return quotas


def stratified_sample(
    rows: list[dict[str, Any]],
    size: int,
    seed: int,
    label_key: str = "gold_strategy",
) -> list[dict[str, Any]]:
    groups: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for row in rows:
        if label_key not in row:
            raise ValueError(
                "missing stratification "
                f"field: {label_key}"
            )

        label = str(
            row[label_key]
        )
        groups.setdefault(
            label,
            [],
        ).append(row)

    counts = {
        label: len(group)
        for label, group in (
            groups.items()
        )
    }

    quotas = proportional_quotas(
        counts,
        size,
    )

    selected: list[
        dict[str, Any]
    ] = []

    for label in sorted(groups):
        group = sorted(
            groups[label],
            key=lambda row: str(
                row["sample_id"]
            ),
        )

        class_rng = random.Random(
            f"{seed}:{label}"
        )
        class_rng.shuffle(group)

        selected.extend(
            group[:quotas[label]]
        )

    return sorted(
        selected,
        key=lambda row: str(
            row["sample_id"]
        ),
    )


def _sorted_pairs(
    pairs: list[
        tuple[
            dict[str, Any],
            dict[str, Any],
        ]
    ],
) -> list[
    tuple[
        dict[str, Any],
        dict[str, Any],
    ]
]:
    return sorted(
        pairs,
        key=lambda pair: str(
            pair[0]["sample_id"]
        ),
    )


def _sampling_rows(
    pairs: list[
        tuple[
            dict[str, Any],
            dict[str, Any],
        ]
    ],
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": (
                base["sample_id"]
            ),
            "gold_strategy": (
                sidecar[
                    "gold_strategy"
                ]
            ),
            "base": base,
            "sidecar": sidecar,
        }
        for base, sidecar in pairs
    ]


def _pairs_from_sampling_rows(
    rows: list[dict[str, Any]],
) -> list[
    tuple[
        dict[str, Any],
        dict[str, Any],
    ]
]:
    return _sorted_pairs([
        (
            row["base"],
            row["sidecar"],
        )
        for row in rows
    ])


def build_collections(
    dialogues: list[dict[str, Any]],
    label_order: list[str],
    official_seed: int,
    sampling_seed: int,
    pilot_size: int,
    smoke_size: int,
) -> dict[
    str,
    list[
        tuple[
            dict[str, Any],
            dict[str, Any],
        ]
    ],
]:
    split = official_split_indices(
        dialogues,
        seed=official_seed,
    )

    dev_pairs: list[
        tuple[
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []

    for dialogue_index in split[
        "valid"
    ]:
        dev_pairs.extend(
            extract_strategy_targets(
                dialogue=dialogues[
                    dialogue_index
                ],
                dialogue_index=(
                    dialogue_index
                ),
                split="valid",
                label_order=label_order,
            )
        )

    test_pairs: list[
        tuple[
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []

    for dialogue_index in split[
        "test"
    ]:
        test_pairs.extend(
            extract_strategy_targets(
                dialogue=dialogues[
                    dialogue_index
                ],
                dialogue_index=(
                    dialogue_index
                ),
                split="test",
                label_order=label_order,
            )
        )

    test_last_pairs = (
        last_target_per_dialogue(
            test_pairs
        )
    )

    pilot_rows = stratified_sample(
        _sampling_rows(
            test_last_pairs
        ),
        size=pilot_size,
        seed=sampling_seed,
    )
    pilot_pairs = (
        _pairs_from_sampling_rows(
            pilot_rows
        )
    )

    smoke_rows = stratified_sample(
        _sampling_rows(
            pilot_pairs
        ),
        size=smoke_size,
        seed=sampling_seed,
    )
    smoke_pairs = (
        _pairs_from_sampling_rows(
            smoke_rows
        )
    )

    return {
        "dev_cal": _sorted_pairs(
            dev_pairs
        ),
        "pilot100": pilot_pairs,
        "smoke20": smoke_pairs,
    }


def write_collections(
    output_root: Any,
    collections: dict[
        str,
        list[
            tuple[
                dict[str, Any],
                dict[str, Any],
            ]
        ],
    ],
) -> None:
    root = Path(
        output_root
    )

    for (
        collection_name,
        pairs,
    ) in collections.items():
        ordered = _sorted_pairs(
            pairs
        )

        base_rows = [
            base
            for base, _ in ordered
        ]
        sidecar_rows = [
            sidecar
            for _, sidecar in ordered
        ]

        collection_root = (
            root
            / collection_name
        )

        write_jsonl_atomic(
            collection_root
            / "base_manifest.jsonl",
            base_rows,
        )
        write_jsonl_atomic(
            collection_root
            / "evaluation_sidecar.jsonl",
            sidecar_rows,
        )


def _read_json(
    path: Path,
) -> Any:
    with path.open(
        encoding="utf-8"
    ) as handle:
        return json.load(handle)


def _argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic "
            "Rapid State + Adaptive-k "
            "manifests."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to experiment.json."
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help=(
            "Directory for generated "
            "manifest collections."
        ),
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)

    config_path = Path(
        args.config
    ).resolve()
    config = _read_json(
        config_path
    )

    dataset_path = (
        config_path.parent
        / str(config["dataset"])
    ).resolve()

    dialogues = _read_json(
        dataset_path
    )

    if not isinstance(
        dialogues,
        list,
    ):
        raise ValueError(
            "ESConv dataset must be "
            "a JSON list"
        )

    collections = build_collections(
        dialogues=dialogues,
        label_order=list(
            config["label_order"]
        ),
        official_seed=int(
            config[
                "official_split_seed"
            ]
        ),
        sampling_seed=int(
            config["sampling_seed"]
        ),
        pilot_size=int(
            config["pilot_size"]
        ),
        smoke_size=int(
            config["smoke_size"]
        ),
    )

    write_collections(
        output_root=Path(
            args.output_root
        ),
        collections=collections,
    )

    split = official_split_indices(
        dialogues,
        seed=int(
            config[
                "official_split_seed"
            ]
        ),
    )

    summary = {
        "train_dialogues": len(
            split["train"]
        ),
        "valid_dialogues": len(
            split["valid"]
        ),
        "test_dialogues": len(
            split["test"]
        ),
        "dev_cal": len(
            collections["dev_cal"]
        ),
        "pilot100": len(
            collections["pilot100"]
        ),
        "smoke20": len(
            collections["smoke20"]
        ),
    }

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )