import json
from collections import Counter
from pathlib import Path

from scripts.build_manifests import (
    build_collections,
    extract_strategy_targets,
    last_target_per_dialogue,
    main,
    official_split_indices,
    proportional_quotas,
    stratified_sample,
    write_collections,
)
from scripts.io_utils import read_jsonl


def _dialogue(index: int) -> dict:
    return {
        "dialog": [
            {
                "speaker": "seeker",
                "content": f"u{index}",
                "annotation": {},
            },
            {
                "speaker": "supporter",
                "content": "a1",
                "annotation": {
                    "strategy": "Question",
                },
            },
            {
                "speaker": "supporter",
                "content": "a2",
                "annotation": {
                    "strategy": "Information",
                },
            },
        ]
    }


def test_official_dialogue_split_is_910_195_195_and_disjoint():
    split = official_split_indices(
        [
            _dialogue(i)
            for i in range(1300)
        ],
        seed=13,
    )

    assert {
        name: len(ids)
        for name, ids in split.items()
    } == {
        "train": 910,
        "valid": 195,
        "test": 195,
    }

    assert not (
        set(split["train"])
        & set(split["valid"])
    )
    assert not (
        set(split["train"])
        & set(split["test"])
    )
    assert not (
        set(split["valid"])
        & set(split["test"])
    )


def test_consecutive_supporter_turns_are_separate_targets():
    pairs = extract_strategy_targets(
        _dialogue(0),
        dialogue_index=0,
        split="valid",
        label_order=[
            "Question",
            "Information",
        ],
    )

    assert [
        base["target_turn_id"]
        for base, _ in pairs
    ] == [
        "u001",
        "u002",
    ]

    assert [
        sidecar["gold_strategy"]
        for _, sidecar in pairs
    ] == [
        "Question",
        "Information",
    ]


def test_early_model_window_contains_start_but_not_target():
    pairs = extract_strategy_targets(
        _dialogue(0),
        dialogue_index=0,
        split="valid",
        label_order=[
            "Question",
            "Information",
        ],
    )

    first, _ = pairs[0]

    assert (
        first["model_context"]["dialogue_history"]
        == "<START> </s> u0"
    )
    assert (
        "a1"
        not in first["model_context"]["dialogue_history"]
    )


def test_last_target_per_dialogue_keeps_final_strategy_turn():
    pairs = extract_strategy_targets(
        _dialogue(0),
        dialogue_index=0,
        split="test",
        label_order=[
            "Question",
            "Information",
        ],
    )

    selected = last_target_per_dialogue(
        pairs
    )

    assert len(selected) == 1

    base, sidecar = selected[0]

    assert base["target_turn_id"] == "u002"
    assert (
        sidecar["gold_strategy"]
        == "Information"
    )


def test_proportional_quotas_use_largest_remainders():
    quotas = proportional_quotas(
        {
            "A": 5,
            "B": 3,
            "C": 2,
        },
        size=5,
    )

    assert quotas == {
        "A": 3,
        "B": 1,
        "C": 1,
    }


def test_stratified_sample_is_deterministic_and_proportional():
    rows = [
        *[
            {
                "sample_id": f"a{i}",
                "gold_strategy": "A",
            }
            for i in range(5)
        ],
        *[
            {
                "sample_id": f"b{i}",
                "gold_strategy": "B",
            }
            for i in range(3)
        ],
        *[
            {
                "sample_id": f"c{i}",
                "gold_strategy": "C",
            }
            for i in range(2)
        ],
    ]

    first = stratified_sample(
        rows,
        size=5,
        seed=42,
    )
    second = stratified_sample(
        rows,
        size=5,
        seed=42,
    )

    assert first == second

    assert [
        row["sample_id"]
        for row in first
    ] == sorted(
        row["sample_id"]
        for row in first
    )

    assert Counter(
        row["gold_strategy"]
        for row in first
    ) == {
        "A": 3,
        "B": 1,
        "C": 1,
    }

    smoke = stratified_sample(
        first,
        size=2,
        seed=42,
    )

    assert {
        row["sample_id"]
        for row in smoke
    }.issubset({
        row["sample_id"]
        for row in first
    })


def test_build_collections_uses_valid_targets_and_nested_test_samples():
    dialogues = [
        _dialogue(i)
        for i in range(40)
    ]

    collections = build_collections(
        dialogues=dialogues,
        label_order=[
            "Question",
            "Information",
        ],
        official_seed=13,
        sampling_seed=42,
        pilot_size=4,
        smoke_size=2,
    )

    assert len(
        collections["dev_cal"]
    ) == 12
    assert len(
        collections["pilot100"]
    ) == 4
    assert len(
        collections["smoke20"]
    ) == 2

    pilot_dialogues = {
        base["dialogue_id"]
        for base, _ in collections["pilot100"]
    }

    assert len(pilot_dialogues) == 4

    assert all(
        base["target_turn_id"] == "u002"
        for base, _ in collections["pilot100"]
    )

    pilot_ids = {
        base["sample_id"]
        for base, _ in collections["pilot100"]
    }
    smoke_ids = {
        base["sample_id"]
        for base, _ in collections["smoke20"]
    }

    assert smoke_ids.issubset(
        pilot_ids
    )


def test_write_collections_separates_base_and_evaluation_fields(
    tmp_path: Path,
):
    collections = build_collections(
        dialogues=[
            _dialogue(i)
            for i in range(40)
        ],
        label_order=[
            "Question",
            "Information",
        ],
        official_seed=13,
        sampling_seed=42,
        pilot_size=4,
        smoke_size=2,
    )

    write_collections(
        tmp_path,
        collections,
    )

    base_rows = read_jsonl(
        tmp_path
        / "pilot100"
        / "base_manifest.jsonl"
    )
    sidecar_rows = read_jsonl(
        tmp_path
        / "pilot100"
        / "evaluation_sidecar.jsonl"
    )

    assert {
        row["sample_id"]
        for row in base_rows
    } == {
        row["sample_id"]
        for row in sidecar_rows
    }

    assert all(
        "gold_strategy" not in row
        and "reference_response" not in row
        for row in base_rows
    )

    assert all(
        "gold_strategy" in row
        and "reference_response" in row
        for row in sidecar_rows
    )


def test_cli_resolves_dataset_relative_to_config(
    tmp_path: Path,
    capsys,
):
    data_root = tmp_path / "data"
    config_root = tmp_path / "config"
    output_root = tmp_path / "output"

    data_root.mkdir()
    config_root.mkdir()

    dataset_path = (
        data_root
        / "ESConv.json"
    )
    dataset_path.write_text(
        json.dumps(
            [
                _dialogue(i)
                for i in range(40)
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    config_path = (
        config_root
        / "experiment.json"
    )
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset": "../data/ESConv.json",
                "official_split_seed": 13,
                "sampling_seed": 42,
                "pilot_size": 4,
                "smoke_size": 2,
                "label_order": [
                    "Question",
                    "Information",
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exit_code = main([
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
    ])

    summary = json.loads(
        capsys.readouterr().out
    )

    assert exit_code == 0

    assert summary == {
        "train_dialogues": 28,
        "valid_dialogues": 6,
        "test_dialogues": 6,
        "dev_cal": 12,
        "pilot100": 4,
        "smoke20": 2,
    }

    assert len(
        read_jsonl(
            output_root
            / "pilot100"
            / "base_manifest.jsonl"
        )
    ) == 4