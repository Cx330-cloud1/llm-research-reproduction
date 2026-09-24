import json
from pathlib import Path

import pytest

from scripts.io_utils import (
    write_jsonl_atomic,
)
from scripts.validate_manifests import (
    compare_manifest_hashes,
    main,
    rebuild_check,
    validate_all,
    validate_all_rows,
    validate_collection,
)


def _base_row(
    sample_id: str,
    dialogue_id: str = "d1",
    context_hash: str | None = None,
) -> dict:
    return {
        "sample_id": sample_id,
        "dialogue_id": dialogue_id,
        "target_turn_id": "u002",
        "visible_turn_ids": [
            "u000",
            "u001",
        ],
        "context_hash": (
            context_hash
            if context_hash is not None
            else f"hash-{sample_id}"
        ),
    }


def _sidecar_row(
    sample_id: str,
) -> dict:
    return {
        "sample_id": sample_id,
    }


def _collection_rows(
    count: int,
    prefix: str,
) -> list[dict]:
    return [
        {
            "sample_id": (
                f"{prefix}-s{i:03d}"
            ),
            "dialogue_id": (
                f"{prefix}-d{i:03d}"
            ),
        }
        for i in range(count)
    ]


def test_gold_fields_are_rejected_from_base_manifest():
    base = _base_row("s1")
    base["gold_strategy"] = (
        "Question"
    )

    with pytest.raises(
        ValueError,
        match=(
            "forbidden field: "
            "gold_strategy"
        ),
    ):
        validate_collection(
            [base],
            [_sidecar_row("s1")],
        )


def test_nested_reference_response_is_rejected():
    base = _base_row("s1")
    base["model_context"] = {
        "reference_response": (
            "leaked answer"
        ),
    }

    with pytest.raises(
        ValueError,
        match=(
            "forbidden field: "
            "reference_response"
        ),
    ):
        validate_collection(
            [base],
            [_sidecar_row("s1")],
        )


def test_base_and_sidecar_ids_must_match():
    with pytest.raises(
        ValueError,
        match=(
            "base/sidecar "
            "sample_id mismatch"
        ),
    ):
        validate_collection(
            [_base_row("s1")],
            [_sidecar_row("s2")],
        )


def test_duplicate_sample_ids_are_rejected():
    with pytest.raises(
        ValueError,
        match="duplicate sample_id",
    ):
        validate_collection(
            [
                _base_row(
                    "s1",
                    context_hash="hash-1",
                ),
                _base_row(
                    "s1",
                    context_hash="hash-2",
                ),
            ],
            [
                _sidecar_row("s1"),
                _sidecar_row("s1"),
            ],
        )


def test_duplicate_context_hashes_are_rejected():
    with pytest.raises(
        ValueError,
        match="duplicate context_hash",
    ):
        validate_collection(
            [
                _base_row(
                    "s1",
                    context_hash="same-hash",
                ),
                _base_row(
                    "s2",
                    context_hash="same-hash",
                ),
            ],
            [
                _sidecar_row("s1"),
                _sidecar_row("s2"),
            ],
        )


def test_visible_turns_must_precede_target():
    base = _base_row("s1")
    base["visible_turn_ids"] = [
        "u000",
        "u002",
    ]

    with pytest.raises(
        ValueError,
        match="future turn visible: s1",
    ):
        validate_collection(
            [base],
            [_sidecar_row("s1")],
        )


def test_pilot_and_dev_dialogues_must_be_disjoint():
    with pytest.raises(
        ValueError,
        match=(
            "dev_cal/pilot100 "
            "overlap"
        ),
    ):
        validate_all_rows(
            dev=[
                {
                    "sample_id": (
                        "valid-s1"
                    ),
                    "dialogue_id": "d1",
                }
            ],
            pilot=[
                {
                    "sample_id": (
                        "test-s1"
                    ),
                    "dialogue_id": "d1",
                }
            ],
            smoke=[],
        )


def test_pilot100_must_contain_exactly_100_rows():
    with pytest.raises(
        ValueError,
        match=(
            "pilot100 must contain "
            "100 rows"
        ),
    ):
        validate_all_rows(
            dev=[],
            pilot=_collection_rows(
                99,
                "pilot",
            ),
            smoke=_collection_rows(
                20,
                "smoke",
            ),
        )


def test_pilot100_must_use_100_distinct_dialogues():
    pilot = _collection_rows(
        100,
        "pilot",
    )
    pilot[-1]["dialogue_id"] = (
        pilot[0]["dialogue_id"]
    )

    with pytest.raises(
        ValueError,
        match=(
            "pilot100 must contain "
            "100 distinct dialogues"
        ),
    ):
        validate_all_rows(
            dev=[],
            pilot=pilot,
            smoke=_collection_rows(
                20,
                "smoke",
            ),
        )


def test_smoke20_must_contain_exactly_20_rows():
    with pytest.raises(
        ValueError,
        match=(
            "smoke20 must contain "
            "20 rows"
        ),
    ):
        validate_all_rows(
            dev=[],
            pilot=_collection_rows(
                100,
                "pilot",
            ),
            smoke=_collection_rows(
                19,
                "smoke",
            ),
        )


def test_smoke20_must_be_subset_of_pilot100():
    pilot = _collection_rows(
        100,
        "pilot",
    )
    smoke = [
        *pilot[:19],
        {
            "sample_id": (
                "outside-sample"
            ),
            "dialogue_id": (
                "outside-dialogue"
            ),
        },
    ]

    with pytest.raises(
        ValueError,
        match=(
            "smoke20 is not a subset "
            "of pilot100"
        ),
    ):
        validate_all_rows(
            dev=[],
            pilot=pilot,
            smoke=smoke,
        )


def _complete_base_row(
    sample_id: str,
    dialogue_id: str,
) -> dict:
    return {
        "sample_id": sample_id,
        "dialogue_id": dialogue_id,
        "target_turn_id": "u002",
        "visible_turn_ids": [
            "u000",
            "u001",
        ],
        "context_hash": (
            f"hash-{sample_id}"
        ),
    }


def _write_collection(
    root: Path,
    name: str,
    base_rows: list[dict],
) -> None:
    sidecar_rows = [
        {
            "sample_id": (
                row["sample_id"]
            ),
        }
        for row in base_rows
    ]

    write_jsonl_atomic(
        root
        / name
        / "base_manifest.jsonl",
        base_rows,
    )
    write_jsonl_atomic(
        root
        / name
        / "evaluation_sidecar.jsonl",
        sidecar_rows,
    )


def _synthetic_dialogue(
    index: int,
) -> dict:
    return {
        "dialog": [
            {
                "speaker": "seeker",
                "content": (
                    f"user-{index}"
                ),
                "annotation": {},
            },
            {
                "speaker": "supporter",
                "content": "answer-1",
                "annotation": {
                    "strategy": "Question",
                },
            },
            {
                "speaker": "supporter",
                "content": "answer-2",
                "annotation": {
                    "strategy": (
                        "Information"
                    ),
                },
            },
        ]
    }


def _write_rebuild_config(
    tmp_path: Path,
) -> Path:
    data_root = tmp_path / "data"
    config_root = (
        tmp_path
        / "configs"
    )

    data_root.mkdir(
        exist_ok=True
    )
    config_root.mkdir(
        exist_ok=True
    )

    (
        data_root
        / "ESConv.json"
    ).write_text(
        json.dumps(
            [
                _synthetic_dialogue(i)
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
                "dataset": (
                    "../data/ESConv.json"
                ),
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

    return config_path


def _write_valid_manifest_root(
    tmp_path: Path,
) -> Path:
    manifest_root = (
        tmp_path
        / "manifests"
    )

    dev = [
        _complete_base_row(
            f"valid-s{i:03d}",
            f"valid-d{i:03d}",
        )
        for i in range(10)
    ]
    pilot = [
        _complete_base_row(
            f"test-s{i:03d}",
            f"test-d{i:03d}",
        )
        for i in range(100)
    ]
    smoke = [
        dict(row)
        for row in pilot[:20]
    ]

    _write_collection(
        manifest_root,
        "dev_cal",
        dev,
    )
    _write_collection(
        manifest_root,
        "pilot100",
        pilot,
    )
    _write_collection(
        manifest_root,
        "smoke20",
        smoke,
    )

    return manifest_root


def test_validate_all_reads_complete_manifest_layout(
    tmp_path: Path,
):
    validate_all(
        _write_valid_manifest_root(
            tmp_path
        )
    )


def test_compare_manifest_hashes_rejects_difference():
    with pytest.raises(
        ValueError,
        match=(
            "deterministic rebuild "
            "mismatch"
        ),
    ):
        compare_manifest_hashes(
            {
                "a.jsonl": "hash-one",
            },
            {
                "a.jsonl": "hash-two",
            },
        )


def test_rebuild_check_produces_six_matching_hashes(
    tmp_path: Path,
):
    hashes = rebuild_check(
        _write_rebuild_config(
            tmp_path
        )
    )

    assert len(hashes) == 6

    assert (
        "pilot100/base_manifest.jsonl"
        in hashes
    )
    assert (
        "smoke20/"
        "evaluation_sidecar.jsonl"
        in hashes
    )


def test_cli_validates_root_and_runs_rebuild_check(
    tmp_path: Path,
    capsys,
):
    manifest_root = (
        _write_valid_manifest_root(
            tmp_path
        )
    )
    _write_rebuild_config(
        tmp_path
    )

    exit_code = main([
        "--root",
        str(manifest_root),
        "--rebuild-check",
    ])

    summary = json.loads(
        capsys.readouterr().out
    )

    assert exit_code == 0
    assert summary["status"] == "ok"
    assert (
        summary[
            "collections_validated"
        ]
        == 3
    )
    assert (
        summary["rebuild_check"]
        is True
    )
    assert len(
        summary["manifest_sha256"]
    ) == 6
    assert len(
        summary["rebuild_sha256"]
    ) == 6