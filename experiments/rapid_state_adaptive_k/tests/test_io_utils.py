from pathlib import Path

from scripts.io_utils import (
    append_jsonl,
    read_jsonl,
    sha256_json,
    successful_ids,
    write_jsonl_atomic,
)


def test_sha256_json_ignores_dict_insertion_order():
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def test_jsonl_round_trip_and_success_filter(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    rows = [
        {"sample_id": "s1", "status": "ok", "text": "中文"},
        {"sample_id": "s2", "status": "error", "error": "boom"},
    ]

    write_jsonl_atomic(path, rows)

    assert read_jsonl(path) == rows
    assert successful_ids(path) == {"s1"}


def test_append_jsonl_preserves_existing_rows(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    first = {"sample_id": "s1", "status": "ok"}
    second = {"sample_id": "s2", "status": "ok"}

    write_jsonl_atomic(path, [first])
    append_jsonl(path, second)

    assert read_jsonl(path) == [first, second]