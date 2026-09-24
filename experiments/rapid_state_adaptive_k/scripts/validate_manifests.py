from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from .build_manifests import (
    main as build_manifests_main,
)
from .io_utils import read_jsonl


FORBIDDEN_BASE_FIELDS = {
    "gold_strategy",
    "reference_response",
    "target_response",
    "future_turns",
}

TURN_ID_PATTERN = re.compile(r"^u(\d+)$")

COLLECTION_NAMES = (
    "dev_cal",
    "pilot100",
    "smoke20",
)


def _keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _keys(child)

    elif isinstance(value, list):
        for child in value:
            yield from _keys(child)


def _turn_number(value: Any) -> int:
    match = TURN_ID_PATTERN.fullmatch(
        str(value)
    )

    if match is None:
        raise ValueError(
            f"invalid turn id: {value}"
        )

    return int(match.group(1))


def validate_collection(
    base_rows: list[dict[str, Any]],
    sidecar_rows: list[dict[str, Any]],
) -> None:
    errors: list[str] = []

    base_ids = [
        str(row.get("sample_id"))
        for row in base_rows
    ]
    sidecar_ids = [
        str(row.get("sample_id"))
        for row in sidecar_rows
    ]

    if len(base_ids) != len(
        set(base_ids)
    ):
        errors.append(
            "duplicate sample_id"
        )

    if len(sidecar_ids) != len(
        set(sidecar_ids)
    ):
        errors.append(
            "duplicate sample_id"
        )

    if set(base_ids) != set(
        sidecar_ids
    ):
        errors.append(
            "base/sidecar sample_id mismatch"
        )

    hashes = [
        row.get("context_hash")
        for row in base_rows
    ]

    if any(
        value is None
        for value in hashes
    ):
        errors.append(
            "missing context_hash"
        )

    elif len(hashes) != len(
        set(hashes)
    ):
        errors.append(
            "duplicate context_hash"
        )

    for row in base_rows:
        sample_id = str(
            row.get("sample_id")
        )

        forbidden = (
            FORBIDDEN_BASE_FIELDS
            & set(_keys(row))
        )

        for key in sorted(forbidden):
            errors.append(
                f"forbidden field: {key}"
            )

        try:
            target = _turn_number(
                row["target_turn_id"]
            )

        except KeyError:
            errors.append(
                "missing target_turn_id: "
                f"{sample_id}"
            )
            continue

        except ValueError as error:
            errors.append(str(error))
            continue

        visible_turn_ids = row.get(
            "visible_turn_ids"
        )

        if not isinstance(
            visible_turn_ids,
            list,
        ):
            errors.append(
                "invalid visible_turn_ids: "
                f"{sample_id}"
            )
            continue

        for turn_id in visible_turn_ids:
            try:
                visible = _turn_number(
                    turn_id
                )

            except ValueError as error:
                errors.append(str(error))
                continue

            if visible >= target:
                errors.append(
                    "future turn visible: "
                    f"{sample_id}"
                )

    if errors:
        raise ValueError(
            "; ".join(errors)
        )


def validate_all_rows(
    dev: list[dict[str, Any]],
    pilot: list[dict[str, Any]],
    smoke: list[dict[str, Any]],
) -> None:
    errors: list[str] = []

    dev_dialogues = {
        str(row["dialogue_id"])
        for row in dev
    }
    pilot_dialogues = {
        str(row["dialogue_id"])
        for row in pilot
    }

    if dev_dialogues & pilot_dialogues:
        errors.append(
            "dev_cal/pilot100 overlap"
        )

    if len(pilot) != 100:
        errors.append(
            "pilot100 must contain 100 rows"
        )

    if len(pilot_dialogues) != 100:
        errors.append(
            "pilot100 must contain "
            "100 distinct dialogues"
        )

    if len(smoke) != 20:
        errors.append(
            "smoke20 must contain 20 rows"
        )

    pilot_ids = {
        str(row["sample_id"])
        for row in pilot
    }
    smoke_ids = {
        str(row["sample_id"])
        for row in smoke
    }

    if not smoke_ids.issubset(
        pilot_ids
    ):
        errors.append(
            "smoke20 is not a subset "
            "of pilot100"
        )

    if errors:
        raise ValueError(
            "; ".join(errors)
        )


def validate_all(
    root: Any,
) -> None:
    manifest_root = Path(root)

    loaded: dict[
        str,
        tuple[
            list[dict[str, Any]],
            list[dict[str, Any]],
        ],
    ] = {}

    for collection_name in (
        COLLECTION_NAMES
    ):
        collection_root = (
            manifest_root
            / collection_name
        )

        base_rows = read_jsonl(
            collection_root
            / "base_manifest.jsonl"
        )
        sidecar_rows = read_jsonl(
            collection_root
            / "evaluation_sidecar.jsonl"
        )

        validate_collection(
            base_rows,
            sidecar_rows,
        )

        loaded[collection_name] = (
            base_rows,
            sidecar_rows,
        )

    validate_all_rows(
        dev=loaded["dev_cal"][0],
        pilot=loaded["pilot100"][0],
        smoke=loaded["smoke20"][0],
    )


def file_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def manifest_hashes(
    root: Any,
) -> dict[str, str]:
    manifest_root = Path(root)
    hashes: dict[str, str] = {}

    for collection_name in (
        COLLECTION_NAMES
    ):
        for filename in (
            "base_manifest.jsonl",
            "evaluation_sidecar.jsonl",
        ):
            path = (
                manifest_root
                / collection_name
                / filename
            )

            if not path.is_file():
                raise ValueError(
                    "missing manifest file: "
                    f"{path}"
                )

            relative = path.relative_to(
                manifest_root
            ).as_posix()

            hashes[relative] = (
                file_sha256(path)
            )

    return dict(
        sorted(hashes.items())
    )


def compare_manifest_hashes(
    first: dict[str, str],
    second: dict[str, str],
) -> dict[str, str]:
    if first == second:
        return first

    all_paths = sorted(
        set(first)
        | set(second)
    )

    mismatches = [
        path
        for path in all_paths
        if first.get(path)
        != second.get(path)
    ]

    raise ValueError(
        "deterministic rebuild mismatch: "
        + ", ".join(mismatches)
    )


def rebuild_check(
    config_path: Any,
) -> dict[str, str]:
    resolved_config = Path(
        config_path
    ).resolve()

    with (
        tempfile.TemporaryDirectory()
        as first_directory,
        tempfile.TemporaryDirectory()
        as second_directory,
    ):
        first_root = (
            Path(first_directory)
            / "manifests"
        )
        second_root = (
            Path(second_directory)
            / "manifests"
        )

        for output_root in (
            first_root,
            second_root,
        ):
            captured_stdout = io.StringIO()

            with contextlib.redirect_stdout(
                captured_stdout
            ):
                exit_code = (
                    build_manifests_main([
                        "--config",
                        str(resolved_config),
                        "--output-root",
                        str(output_root),
                    ])
                )

            if exit_code != 0:
                raise ValueError(
                    "manifest rebuild failed "
                    f"with exit code {exit_code}"
                )

        first_hashes = manifest_hashes(
            first_root
        )
        second_hashes = manifest_hashes(
            second_root
        )

        return compare_manifest_hashes(
            first_hashes,
            second_hashes,
        )


def _argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Rapid State + "
            "Adaptive-k manifests."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
    )
    parser.add_argument(
        "--rebuild-check",
        action="store_true",
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = _argument_parser().parse_args(
        argv
    )

    manifest_root = Path(
        args.root
    ).resolve()

    validate_all(
        manifest_root
    )

    current_hashes = manifest_hashes(
        manifest_root
    )
    rebuilt_hashes: dict[
        str,
        str,
    ] = {}

    if args.rebuild_check:
        config_path = (
            manifest_root.parent
            / "configs"
            / "experiment.json"
        )

        rebuilt_hashes = rebuild_check(
            config_path
        )

    summary = {
        "status": "ok",
        "collections_validated": len(
            COLLECTION_NAMES
        ),
        "rebuild_check": bool(
            args.rebuild_check
        ),
        "manifest_sha256": (
            current_hashes
        ),
        "rebuild_sha256": (
            rebuilt_hashes
        ),
    }

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )