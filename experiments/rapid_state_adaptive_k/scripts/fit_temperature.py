from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .io_utils import read_jsonl
except ImportError:
    from scripts.io_utils import read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LABEL_ORDER = [
    "Reflection of feelings",
    "Self-disclosure",
    "Question",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Restatement or Paraphrasing",
    "Information",
    "Others",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def logsumexp(
    values: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    maximum = np.max(
        values,
        axis=axis,
        keepdims=True,
    )

    return (
        np.squeeze(
            maximum,
            axis=axis,
        )
        + np.log(
            np.sum(
                np.exp(values - maximum),
                axis=axis,
            )
        )
    )


def softmax(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if not math.isfinite(temperature):
        raise ValueError(
            "temperature must be finite"
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be positive"
        )

    scaled = logits / temperature

    scaled = (
        scaled
        - np.max(
            scaled,
            axis=1,
            keepdims=True,
        )
    )

    probabilities = np.exp(scaled)

    probabilities /= np.sum(
        probabilities,
        axis=1,
        keepdims=True,
    )

    return probabilities


def negative_log_likelihood(
    logits: np.ndarray,
    targets: np.ndarray,
    temperature: float,
) -> float:
    if temperature <= 0:
        return float("inf")

    scaled = logits / temperature

    row_indices = np.arange(
        logits.shape[0]
    )

    target_logits = scaled[
        row_indices,
        targets,
    ]

    losses = (
        logsumexp(
            scaled,
            axis=1,
        )
        - target_logits
    )

    return float(
        np.mean(losses)
    )


def accuracy(
    logits: np.ndarray,
    targets: np.ndarray,
) -> float:
    predictions = np.argmax(
        logits,
        axis=1,
    )

    return float(
        np.mean(
            predictions == targets
        )
    )


def expected_calibration_error(
    logits: np.ndarray,
    targets: np.ndarray,
    temperature: float,
    bins: int = 15,
) -> float:
    if bins < 1:
        raise ValueError(
            "bins must be at least 1"
        )

    probabilities = softmax(
        logits,
        temperature,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    confidences = np.max(
        probabilities,
        axis=1,
    )

    correct = (
        predictions == targets
    ).astype(float)

    bin_indices = np.minimum(
        (
            confidences
            * bins
        ).astype(int),
        bins - 1,
    )

    ece = 0.0
    total = len(targets)

    for bin_index in range(bins):
        mask = (
            bin_indices == bin_index
        )

        count = int(
            np.sum(mask)
        )

        if count == 0:
            continue

        bin_accuracy = float(
            np.mean(
                correct[mask]
            )
        )

        bin_confidence = float(
            np.mean(
                confidences[mask]
            )
        )

        ece += (
            count / total
        ) * abs(
            bin_accuracy
            - bin_confidence
        )

    return float(ece)


def golden_section_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    minimum_temperature: float = 0.01,
    maximum_temperature: float = 100.0,
    iterations: int = 120,
) -> float:
    """
    Minimize NLL over a positive temperature.

    Search is performed in log-temperature space so that
    small and large temperatures are explored symmetrically
    on a multiplicative scale.
    """
    if minimum_temperature <= 0:
        raise ValueError(
            "minimum_temperature must be positive"
        )

    if (
        maximum_temperature
        <= minimum_temperature
    ):
        raise ValueError(
            "maximum_temperature must exceed minimum_temperature"
        )

    if iterations < 1:
        raise ValueError(
            "iterations must be at least 1"
        )

    left = math.log(
        minimum_temperature
    )
    right = math.log(
        maximum_temperature
    )

    ratio = (
        math.sqrt(5.0)
        - 1.0
    ) / 2.0

    x1 = (
        right
        - ratio
        * (right - left)
    )

    x2 = (
        left
        + ratio
        * (right - left)
    )

    f1 = negative_log_likelihood(
        logits,
        targets,
        math.exp(x1),
    )

    f2 = negative_log_likelihood(
        logits,
        targets,
        math.exp(x2),
    )

    for _ in range(iterations):
        if f1 > f2:
            left = x1
            x1 = x2
            f1 = f2

            x2 = (
                left
                + ratio
                * (right - left)
            )

            f2 = negative_log_likelihood(
                logits,
                targets,
                math.exp(x2),
            )
        else:
            right = x2
            x2 = x1
            f2 = f1

            x1 = (
                right
                - ratio
                * (right - left)
            )

            f1 = negative_log_likelihood(
                logits,
                targets,
                math.exp(x1),
            )

    best_log_temperature = (
        left + right
    ) / 2.0

    return float(
        math.exp(
            best_log_temperature
        )
    )


def load_calibration_data(
    logits_path: Path,
    sidecar_path: Path,
    expected_label_order: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
]:
    logits_rows = read_jsonl(
        logits_path
    )

    sidecar_rows = read_jsonl(
        sidecar_path
    )

    if not logits_rows:
        raise ValueError(
            "logits file is empty"
        )

    if not sidecar_rows:
        raise ValueError(
            "sidecar file is empty"
        )

    sidecar_by_id = {}

    for row in sidecar_rows:
        sample_id = str(
            row["sample_id"]
        )

        if sample_id in sidecar_by_id:
            raise ValueError(
                "duplicate sidecar sample_id: "
                f"{sample_id}"
            )

        sidecar_by_id[
            sample_id
        ] = row

    label_to_id = {
        label: index
        for index, label in enumerate(
            expected_label_order
        )
    }

    matrix = []
    targets = []
    sample_ids = []

    seen_ids = set()

    for row in logits_rows:
        sample_id = str(
            row["sample_id"]
        )

        if sample_id in seen_ids:
            raise ValueError(
                "duplicate logits sample_id: "
                f"{sample_id}"
            )

        seen_ids.add(
            sample_id
        )

        if sample_id not in sidecar_by_id:
            raise ValueError(
                "missing sidecar row for "
                f"{sample_id}"
            )

        if row.get("status") != "ok":
            raise ValueError(
                "non-ok logits row for "
                f"{sample_id}"
            )

        if (
            row.get("label_order")
            != expected_label_order
        ):
            raise ValueError(
                "label order mismatch for "
                f"{sample_id}"
            )

        values = [
            float(value)
            for value in row[
                "strategy_logits"
            ]
        ]

        if (
            len(values)
            != len(
                expected_label_order
            )
        ):
            raise ValueError(
                "invalid logit count for "
                f"{sample_id}"
            )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "non-finite logits for "
                f"{sample_id}"
            )

        gold_strategy = str(
            sidecar_by_id[
                sample_id
            ]["gold_strategy"]
        )

        if gold_strategy not in label_to_id:
            raise ValueError(
                "unknown gold strategy: "
                f"{gold_strategy}"
            )

        matrix.append(
            values
        )

        targets.append(
            label_to_id[
                gold_strategy
            ]
        )

        sample_ids.append(
            sample_id
        )

    if len(sample_ids) != len(
        sidecar_rows
    ):
        logits_ids = set(
            sample_ids
        )

        sidecar_ids = set(
            sidecar_by_id
        )

        missing_logits = sorted(
            sidecar_ids
            - logits_ids
        )

        raise ValueError(
            "logits/sidecar size mismatch; "
            f"missing_logits={missing_logits[:10]}"
        )

    logits = np.asarray(
        matrix,
        dtype=np.float64,
    )

    target_array = np.asarray(
        targets,
        dtype=np.int64,
    )

    return (
        logits,
        target_array,
        sample_ids,
    )


def write_json_atomic(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit scalar Temperature Scaling "
            "on frozen dev_cal strategy logits."
        )
    )

    parser.add_argument(
        "--logits",
        required=True,
        help=(
            "Path to "
            "strategy_results.raw.jsonl."
        ),
    )

    parser.add_argument(
        "--sidecar",
        required=True,
        help=(
            "Path to dev_cal "
            "evaluation_sidecar.jsonl."
        ),
    )

    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "experiment.json"
        ),
        help="Frozen experiment configuration.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Destination JSON report.",
    )

    parser.add_argument(
        "--expected-logits-sha256",
        help=(
            "Optional expected SHA256 for "
            "the raw logits input."
        ),
    )

    parser.add_argument(
        "--ece-bins",
        type=int,
        default=15,
    )

    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()

    logits_path = Path(
        args.logits
    ).expanduser().resolve()

    sidecar_path = Path(
        args.sidecar
    ).expanduser().resolve()

    config_path = Path(
        args.config
    ).expanduser().resolve()

    output_path = Path(
        args.output
    ).expanduser().resolve()

    for path in (
        logits_path,
        sidecar_path,
        config_path,
    ):
        if not path.is_file():
            parser.error(
                f"missing file: {path}"
            )

    logits_sha256 = sha256_file(
        logits_path
    )

    if (
        args.expected_logits_sha256
        and logits_sha256
        != args.expected_logits_sha256
    ):
        raise ValueError(
            "raw logits SHA256 mismatch: "
            f"expected="
            f"{args.expected_logits_sha256}, "
            f"actual={logits_sha256}"
        )

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    label_order = list(
        config["label_order"]
    )

    if (
        label_order
        != DEFAULT_LABEL_ORDER
    ):
        raise ValueError(
            "frozen label order changed"
        )

    (
        logits,
        targets,
        sample_ids,
    ) = load_calibration_data(
        logits_path,
        sidecar_path,
        label_order,
    )

    baseline_temperature = 1.0

    baseline_nll = (
        negative_log_likelihood(
            logits,
            targets,
            baseline_temperature,
        )
    )

    baseline_ece = (
        expected_calibration_error(
            logits,
            targets,
            baseline_temperature,
            bins=args.ece_bins,
        )
    )

    baseline_accuracy = accuracy(
        logits,
        targets,
    )

    fitted_temperature = (
        golden_section_temperature(
            logits,
            targets,
        )
    )

    fitted_nll = (
        negative_log_likelihood(
            logits,
            targets,
            fitted_temperature,
        )
    )

    fitted_ece = (
        expected_calibration_error(
            logits,
            targets,
            fitted_temperature,
            bins=args.ece_bins,
        )
    )

    fitted_accuracy = accuracy(
        logits / fitted_temperature,
        targets,
    )

    nll_improvement = (
        baseline_nll
        - fitted_nll
    )

    use_fitted_temperature = (
        math.isfinite(
            fitted_temperature
        )
        and nll_improvement > 1e-12
    )

    selected_temperature = (
        fitted_temperature
        if use_fitted_temperature
        else 1.0
    )

    if abs(
        fitted_accuracy
        - baseline_accuracy
    ) > 1e-12:
        raise ValueError(
            "temperature scaling changed accuracy"
        )

    report = {
        "status": "ok",
        "method": "scalar_temperature_scaling",
        "fit_split": "dev_cal",
        "rows": int(
            logits.shape[0]
        ),
        "classes": int(
            logits.shape[1]
        ),
        "label_order": label_order,
        "input": {
            "logits_path": str(
                logits_path
            ),
            "logits_sha256": (
                logits_sha256
            ),
            "sidecar_path": str(
                sidecar_path
            ),
            "sidecar_sha256": (
                sha256_file(
                    sidecar_path
                )
            ),
        },
        "optimization": {
            "objective": "negative_log_likelihood",
            "search": "golden_section_log_temperature",
            "temperature_bounds": [
                0.01,
                100.0,
            ],
            "iterations": 120,
            "fallback_rule": (
                "use T=1 unless fitted T "
                "strictly improves dev_cal NLL"
            ),
        },
        "baseline": {
            "temperature": 1.0,
            "nll": baseline_nll,
            "ece_15bin": baseline_ece,
            "accuracy": baseline_accuracy,
        },
        "fitted": {
            "temperature": fitted_temperature,
            "nll": fitted_nll,
            "ece_15bin": fitted_ece,
            "accuracy": fitted_accuracy,
            "nll_improvement": nll_improvement,
        },
        "selection": {
            "use_fitted_temperature": (
                use_fitted_temperature
            ),
            "selected_temperature": (
                selected_temperature
            ),
            "fallback_to_one": (
                not use_fitted_temperature
            ),
        },
        "sample_id_first": (
            sample_ids[0]
        ),
        "sample_id_last": (
            sample_ids[-1]
        ),
    }

    write_json_atomic(
        output_path,
        report,
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())