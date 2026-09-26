from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at line {line_number}"
                ) from exc

    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def softmax(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if (
        not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError(
            "temperature must be positive and finite"
        )

    scaled = logits / temperature

    scaled -= np.max(
        scaled,
        axis=1,
        keepdims=True,
    )

    probabilities = np.exp(scaled)

    probabilities /= np.sum(
        probabilities,
        axis=1,
        keepdims=True,
    )

    return probabilities


def quantile_values(
    values: np.ndarray,
) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def scan_candidate(
    top1_correct: np.ndarray,
    top2_correct: np.ndarray,
    confidence: np.ndarray,
    margin: np.ndarray,
    confidence_threshold: float,
    margin_threshold: float,
    fixed_k1_coverage: float,
    fixed_k2_coverage: float,
) -> dict[str, float]:
    """
    Adaptive rule:

    K=1 only when BOTH:
      top1 confidence >= confidence threshold
      top1-top2 margin >= margin threshold

    Otherwise K=2.
    """
    use_k1 = (
        (confidence >= confidence_threshold)
        & (margin >= margin_threshold)
    )

    use_k2 = ~use_k1

    covered = np.where(
        use_k1,
        top1_correct,
        top2_correct,
    )

    k1_rate = float(
        np.mean(use_k1)
    )

    k2_rate = float(
        np.mean(use_k2)
    )

    average_k = (
        k1_rate
        + 2.0 * k2_rate
    )

    coverage = float(
        np.mean(covered)
    )

    rescue = (
        coverage
        - fixed_k1_coverage
    )

    extra_candidates = (
        average_k - 1.0
    )

    if extra_candidates > 0:
        rescue_per_extra_candidate = (
            rescue
            / extra_candidates
        )
    else:
        rescue_per_extra_candidate = 0.0

    return {
        "confidence_threshold": float(
            confidence_threshold
        ),
        "margin_threshold": float(
            margin_threshold
        ),
        "k1_rate": k1_rate,
        "k2_rate": k2_rate,
        "average_k": float(
            average_k
        ),
        "gold_coverage": coverage,
        "gain_over_fixed_k1": float(
            rescue
        ),
        "gap_to_fixed_k2": float(
            fixed_k2_coverage
            - coverage
        ),
        "rescue_per_extra_candidate": float(
            rescue_per_extra_candidate
        ),
    }


def pareto_frontier(
    candidates: list[dict[str, float]],
) -> list[dict[str, float]]:
    frontier = []

    for candidate in candidates:
        dominated = False

        for other in candidates:
            if other is candidate:
                continue

            no_more_cost = (
                other["average_k"]
                <= candidate["average_k"]
                + 1e-12
            )

            no_less_coverage = (
                other["gold_coverage"]
                >= candidate["gold_coverage"]
                - 1e-12
            )

            strictly_better = (
                other["average_k"]
                < candidate["average_k"]
                - 1e-12
                or other["gold_coverage"]
                > candidate["gold_coverage"]
                + 1e-12
            )

            if (
                no_more_cost
                and no_less_coverage
                and strictly_better
            ):
                dominated = True
                break

        if not dominated:
            frontier.append(candidate)

    frontier.sort(
        key=lambda row: (
            row["average_k"],
            -row["gold_coverage"],
        )
    )

    return frontier


def best_under_budget(
    candidates: list[dict[str, float]],
    budget: float,
) -> dict[str, float] | None:
    eligible = [
        row
        for row in candidates
        if row["average_k"]
        <= budget + 1e-12
    ]

    if not eligible:
        return None

    return max(
        eligible,
        key=lambda row: (
            row["gold_coverage"],
            -row["average_k"],
            row["rescue_per_extra_candidate"],
        ),
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
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze calibrated strategy probabilities "
            "and Adaptive-K trade-offs on dev_cal."
        )
    )

    parser.add_argument(
        "--logits",
        required=True,
    )

    parser.add_argument(
        "--sidecar",
        required=True,
    )

    parser.add_argument(
        "--calibration",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    logits_path = Path(
        args.logits
    ).expanduser().resolve()

    sidecar_path = Path(
        args.sidecar
    ).expanduser().resolve()

    calibration_path = Path(
        args.calibration
    ).expanduser().resolve()

    output_path = Path(
        args.output
    ).expanduser().resolve()

    for path in (
        logits_path,
        sidecar_path,
        calibration_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    calibration = json.loads(
        calibration_path.read_text(
            encoding="utf-8"
        )
    )

    if calibration.get("status") != "ok":
        raise ValueError(
            "calibration report is not ok"
        )

    expected_logits_sha = calibration[
        "input"
    ]["logits_sha256"]

    actual_logits_sha = sha256_file(
        logits_path
    )

    if (
        actual_logits_sha
        != expected_logits_sha
    ):
        raise ValueError(
            "logits SHA256 does not match "
            "calibration report"
        )

    temperature = float(
        calibration[
            "selection"
        ]["selected_temperature"]
    )

    label_order = list(
        calibration["label_order"]
    )

    label_to_id = {
        label: index
        for index, label in enumerate(
            label_order
        )
    }

    logits_rows = read_jsonl(
        logits_path
    )

    sidecar_rows = read_jsonl(
        sidecar_path
    )

    sidecar_by_id = {
        str(row["sample_id"]): row
        for row in sidecar_rows
    }

    if (
        len(sidecar_by_id)
        != len(sidecar_rows)
    ):
        raise ValueError(
            "duplicate sample_id in sidecar"
        )

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
                f"duplicate logits sample_id: {sample_id}"
            )

        seen_ids.add(sample_id)

        if sample_id not in sidecar_by_id:
            raise ValueError(
                f"missing sidecar row: {sample_id}"
            )

        if (
            row.get("label_order")
            != label_order
        ):
            raise ValueError(
                f"label order mismatch: {sample_id}"
            )

        values = np.asarray(
            row["strategy_logits"],
            dtype=np.float64,
        )

        if values.shape != (
            len(label_order),
        ):
            raise ValueError(
                f"invalid logits shape: {sample_id}"
            )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                f"non-finite logits: {sample_id}"
            )

        gold_strategy = str(
            sidecar_by_id[
                sample_id
            ]["gold_strategy"]
        )

        if gold_strategy not in label_to_id:
            raise ValueError(
                f"unknown gold strategy: {gold_strategy}"
            )

        matrix.append(values)

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
        raise ValueError(
            "logits/sidecar size mismatch"
        )

    logits = np.vstack(
        matrix
    )

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    probabilities = softmax(
        logits,
        temperature,
    )

    if not np.allclose(
        np.sum(
            probabilities,
            axis=1,
        ),
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "probabilities do not sum to 1"
        )

    ranking = np.argsort(
        -probabilities,
        axis=1,
    )

    top1 = ranking[:, 0]
    top2 = ranking[:, :2]

    top1_probability = probabilities[
        np.arange(len(targets)),
        top1,
    ]

    second_probability = probabilities[
        np.arange(len(targets)),
        ranking[:, 1],
    ]

    margin = (
        top1_probability
        - second_probability
    )

    top1_correct = (
        top1 == targets
    )

    top2_correct = np.any(
        top2 == targets[:, None],
        axis=1,
    )

    fixed_k1_coverage = float(
        np.mean(
            top1_correct
        )
    )

    fixed_k2_coverage = float(
        np.mean(
            top2_correct
        )
    )

    if (
        fixed_k2_coverage
        + 1e-12
        < fixed_k1_coverage
    ):
        raise ValueError(
            "K=2 coverage lower than K=1"
        )

    confidence_quantiles = (
        quantile_values(
            top1_probability
        )
    )

    margin_quantiles = (
        quantile_values(
            margin
        )
    )

    quantile_points = [
        0.10,
        0.25,
        0.40,
        0.50,
        0.60,
        0.75,
        0.90,
    ]

    confidence_thresholds = sorted(
        {
            float(
                np.quantile(
                    top1_probability,
                    q,
                )
            )
            for q in quantile_points
        }
    )

    margin_thresholds = sorted(
        {
            float(
                np.quantile(
                    margin,
                    q,
                )
            )
            for q in quantile_points
        }
    )

    candidates = []

    for confidence_threshold in (
        confidence_thresholds
    ):
        for margin_threshold in (
            margin_thresholds
        ):
            result = scan_candidate(
                top1_correct=top1_correct,
                top2_correct=top2_correct,
                confidence=top1_probability,
                margin=margin,
                confidence_threshold=(
                    confidence_threshold
                ),
                margin_threshold=(
                    margin_threshold
                ),
                fixed_k1_coverage=(
                    fixed_k1_coverage
                ),
                fixed_k2_coverage=(
                    fixed_k2_coverage
                ),
            )

            if not (
                1.0 - 1e-12
                <= result["average_k"]
                <= 2.0 + 1e-12
            ):
                raise ValueError(
                    "invalid average K"
                )

            if not (
                fixed_k1_coverage - 1e-12
                <= result["gold_coverage"]
                <= fixed_k2_coverage + 1e-12
            ):
                raise ValueError(
                    "adaptive coverage outside "
                    "K=1/K=2 bounds"
                )

            candidates.append(result)

    frontier = pareto_frontier(
        candidates
    )

    budgets = {
        str(budget): best_under_budget(
            candidates,
            budget,
        )
        for budget in (
            1.25,
            1.50,
            1.75,
        )
    }

    report = {
        "status": "ok",
        "split": "dev_cal",
        "rows": int(
            len(targets)
        ),
        "temperature": temperature,
        "input": {
            "logits_sha256": (
                actual_logits_sha
            ),
            "calibration_path": str(
                calibration_path
            ),
            "calibration_sha256": (
                sha256_file(
                    calibration_path
                )
            ),
        },
        "fixed_baselines": {
            "k1": {
                "average_k": 1.0,
                "gold_coverage": (
                    fixed_k1_coverage
                ),
            },
            "k2": {
                "average_k": 2.0,
                "gold_coverage": (
                    fixed_k2_coverage
                ),
            },
            "absolute_gain_k2_over_k1": (
                fixed_k2_coverage
                - fixed_k1_coverage
            ),
        },
        "uncertainty_distribution": {
            "top1_confidence": (
                confidence_quantiles
            ),
            "top1_top2_margin": (
                margin_quantiles
            ),
        },
        "rule": (
            "K=1 iff top1_confidence >= tau_conf "
            "AND top1_top2_margin >= tau_margin; "
            "otherwise K=2"
        ),
        "scan": {
            "quantile_points": (
                quantile_points
            ),
            "confidence_thresholds": (
                confidence_thresholds
            ),
            "margin_thresholds": (
                margin_thresholds
            ),
            "candidate_count": (
                len(candidates)
            ),
            "candidates": candidates,
        },
        "pareto_frontier": frontier,
        "best_under_average_k_budget": (
            budgets
        ),
    }

    write_json_atomic(
        output_path,
        report,
    )

    summary = {
        "temperature": temperature,
        "fixed_k1_coverage": (
            fixed_k1_coverage
        ),
        "fixed_k2_coverage": (
            fixed_k2_coverage
        ),
        "gain_k2_over_k1": (
            fixed_k2_coverage
            - fixed_k1_coverage
        ),
        "top1_confidence": (
            confidence_quantiles
        ),
        "margin": (
            margin_quantiles
        ),
        "pareto_points": len(
            frontier
        ),
        "best_under_average_k_budget": (
            budgets
        ),
        "output": str(
            output_path
        ),
    }

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())