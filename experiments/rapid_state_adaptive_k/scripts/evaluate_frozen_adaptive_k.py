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
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def softmax(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid temperature")

    scaled = logits / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)

    probs = np.exp(scaled)
    probs /= np.sum(probs, axis=1, keepdims=True)

    return probs


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--logits", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--adaptive-config", required=True)
    parser.add_argument("--expected-logits-sha256", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    logits_path = Path(args.logits).resolve()
    sidecar_path = Path(args.sidecar).resolve()
    config_path = Path(args.adaptive_config).resolve()
    output_path = Path(args.output).resolve()

    actual_sha = sha256_file(logits_path)

    if actual_sha != args.expected_logits_sha256:
        raise ValueError(
            f"logits SHA mismatch: {actual_sha}"
        )

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    temperature = float(config["temperature"])
    tau_conf = float(config["tau_conf"])
    tau_margin = float(config["tau_margin"])

    logits_rows = read_jsonl(logits_path)
    sidecar_rows = read_jsonl(sidecar_path)

    sidecar_by_id = {
        str(row["sample_id"]): row
        for row in sidecar_rows
    }

    label_order = list(
        logits_rows[0]["label_order"]
    )

    label_to_id = {
        label: index
        for index, label in enumerate(label_order)
    }

    matrix = []
    targets = []

    seen_ids = set()

    for row in logits_rows:
        sample_id = str(row["sample_id"])

        if sample_id in seen_ids:
            raise ValueError(
                f"duplicate logits sample_id: {sample_id}"
            )

        seen_ids.add(sample_id)

        if sample_id not in sidecar_by_id:
            raise ValueError(
                f"missing sidecar row: {sample_id}"
            )

        if row["label_order"] != label_order:
            raise ValueError(
                f"label order mismatch: {sample_id}"
            )

        values = np.asarray(
            row["strategy_logits"],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"non-finite logits: {sample_id}"
            )

        gold_strategy = str(
            sidecar_by_id[sample_id]["gold_strategy"]
        )

        matrix.append(values)
        targets.append(label_to_id[gold_strategy])

    if len(matrix) != len(sidecar_rows):
        raise ValueError(
            "logits/sidecar size mismatch"
        )

    logits = np.vstack(matrix)
    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    probs = softmax(
        logits,
        temperature,
    )

    ranking = np.argsort(
        -probs,
        axis=1,
    )

    top1 = ranking[:, 0]
    top2 = ranking[:, :2]

    top1_prob = probs[
        np.arange(len(targets)),
        top1,
    ]

    second_prob = probs[
        np.arange(len(targets)),
        ranking[:, 1],
    ]

    margin = (
        top1_prob - second_prob
    )

    fixed_k1_correct = (
        top1 == targets
    )

    fixed_k2_correct = np.any(
        top2 == targets[:, None],
        axis=1,
    )

    use_k1 = (
        (top1_prob >= tau_conf)
        & (margin >= tau_margin)
    )

    use_k2 = ~use_k1

    adaptive_correct = np.where(
        use_k1,
        fixed_k1_correct,
        fixed_k2_correct,
    )

    fixed_k1_coverage = float(
        np.mean(fixed_k1_correct)
    )

    fixed_k2_coverage = float(
        np.mean(fixed_k2_correct)
    )

    adaptive_coverage = float(
        np.mean(adaptive_correct)
    )

    k1_rate = float(
        np.mean(use_k1)
    )

    k2_rate = float(
        np.mean(use_k2)
    )

    average_k = (
        k1_rate + 2.0 * k2_rate
    )

    report = {
        "status": "ok",
        "split": "pilot100",
        "rows": len(targets),
        "input": {
            "logits_sha256": actual_sha,
            "sidecar_sha256": sha256_file(
                sidecar_path
            ),
        },
        "frozen_rule": {
            "temperature": temperature,
            "tau_conf": tau_conf,
            "tau_margin": tau_margin,
        },
        "fixed_k1": {
            "average_k": 1.0,
            "gold_coverage": fixed_k1_coverage,
        },
        "fixed_k2": {
            "average_k": 2.0,
            "gold_coverage": fixed_k2_coverage,
        },
        "adaptive_k": {
            "average_k": average_k,
            "k1_rate": k1_rate,
            "k2_rate": k2_rate,
            "gold_coverage": adaptive_coverage,
            "gain_over_fixed_k1": (
                adaptive_coverage
                - fixed_k1_coverage
            ),
            "gap_to_fixed_k2": (
                fixed_k2_coverage
                - adaptive_coverage
            ),
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())