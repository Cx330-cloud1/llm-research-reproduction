#!/usr/bin/env python3
"""Recompute Table 1 metrics from the released-checkpoint confusion matrices."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent

PAPER = {
    "esconv": {"macro_f1": 27.70, "weighted_f1": 32.71, "bias": 0.45},
    "annomi": {"macro_f1": 27.92, "weighted_f1": 35.33, "bias": 0.50},
}


def per_class_f1(matrix: list[list[int]]) -> list[float]:
    n = len(matrix)
    scores = []
    for i in range(n):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(matrix[row][i] for row in range(n))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(score)
    return scores


def f1_metrics(matrix: list[list[int]]) -> tuple[float, float]:
    scores = per_class_f1(matrix)
    supports = [sum(row) for row in matrix]
    macro = sum(scores) / len(scores)
    weighted = sum(score * support for score, support in zip(scores, supports)) / sum(supports)
    return macro, weighted


def preference_bias(matrix: list[list[int]], iterations: int = 20) -> float:
    eps = 1e-6
    n = len(matrix)
    preference = [1.0] * n
    for _ in range(iterations):
        for i in range(n):
            numerator = sum(
                matrix[j][i] * preference[j] / (preference[i] + preference[j] + eps)
                for j in range(n)
            )
            denominator = sum(
                matrix[i][j] / (preference[i] + preference[j] + eps)
                for j in range(n)
            )
            preference[i] = numerator / denominator
        scale = n / sum(preference)
        preference = [value * scale for value in preference]
    mean = sum(preference) / n
    return math.sqrt(sum((value - mean) ** 2 for value in preference) / n)


def main() -> None:
    all_ok = True
    print("| Dataset | Metric | Paper | Recomputed | Match at paper precision |")
    print("| --- | --- | ---: | ---: | :---: |")
    for dataset in ("esconv", "annomi"):
        path = ROOT / "results" / f"{dataset}_result.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        matrix = data["confusion matrix"]
        macro, weighted = f1_metrics(matrix)
        bias = preference_bias(matrix)
        values = {
            "macro_f1": macro * 100,
            "weighted_f1": weighted * 100,
            "bias": bias,
        }
        for metric, value in values.items():
            paper = PAPER[dataset][metric]
            ok = round(value, 2) == paper
            all_ok &= ok
            print(f"| {dataset} | {metric} | {paper:.2f} | {value:.4f} | {'yes' if ok else 'no'} |")
    if not all_ok:
        raise SystemExit("At least one metric does not match the paper at the reported precision.")
    print("\nPASS: all six metrics match the paper at its reported precision.")


if __name__ == "__main__":
    main()

