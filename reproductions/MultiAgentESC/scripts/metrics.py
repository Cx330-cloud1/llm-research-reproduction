from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence


TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def distinct_n(predictions: Sequence[str], n: int) -> float:
    grams = [gram for text in predictions for gram in ngrams(tokenize(text), n)]
    return 0.0 if not grams else 100.0 * len(set(grams)) / len(grams)


def corpus_bleu(predictions: Sequence[str], references: Sequence[str], max_n: int) -> float:
    clipped = [0] * max_n
    totals = [0] * max_n
    candidate_length = reference_length = 0
    for prediction, reference in zip(predictions, references):
        candidate, ref = tokenize(prediction), tokenize(reference)
        candidate_length += len(candidate)
        reference_length += len(ref)
        for index in range(max_n):
            n = index + 1
            candidate_counts = Counter(ngrams(candidate, n))
            reference_counts = Counter(ngrams(ref, n))
            totals[index] += sum(candidate_counts.values())
            clipped[index] += sum(min(count, reference_counts[gram]) for gram, count in candidate_counts.items())
    if candidate_length == 0 or any(total == 0 or match == 0 for match, total in zip(clipped, totals)):
        return 0.0
    precisions = [match / total for match, total in zip(clipped, totals)]
    bp = 1.0 if candidate_length > reference_length else math.exp(1 - reference_length / candidate_length)
    return 100.0 * bp * math.exp(sum(math.log(value) for value in precisions) / max_n)


def _lcs(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(previous[index-1] + 1 if left_token == right_token else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float:
    values = []
    for prediction, reference in zip(predictions, references):
        left, right = tokenize(prediction), tokenize(reference)
        if not left or not right:
            values.append(0.0)
            continue
        common = _lcs(left, right)
        precision, recall = common / len(left), common / len(right)
        values.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return 100.0 * sum(values) / len(values) if values else 0.0


def token_f1(predictions: Sequence[str], references: Sequence[str]) -> float:
    values = []
    for prediction, reference in zip(predictions, references):
        left, right = Counter(tokenize(prediction)), Counter(tokenize(reference))
        overlap, left_n, right_n = sum((left & right).values()), sum(left.values()), sum(right.values())
        if not overlap or not left_n or not right_n:
            values.append(0.0)
        else:
            precision, recall = overlap / left_n, overlap / right_n
            values.append(2 * precision * recall / (precision + recall))
    return 100.0 * sum(values) / len(values) if values else 0.0


def automatic_metrics(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal lengths")
    return {
        "D-1": distinct_n(predictions, 1),
        "D-2": distinct_n(predictions, 2),
        "B-1": corpus_bleu(predictions, references, 1),
        "B-2": corpus_bleu(predictions, references, 2),
        "B-3": corpus_bleu(predictions, references, 3),
        "F1": token_f1(predictions, references),
        "R-L": rouge_l(predictions, references),
    }


def exact_sign_test(win: int, lose: int) -> float:
    n = win + lose
    if n == 0:
        return 1.0
    smaller = min(win, lose)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2 ** n)
    return min(1.0, 2 * tail)
