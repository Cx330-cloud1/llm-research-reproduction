from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Target:
    dialogue_id: str
    target_id: str
    dialogue_index: int
    target_index: int
    turn_index: int
    context: str
    post: str
    reference: str
    gold_strategy: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def load_dialogues(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("ESConv root must be a list")
    return data


def _content(turn: dict[str, Any]) -> str:
    return re.sub(r"\s+", " ", str(turn.get("content", "")).strip())


def _natural(history: list[dict[str, str]]) -> str:
    parts = []
    for turn in history:
        role = "User" if turn["speaker"] == "seeker" else "Assistant"
        parts.append(f"{role}: {turn['content']}")
    return " ".join(parts)


def select_dialogues(
    dialogues: list[dict[str, Any]], limit: int, selection: str = "first", seed: int = 42
) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(dialogues))
    if selection == "random":
        random.Random(seed).shuffle(indexed)
    elif selection != "first":
        raise ValueError("selection must be 'first' or 'random'")
    return indexed[:limit]


def extract_targets(
    dialogues: list[dict[str, Any]], limit: int = 100, selection: str = "first", seed: int = 42
) -> list[Target]:
    """Replicate the public main.py target construction, including pairs of consecutive supporter turns."""
    records: list[Target] = []
    for original_index, sample in select_dialogues(dialogues, limit, selection, seed):
        dialog = sample["dialog"]
        history: list[dict[str, str]] = []
        count = 0
        target_index = 0
        while count < len(dialog):
            turn = dialog[count]
            if count != 0 and turn["speaker"] == "supporter":
                context = _natural(history)
                post = history[-1]["content"] if history else ""
                if count < len(dialog) - 1 and dialog[count + 1]["speaker"] == "supporter":
                    chosen = [dialog[count], dialog[count + 1]]
                    count_after = count + 2
                else:
                    chosen = [dialog[count]]
                    count_after = count + 1
                reference = " ".join(_content(item) for item in chosen)
                strategy = " and ".join(
                    str(item.get("annotation", {}).get("strategy", "")) for item in chosen
                )
                target_index += 1
                records.append(Target(
                    dialogue_id=f"d{original_index:04d}",
                    target_id=f"d{original_index:04d}_t{target_index:02d}_u{count:02d}",
                    dialogue_index=original_index,
                    target_index=target_index,
                    turn_index=count_after,
                    context=context,
                    post=post,
                    reference=reference,
                    gold_strategy=strategy,
                ))
                for item in chosen:
                    history.append({"speaker": item["speaker"], "content": _content(item)})
                count = count_after
            else:
                history.append({"speaker": turn["speaker"], "content": _content(turn)})
                count += 1
    return records


def build_experiences(dialogues: list[dict[str, Any]], start: int = 100) -> list[dict[str, str]]:
    experiences = []
    for sample in dialogues[start:]:
        dialog = sample["dialog"]
        for index in range(len(dialog) - 1):
            left, right = dialog[index], dialog[index + 1]
            if left["speaker"] == "seeker" and right["speaker"] == "supporter":
                experiences.append({
                    "post": _content(left),
                    "response": _content(right),
                    "strategy": str(right.get("annotation", {}).get("strategy", "Others")),
                })
    return experiences


def few_shot_examples(dialogues: list[dict[str, Any]], count: int = 3) -> list[dict[str, str]]:
    pool = build_experiences(dialogues, start=min(400, len(dialogues)))
    if not pool:
        pool = build_experiences(dialogues, start=100)
    return pool[:count]


def final_target_per_dialogue(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    final: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["dialogue_id"])
        if key not in final or int(row.get("target_index", 0)) > int(final[key].get("target_index", 0)):
            final[key] = row
    return final


class Retriever:
    def __init__(self, experiences: list[dict[str, str]], backend: str, model_name: str):
        self.experiences = experiences
        self.backend = backend
        self.model = None
        self.embeddings = None
        if backend == "sbert":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("sbert retrieval requires: pip install -r requirements-full.txt") from exc
            self.model = SentenceTransformer(model_name)
            self.embeddings = self.model.encode([item["post"] for item in experiences], normalize_embeddings=True)
        elif backend != "lexical":
            raise ValueError("retrieval backend must be lexical or sbert")

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9']+", text.lower()))

    def search(self, query: str, k: int = 10) -> list[dict[str, str]]:
        if self.backend == "sbert":
            import numpy as np
            query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
            scores = np.asarray(self.embeddings) @ np.asarray(query_embedding)
            indices = np.argsort(-scores)[:k]
        else:
            query_tokens = self._tokens(query)
            scored = []
            for index, item in enumerate(self.experiences):
                tokens = self._tokens(item["post"])
                score = len(query_tokens & tokens) / max(1, len(query_tokens | tokens))
                scored.append((score, index))
            indices = [index for _, index in sorted(scored, reverse=True)[:k]]
        return [self.experiences[int(index)] for index in indices]
