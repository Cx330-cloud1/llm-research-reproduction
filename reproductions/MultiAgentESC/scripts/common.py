from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["_root"] = str(path.resolve().parent)
    return config


def resolve_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["_root"]) / path


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def completed_ids(path: Path) -> set[str]:
    return {
        str(row["target_id"])
        for row in read_jsonl(path)
        if "target_id" in row and row.get("status", "ok") == "ok"
    }


def extract_field(text: str, name: str) -> str:
    match = re.search(rf"(?im)^\s*{re.escape(name)}\s*:\s*(.+)$", text)
    return match.group(1).strip() if match else ""


def clean_response(text: str) -> str:
    response = extract_field(text, "Response") or text.strip()
    response = re.sub(r"^\s*\[[^\]]+\]\s*", "", response)
    return response.strip().strip('"')


def parse_strategies(text: str, canonical: list[str], limit: int = 3) -> list[str]:
    found = []
    lower = text.lower()
    for strategy in canonical:
        if strategy.lower() in lower and strategy not in found:
            found.append(strategy)
    return found[:limit]


@dataclass
class CallResult:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    elapsed_seconds: float


class LLMClient:
    def __init__(self, provider: dict[str, Any], model: str, log_path: Path, seed: int = 42):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing dependency: run 'pip install -r requirements.txt'") from exc
        api_key = os.getenv(provider.get("api_key_env", ""), provider.get("api_key_default", "ollama"))
        self.client = OpenAI(
            base_url=provider["base_url"],
            api_key=api_key,
            timeout=provider.get("timeout_seconds", 300),
        )
        self.model = model
        self.log_path = log_path
        self.retries = int(provider.get("retries", 3))
        self.seed = seed

    def complete(
        self,
        prompt: str,
        *,
        system: str = "You are a psychological counseling expert.",
        temperature: float = 0.0,
        max_tokens: int = 400,
        tag: str = "generation",
    ) -> CallResult:
        started = time.time()
        error = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=self.seed,
                )
                text = response.choices[0].message.content or ""
                usage = response.usage
                result = CallResult(
                    text=text,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    elapsed_seconds=time.time() - started,
                )
                append_jsonl(self.log_path, {
                    "tag": tag,
                    "model": self.model,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "elapsed_seconds": round(result.elapsed_seconds, 3),
                    "status": "ok",
                })
                return result
            except Exception as exc:  # provider errors vary
                error = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 8))
        append_jsonl(self.log_path, {"tag": tag, "model": self.model, "status": "error", "error": repr(error)})
        raise RuntimeError(f"LLM call failed after {self.retries} attempts: {error}")


class MockClient:
    """Offline pipeline test only. Its output must never be reported as an experiment."""

    def __init__(self, model: str = "MOCK"):
        self.model = model
        self.rng = random.Random(42)

    def complete(self, prompt: str, **_: Any) -> CallResult:
        lower = prompt.lower()
        if "yes or no" in lower or 'reply "yes"' in lower:
            text = "1. YES\n2. Emotion, event, and intention are present."
        elif "emotion:" in prompt and "infer the emotional state" in lower:
            text = "Emotion: anxious\nReasoning: The user describes uncertainty and distress."
        elif "event:" in prompt and "specific event" in lower:
            text = "Event: an unresolved stressful situation\nReasoning: It is the stated source of distress."
        elif "intention:" in prompt and "infer the user" in lower:
            text = "Intention: seek practical and emotional support\nReasoning: The user is asking for help."
        elif "fluency:" in lower and "identification:" in lower and "a:" in lower:
            text = "\n".join([
                "Fluency: 4, 4, 4, 4, 4; reasons",
                "Identification: 3, 3, 3, 3, 3; reasons",
                "Comforting: 4, 4, 4, 4, 4; reasons",
                "Suggestion: 3, 3, 3, 3, 3; reasons",
                "Overall: 4, 4, 4, 4, 4; reasons",
            ])
        elif "strategy:" in lower and "response:" in lower:
            text = "Strategy: [Providing Suggestions]\nReasoning: It balances empathy and action.\nResponse: [Providing Suggestions] That sounds difficult; could one small next step make the situation feel more manageable?"
        elif "strategy:" in lower:
            text = "Strategy: [Providing Suggestions]\nReasoning: Practical support may help."
        elif "feedback:" in lower:
            text = "Feedback: Add empathy and one gentle, actionable question."
        else:
            text = "Response: That sounds difficult; what kind of support would feel most helpful right now?"
        return CallResult(text=text, prompt_tokens=None, completion_tokens=None, elapsed_seconds=0.0)
