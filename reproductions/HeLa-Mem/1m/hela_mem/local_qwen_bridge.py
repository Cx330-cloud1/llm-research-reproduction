import os
# PERSONAMEM_PERSISTENT_QWEN_CACHE
import hashlib
import json
import sqlite3
import threading
import time

import re
from vllm import LLM, SamplingParams

_MODEL_PATH = os.environ.get("HEBBIAN_QWEN_MODEL")
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        if not _MODEL_PATH:
            raise RuntimeError(
                "HEBBIAN_QWEN_MODEL is not configured. "
                "Please set it to your local Qwen model path."
            )

        _llm = LLM(
            model=_MODEL_PATH,
            dtype="auto",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=16384,
            enable_prefix_caching=True,
            guided_decoding_backend="guidance",
            guided_decoding_disable_fallback=True,
        )
    return _llm



_QWEN_CACHE_VERSION = "personamem-qwen-cache-v1"
_QWEN_CACHE_PATH = os.environ.get(
    "PERSONAMEM_QWEN_CACHE",
    "personamem_qwen_cache.sqlite3",
)
_QWEN_CACHE_LOCK = threading.Lock()


def _qwen_cache_log(msg):
    if os.environ.get("PERSONAMEM_QWEN_CACHE_VERBOSE", "0") == "1":
        print(msg)


def _qwen_cache_key(kind, payload):
    canonical = json.dumps(
        {
            "version": _QWEN_CACHE_VERSION,
            "model": _MODEL_PATH,
            "kind": kind,
            "payload": payload,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def _qwen_cache_connect():
    conn = sqlite3.connect(
        _QWEN_CACHE_PATH,
        timeout=30,
    )

    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS qwen_cache (
            cache_key TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )

    return conn


def _qwen_cache_get(kind, payload):
    # Never memoize stochastic requests.
    if float(payload.get("temperature", 0.0)) != 0.0:
        return None

    key = _qwen_cache_key(kind, payload)

    with _QWEN_CACHE_LOCK:
        conn = _qwen_cache_connect()
        try:
            row = conn.execute(
                "SELECT response FROM qwen_cache "
                "WHERE cache_key = ?",
                (key,),
            ).fetchone()
        finally:
            conn.close()

    if row is None:
        _qwen_cache_log(
            f"[QWEN CACHE MISS] {kind} {key[:12]}"
        )
        return None

    _qwen_cache_log(
        f"[QWEN CACHE HIT] {kind} {key[:12]}"
    )

    return row[0]


def _qwen_cache_put(kind, payload, response):
    if float(payload.get("temperature", 0.0)) != 0.0:
        return

    if not isinstance(response, str) or not response.strip():
        return

    key = _qwen_cache_key(kind, payload)

    with _QWEN_CACHE_LOCK:
        conn = _qwen_cache_connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO qwen_cache
                (cache_key, kind, response, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    key,
                    kind,
                    response,
                    time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()




def qwen_complete_batch(
    prompts,
    max_tokens=48,
    temperature=0.0,
    batch_size=16,
):
    prompts = list(prompts)

    if not prompts:
        return []

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    stop = ["\n\n", "Dialogue:", "Example:"]
    results = [None] * len(prompts)

    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )

    llm = get_llm()

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]

        pending_prompts = []
        pending_payloads = []
        pending_indices = []

        for local_idx, prompt in enumerate(chunk):
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
            }

            global_idx = start + local_idx

            cached = _qwen_cache_get("complete", payload)

            if cached is not None:
                results[global_idx] = cached
            else:
                pending_prompts.append(prompt)
                pending_payloads.append(payload)
                pending_indices.append(global_idx)

        if not pending_prompts:
            continue

        outputs = llm.generate(
            pending_prompts,
            params,
            use_tqdm=False,
        )

        if len(outputs) != len(pending_prompts):
            raise RuntimeError("Batched Qwen output count mismatch")

        for idx, output in enumerate(outputs):
            if not output.outputs:
                raise RuntimeError(
                    f"Empty Qwen output at batch index {idx}"
                )

            text = output.outputs[0].text.strip()
            global_idx = pending_indices[idx]

            results[global_idx] = text

            _qwen_cache_put(
                "complete",
                pending_payloads[idx],
                text,
            )

    return results


def qwen_complete(prompt, max_tokens=48, temperature=0.0):
    return qwen_complete_batch(
        [prompt],
        max_tokens=max_tokens,
        temperature=temperature,
    )[0]

def qwen_generate(messages, max_tokens=256, temperature=0.0):
    cache_payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template": "default",
        "add_generation_prompt": True,
    }

    cached = _qwen_cache_get(
        "chat",
        cache_payload,
    )
    if cached is not None:
        return cached

    llm = get_llm()
    tokenizer = llm.get_tokenizer()

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    out = llm.generate([prompt], params, use_tqdm=False)
    text = out[0].outputs[0].text.strip()

    _qwen_cache_put(
        "chat",
        cache_payload,
        text,
    )

    return text


def _clean_keyword(x):
    x = x.strip()
    x = re.sub(r"^[\-\*\d\.\)\s]+", "", x)
    x = re.sub(r"^(keywords?|assistant)\s*:\s*", "", x, flags=re.I)
    x = x.strip(" \"'[]{}()")

    # Reject obvious prompt echoes / sentences
    bad = [
        "extract the keyword",
        "return comma",
        "at most three",
        "following dialogue",
        "conversation topic",
    ]
    if any(b in x.lower() for b in bad):
        return None

    if not x or len(x) > 50:
        return None

    return x



def qwen_extract_keywords_batch(texts, batch_size=16):
    texts = list(texts)
    results = [set() for _ in texts]

    valid_items = []

    for idx, text in enumerate(texts):
        if text is None:
            continue

        text = str(text).strip()

        if not text or text.lower() == "none":
            continue

        prompt = f"""Extract up to three short topic keywords from the dialogue.

Example:
Dialogue:
User: I bought a Honda and the brakes are making noise.
AI: Have the braking system inspected.
Keywords: Honda, brakes, car maintenance

Dialogue:
{text}
Keywords:"""

        valid_items.append((idx, text, prompt))

    if not valid_items:
        return results

    prompts = [item[2] for item in valid_items]

    raw_outputs = qwen_complete_batch(
        prompts,
        max_tokens=32,
        temperature=0.0,
        batch_size=batch_size,
    )

    empty_positions = [
        i for i, raw in enumerate(raw_outputs)
        if not raw.strip()
    ]

    if empty_positions:
        retry_outputs = qwen_complete_batch(
            [prompts[i] for i in empty_positions],
            max_tokens=32,
            temperature=0.0,
            batch_size=batch_size,
        )

        for pos, raw in zip(empty_positions, retry_outputs):
            raw_outputs[pos] = raw

    for item, raw in zip(valid_items, raw_outputs):
        original_idx, text, _ = item

        if not raw.strip():
            print(
                "[KW WARNING] returning empty keyword set for:",
                repr(text[:200]),
            )
            continue

        lines = raw.splitlines()

        if not lines:
            continue

        parts = re.split(r"[,;|]", lines[0])

        keywords = []
        seen = set()

        for part in parts:
            kw = _clean_keyword(part)

            if not kw:
                continue

            if kw.strip().lower() in {
                "none", "null", "n/a",
                "no keywords", "no keyword",
            }:
                continue

            key = kw.lower()

            if key not in seen:
                seen.add(key)
                keywords.append(kw)

            if len(keywords) == 3:
                break

        results[original_idx] = set(keywords)

    return results


def qwen_extract_keywords(text):
    return qwen_extract_keywords_batch([text])[0]
