from __future__ import annotations

import re
"""
Structured-output compatibility bridge for the HeLa-Mem
LongMemEval API-substituted Qwen3-8B-Base reproduction.

The official HeLa algorithm/data/retrieval remain unchanged.
This module only constrains local Qwen outputs so Base-model
prompt continuation cannot leak into memory or final answers.
"""


import json
import os
from openai import OpenAI


SEMANTIC_SCHEMA = {
    "type": "object",
    "properties": {
        "profile": {
            "type": "string",
            "maxLength": 6000,
        },
        "private_facts": {
            "type": "array",
            "items": {
                "type": "string",
                "maxLength": 500,
            },
            "maxItems": 30,
        },
        "assistant_knowledge": {
            "type": "array",
            "items": {
                "type": "string",
                "maxLength": 500,
            },
            "maxItems": 30,
        },
    },
    "required": [
        "profile",
        "private_facts",
        "assistant_knowledge",
    ],
    "additionalProperties": False,
}

KEYWORD_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {
            "type": "array",
            "items": {
                "type": "string",
                "maxLength": 80,
            },
            "maxItems": 3,
        },
    },
    "required": ["keywords"],
    "additionalProperties": False,
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "maxLength": 384,
        },
    },
    "required": ["answer"],
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {
            "type": "boolean",
        },
    },
    "required": ["correct"],
    "additionalProperties": False,
}

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "profile": {
            "type": "string",
            "maxLength": 6000,
        },
    },
    "required": ["profile"],
    "additionalProperties": False,
}


_BAD_SUBSTRINGS = (
    "<context>",
    "# personality and user data analysis task",
    "# assistant knowledge extraction task",
    "and do not exceed three:",
    "please extract the keywords of the conversation topic",
    "actual conversation:",
    "output exactly",
)


def is_qwen_model(model=None):
    model = model or os.environ.get("HEBBIAN_MODEL", "")
    return "qwen" in str(model).lower()


def _get_client(client=None):
    if client is None:
        return OpenAI(
            api_key=os.environ.get(
                "OPENAI_API_KEY",
                "local-qwen",
            ),
            base_url=os.environ.get(
                "OPENAI_BASE_URL",
                "http://127.0.0.1:8000/v1",
            ),
        )

    # Raw OpenAI client.
    if (
        hasattr(client, "chat")
        and hasattr(client.chat, "completions")
    ):
        return client

    # HeLa profile_utils.OpenAIClient wrapper.
    wrapped = getattr(client, "client", None)
    if (
        wrapped is not None
        and hasattr(wrapped, "chat")
        and hasattr(wrapped.chat, "completions")
    ):
        return wrapped

    raise TypeError(
        "Unsupported OpenAI client object for Qwen bridge"
    )


def _validate_clean_text(text):
    if not isinstance(text, str):
        raise ValueError("structured field must be a string")

    text = text.strip()
    low = text.lower()

    for bad in _BAD_SUBSTRINGS:
        if bad in low:
            raise ValueError(
                f"Qwen prompt leakage detected: {bad!r}"
            )

    for line in text.splitlines():
        line_low = line.strip().lower()
        if line_low.startswith(
            ("user:", "assistant:", "ai:", "system:")
        ):
            raise ValueError(
                "Qwen transcript leakage detected: "
                + repr(line[:200])
            )

    return text


def _validate_string_list(values):
    if not isinstance(values, list):
        raise ValueError("expected list")

    cleaned = []
    for value in values:
        value = _validate_clean_text(value)
        if value:
            cleaned.append(value)

    return cleaned


def _structured_request(
    messages,
    schema,
    schema_name,
    max_tokens,
    client=None,
    max_attempts=3,
):
    raw_client = _get_client(client)

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = raw_client.chat.completions.create(
                model=os.environ.get(
                    "HEBBIAN_MODEL",
                    "Qwen3-8B-Base",
                ),
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                extra_body={
                    "guided_json": schema,
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    },
                },
            )

            if not response or not response.choices:
                raise RuntimeError(
                    "Qwen guided_json request returned no choices"
                )

            text = response.choices[0].message.content

            if not text:
                raise RuntimeError(
                    "Qwen guided_json request returned empty content"
                )

            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Qwen guided_json output is not valid JSON: "
                    + repr(text[:500])
                ) from exc

            if not isinstance(payload, dict):
                raise RuntimeError(
                    "Qwen guided_json output is not a JSON object"
                )

            return payload

        except Exception as exc:
            last_error = exc

            if attempt < max_attempts:
                print(
                    f"[QWEN RETRY] schema={schema_name} "
                    f"attempt={attempt}/{max_attempts} "
                    f"reason={exc}",
                    flush=True,
                )

    raise RuntimeError(
        f"Qwen structured request failed after "
        f"{max_attempts} attempts for {schema_name}: "
        f"{last_error}"
    ) from last_error


def _qwen_keyword_plain_fallback(text, client=None):
    raw_client = _get_client(client)

    messages = [
        {
            "role": "user",
            "content": (
                "Extract one to three concise topic keywords "
                "from the dialogue below.\n"
                "Return EXACTLY ONE LINE.\n"
                "Separate keywords with commas.\n"
                "Do not explain.\n"
                "Do not output speaker labels.\n"
                "Do not output JSON.\n\n"
                "Dialogue:\n"
                + str(text)
                + "\n\nKeywords:"
            ),
        }
    ]

    response = raw_client.chat.completions.create(
        model=os.environ.get(
            "HEBBIAN_MODEL",
            "Qwen3-8B-Base",
        ),
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        max_tokens=32,
        stop=["\n"],
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
    )

    if not response or not response.choices:
        raise RuntimeError(
            "Qwen keyword plain fallback returned no choices"
        )

    raw = response.choices[0].message.content or ""
    line = raw.strip()

    if not line:
        raise RuntimeError(
            "Qwen keyword plain fallback returned empty output"
        )

    # Defensive cleanup.
    line = line.splitlines()[0].strip()
    line = re.sub(
        r"^\s*(?:keywords?|topics?)\s*:\s*",
        "",
        line,
        flags=re.IGNORECASE,
    )

    # Qwen Base may prepend a chat speaker label even when
    # explicitly asked for a one-line keyword response.
    # Strip that wrapper locally rather than weakening the
    # global transcript-leakage validator.
    line = re.sub(
        r"^\s*(?:assistant|ai|user|system)\s*:\s*",
        "",
        line,
        flags=re.IGNORECASE,
    )

    parts = re.split(r"[,;|]", line)

    keywords = []
    seen = set()

    for part in parts:
        kw = part.strip(" \t\"'`")

        if not kw:
            continue

        if len(kw) > 80:
            continue

        kw = _validate_clean_text(kw)

        norm = kw.casefold()

        if norm in seen:
            continue

        seen.add(norm)
        keywords.append(kw)

        if len(keywords) == 3:
            break

    if not keywords:
        raise RuntimeError(
            "Qwen keyword plain fallback produced no valid keywords"
        )

    print(
        "[QWEN FALLBACK] keyword plain-text fallback succeeded",
        flush=True,
    )

    return set(keywords)


def qwen_keywords(text, client=None):
    schema = {
        "type": "object",
        "properties": {
            "keyword_1": {"type": "string"},
            "keyword_2": {"type": "string"},
            "keyword_3": {"type": "string"},
        },
        "required": [
            "keyword_1",
            "keyword_2",
            "keyword_3",
        ],
        "additionalProperties": False,
    }

    messages = [
        {
            "role": "user",
            "content": (
                "Extract at most three concise topic keywords "
                "from the dialogue below.\n"
                "Put one keyword in each field.\n"
                "If fewer than three exist, use an empty string.\n"
                "Do not copy raw dialogue sentences.\n"
                "Do not output explanations or speaker labels.\n\n"
                "Dialogue:\n"
                + str(text)
            ),
        }
    ]

    last_error = None

    for attempt in range(1, 4):
        try:
            payload = _structured_request(
                messages,
                schema,
                "hela_keywords_fixed3",
                max_tokens=64,
                client=client,
                max_attempts=1,
            )

            keywords = []

            for key in (
                "keyword_1",
                "keyword_2",
                "keyword_3",
            ):
                kw = payload.get(key, "")

                if not isinstance(kw, str):
                    raise RuntimeError(
                        f"{key} is not a string"
                    )

                kw = kw.strip()

                if not kw:
                    continue

                kw = _validate_clean_text(kw)

                if "\n" in kw:
                    raise RuntimeError(
                        f"Multiline keyword: {kw!r}"
                    )

                if len(kw) > 80:
                    raise RuntimeError(
                        f"Keyword too long: {kw!r}"
                    )

                keywords.append(kw)

            if not keywords:
                raise RuntimeError(
                    "Qwen keyword extraction produced no valid keywords"
                )

            return set(keywords[:3])

        except Exception as exc:
            last_error = exc

            if attempt < 3:
                print(
                    f"[QWEN RETRY] schema=hela_keywords_fixed3 "
                    f"attempt={attempt}/3 reason={exc}",
                    flush=True,
                )

    print(
        "[QWEN FALLBACK] structured keyword extraction "
        f"failed after 3 attempts: {last_error}",
        flush=True,
    )

    return _qwen_keyword_plain_fallback(
        text,
        client=client,
    )


def _fixed_string_schema(*groups):
    """
    Build fixed-field JSON schemas.

    Fixed fields are intentional here: vLLM 0.10.0 + Qwen3-8B-Base
    did not reliably terminate variable-length arrays before max_tokens.
    """
    properties = {}
    required = []

    for prefix, count in groups:
        for i in range(1, count + 1):
            key = f"{prefix}_{i}"
            properties[key] = {"type": "string"}
            required.append(key)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _collect_fixed_strings(payload, prefix, count):
    values = []
    seen = set()

    for i in range(1, count + 1):
        key = f"{prefix}_{i}"
        value = payload.get(key, "")

        if not isinstance(value, str):
            raise RuntimeError(
                f"{key} is not a string"
            )

        value = value.strip()

        if not value:
            continue

        value = _validate_clean_text(value)

        norm = value.casefold()
        if norm in seen:
            continue

        seen.add(norm)
        values.append(value)

    return values


def qwen_profile_analysis(conversation, client=None):
    # USER-ONLY schema.
    # assistant_knowledge cannot be generated structurally.
    schema = _fixed_string_schema(
        ("profile", 4),
        ("private_fact", 12),
    )

    messages = [
        {
            "role": "user",
            "content": (
                "Extract semantic memory ABOUT THE USER only.\n"
                "Do not extract assistant knowledge.\n"
                "Do not copy raw dialogue lines.\n"
                "Do not invent information.\n\n"
                "profile_1 through profile_4:\n"
                "- each field contains at most one concise sentence\n"
                "- summarize observable user traits, preferences, "
                "interaction style, or values\n"
                "- use an empty string for unused fields\n\n"
                "private_fact_1 through private_fact_12:\n"
                "- each field contains exactly one explicit user fact, "
                "event, date, location, preference, or personal detail\n"
                "- keep every field concise\n"
                "- use empty strings for unused fields\n\n"
                "Conversation:\n"
                + str(conversation)
            ),
        }
    ]

    payload = _structured_request(
        messages,
        schema,
        "hela_user_semantic_fixed",
        max_tokens=1800,
        client=client,
    )

    profile_parts = _collect_fixed_strings(
        payload, "profile", 4
    )

    private_facts = _collect_fixed_strings(
        payload, "private_fact", 12
    )

    return {
        "profile": " ".join(profile_parts),
        "private_facts": private_facts,
        "assistant_knowledge": [],
    }


def _qwen_assistant_compact_fallback(
    conversation,
    client=None,
):
    schema = _fixed_string_schema(
        ("knowledge", 8),
    )

    messages = [
        {
            "role": "user",
            "content": (
                "Extract up to eight of the most useful pieces "
                "of knowledge conveyed BY THE ASSISTANT only.\n"
                "PARAPHRASE the knowledge concisely.\n"
                "Never quote or copy a dialogue line.\n"
                "Never begin a field with User:, Assistant:, "
                "AI:, or System:.\n"
                "Do not describe user traits or personal facts.\n"
                "Do not describe the assistant's identity or personality.\n"
                "Use an empty string for unused fields.\n\n"
                "Conversation:\n"
                + str(conversation)
            ),
        }
    ]

    payload = _structured_request(
        messages,
        schema,
        "hela_assistant_semantic_compact",
        max_tokens=1200,
        client=client,
        max_attempts=3,
    )

    values = _collect_fixed_strings(
        payload,
        "knowledge",
        8,
    )

    print(
        "[QWEN FALLBACK] assistant compact semantic "
        "fallback succeeded",
        flush=True,
    )

    return {
        "profile": "",
        "private_facts": [],
        "assistant_knowledge": values,
    }


def qwen_assistant_analysis(conversation, client=None):
    schema = _fixed_string_schema(
        ("knowledge", 16),
    )

    messages = [
        {
            "role": "user",
            "content": (
                "Extract concise knowledge conveyed BY THE ASSISTANT only.\n"
                "Do not extract user traits or user personal facts.\n"
                "Do not describe the assistant's personality, identity, "
                "preferences, hobbies, or traits.\n"
                "Do not copy raw dialogue lines.\n"
                "Do not invent information.\n\n"
                "knowledge_1 through knowledge_16:\n"
                "- each field contains one concise piece of information, "
                "advice, suggestion, or claim explicitly stated in an "
                "Assistant/AI utterance\n"
                "- each field should be at most one short sentence\n"
                "- use empty strings for unused fields\n\n"
                "Conversation:\n"
                + str(conversation)
            ),
        }
    ]

    try:
        payload = _structured_request(
            messages,
            schema,
            "hela_assistant_semantic_fixed",
            max_tokens=2200,
            client=client,
            max_attempts=3,
        )

        assistant_knowledge = _collect_fixed_strings(
            payload,
            "knowledge",
            16,
        )

        return {
            "profile": "",
            "private_facts": [],
            "assistant_knowledge": assistant_knowledge,
        }

    except Exception as exc:
        print(
            "[QWEN FALLBACK] assistant primary semantic "
            f"failed: {exc}",
            flush=True,
        )

        return _qwen_assistant_compact_fallback(
            conversation,
            client=client,
        )


def qwen_profile_merge(
    old_profile,
    new_analysis,
    client=None,
):
    # Fixed fields avoid one unbounded profile string consuming the
    # entire generation budget before the JSON object closes.
    schema = _fixed_string_schema(
        ("profile", 6),
    )

    messages = [
        {
            "role": "user",
            "content": (
                "Merge the existing user profile and new analysis into "
                "one concise updated user profile.\n"
                "Preserve verified information.\n"
                "Resolve conflicts in favor of newer explicit evidence.\n"
                "Do not invent information.\n"
                "Do not copy prompts or raw dialogue.\n\n"
                "profile_1 through profile_6:\n"
                "- each field contains at most one concise profile sentence\n"
                "- use empty strings for unused fields\n\n"
                "Existing profile:\n"
                f"{old_profile}\n\n"
                "New analysis:\n"
                f"{new_analysis}"
            ),
        }
    ]

    payload = _structured_request(
        messages,
        schema,
        "hela_profile_merge_fixed",
        max_tokens=1200,
        client=client,
    )

    parts = _collect_fixed_strings(
        payload, "profile", 6
    )

    return " ".join(parts)


def _qwen_plain_line(
    messages,
    *,
    max_tokens,
    client=None,
):
    raw_client = _get_client(client)

    response = raw_client.chat.completions.create(
        model=os.environ.get(
            "HEBBIAN_MODEL",
            "Qwen3-8B-Base",
        ),
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["\n"],
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
    )

    if not response or not response.choices:
        raise RuntimeError(
            "Qwen plain fallback returned no choices"
        )

    text = response.choices[0].message.content or ""
    text = text.strip()

    if not text:
        raise RuntimeError(
            "Qwen plain fallback returned empty output"
        )

    text = text.splitlines()[0].strip()

    text = re.sub(
        r"^\s*(?:assistant|ai)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    if not text:
        raise RuntimeError(
            "Qwen plain fallback empty after cleanup"
        )

    return text


def qwen_answer(messages, client=None):
    # Preserve the official HeLa prompt, with the previously
    # validated Qwen-Base extractive constraint.
    constrained = [dict(m) for m in messages]

    suffix = (
        "\n\n[QWEN OUTPUT CONSTRAINT]\n"
        "Answer the question using ONLY information explicitly supported "
        "by the supplied context or memory.\n"
        "Do NOT infer causes, speculate, explain, give advice, or add background.\n"
        "Return the shortest concrete answer phrase that directly answers "
        "the question.\n"
        "If the answer wording appears in the context, prefer that wording.\n"
        "Return only the answer itself."
    )

    for i in range(len(constrained) - 1, -1, -1):
        if constrained[i].get("role") == "user":
            constrained[i]["content"] = (
                str(constrained[i].get("content", ""))
                + suffix
            )
            break

    last_error = None

    # Task-level retry is required because valid JSON such as
    # {"answer": ""} is structurally valid but semantically unusable.
    for attempt in range(1, 4):
        try:
            payload = _structured_request(
                constrained,
                ANSWER_SCHEMA,
                "hela_answer",
                max_tokens=96,
                client=client,
                max_attempts=1,
            )

            answer = _validate_clean_text(
                payload.get("answer", "")
            ).strip()

            if not answer:
                raise RuntimeError(
                    "Qwen structured answer is empty"
                )

            return answer

        except Exception as exc:
            last_error = exc

            if attempt < 3:
                print(
                    f"[QWEN RETRY] schema=hela_answer "
                    f"attempt={attempt}/3 reason={exc}",
                    flush=True,
                )

    print(
        "[QWEN FALLBACK] structured answer failed "
        f"after 3 attempts: {last_error}",
        flush=True,
    )

    answer = _qwen_plain_line(
        constrained,
        max_tokens=96,
        client=client,
    )

    # Occasionally Base models wrap an answer as JSON even without
    # guided decoding. Recover the single answer field if possible.
    if answer.startswith("{"):
        try:
            obj = json.loads(answer)
            if isinstance(obj, dict):
                candidate = obj.get("answer", "")
                if isinstance(candidate, str) and candidate.strip():
                    answer = candidate.strip()
        except Exception:
            pass

    answer = _validate_clean_text(answer).strip()

    if not answer:
        raise RuntimeError(
            "Qwen answer fallback produced empty answer"
        )

    print(
        "[QWEN FALLBACK] plain-text answer fallback succeeded",
        flush=True,
    )

    return answer


def qwen_judge(messages, client=None):
    last_error = None

    for attempt in range(1, 4):
        try:
            payload = _structured_request(
                messages,
                JUDGE_SCHEMA,
                "hela_judge",
                max_tokens=32,
                client=client,
                max_attempts=1,
            )

            value = payload.get("correct")

            if not isinstance(value, bool):
                raise RuntimeError(
                    "Qwen judge did not return boolean correct"
                )

            return value

        except Exception as exc:
            last_error = exc

            if attempt < 3:
                print(
                    f"[QWEN RETRY] schema=hela_judge "
                    f"attempt={attempt}/3 reason={exc}",
                    flush=True,
                )

    print(
        "[QWEN FALLBACK] structured judge failed "
        f"after 3 attempts: {last_error}",
        flush=True,
    )

    text = _qwen_plain_line(
        messages,
        max_tokens=8,
        client=client,
    )

    cleaned = text.strip().lower()
    cleaned = cleaned.strip(" .,:;!?\"'`")

    if cleaned.startswith("yes"):
        print(
            "[QWEN FALLBACK] plain-text judge fallback -> yes",
            flush=True,
        )
        return True

    if cleaned.startswith("no"):
        print(
            "[QWEN FALLBACK] plain-text judge fallback -> no",
            flush=True,
        )
        return False

    raise RuntimeError(
        "Qwen plain judge fallback was not yes/no: "
        + repr(text)
    )


