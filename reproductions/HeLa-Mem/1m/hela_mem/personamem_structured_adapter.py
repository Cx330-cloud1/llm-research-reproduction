import json
import re


_BANNED_PREFIXES = (
    "user:",
    "assistant:",
    "time:",
)

_BANNED_EXACT = {
    "actual conversation:",
    "section.",
}

_BANNED_SUBSTRINGS = (
    "actual conversation",
    "extract only statements",
    "analyze only",
    "output exactly",
)


def validate_semantic_facts(facts):
    if not isinstance(facts, list):
        raise ValueError("semantic facts must be a list")

    for fact in facts:
        if not isinstance(fact, str):
            raise ValueError("semantic fact must be a string")

        text = fact.strip()
        low = text.lower()

        if not text:
            raise ValueError("empty semantic fact")

        if low in _BANNED_EXACT:
            raise ValueError(f"prompt leakage detected: {text!r}")

        if low.startswith(_BANNED_PREFIXES):
            raise ValueError(f"transcript leakage detected: {text!r}")

        if any(x in low for x in _BANNED_SUBSTRINGS):
            raise ValueError(f"prompt leakage detected: {text!r}")

    return facts


def parse_semantic_payload(raw):
    if not isinstance(raw, str):
        raise ValueError("semantic payload must be text")

    text = raw.strip()

    # Allow ```json ... ``` emitted by the model.
    m = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        text = m.group(1).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("semantic payload is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("semantic payload must be a JSON object")

    required = {
        "profile",
        "private_facts",
        "assistant_knowledge",
    }

    if set(payload.keys()) != required:
        raise ValueError(
            "semantic payload must contain exactly: "
            "profile, private_facts, assistant_knowledge"
        )

    if not isinstance(payload["profile"], str):
        raise ValueError("profile must be a string")

    profile = payload["profile"].strip()
    if profile:
        validate_semantic_facts([profile])

    validate_semantic_facts(payload["private_facts"])
    validate_semantic_facts(payload["assistant_knowledge"])

    return payload


def render_profile_for_official_parser(payload):
    validate_semantic_facts(payload["private_facts"])

    profile = payload["profile"].strip() or "None"
    facts = payload["private_facts"]

    user_data = (
        "\n".join(f"- {fact}" for fact in facts)
        if facts
        else "None"
    )

    return (
        "【User Profile】\n"
        f"{profile}\n\n"
        "【User Data】\n"
        f"{user_data}"
    )


def render_assistant_for_official_parser(payload):
    validate_semantic_facts(payload["assistant_knowledge"])

    facts = payload["assistant_knowledge"]

    body = (
        "\n".join(f"- {fact}" for fact in facts)
        if facts
        else "None"
    )

    return (
        "【Assistant Knowledge】\n"
        f"{body}"
    )
