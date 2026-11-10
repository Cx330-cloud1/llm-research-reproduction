from hela_mem.local_qwen_bridge import (
    get_llm,
    qwen_complete,
    qwen_extract_keywords,
)

get_llm()

tests = [
    """User: I bought a Toyota and the brakes are making noise.
AI: You should have the braking system inspected.""",

    """User: I've been experimenting with electronic music and Pacific sounds.
AI: That sounds like an interesting fusion of styles.""",

    """The user enjoys digital music remixing and technology.""",

    """None""",

    """""",
]

for i, text in enumerate(tests, 1):
    print("\n" + "=" * 70)
    print("TEST", i)
    print("INPUT:", repr(text))

    prompt = f"""Extract up to three short topic keywords from the dialogue.

Example:
Dialogue:
User: I bought a Honda and the brakes are making noise.
AI: Have the braking system inspected.
Keywords: Honda, brakes, car maintenance

Dialogue:
{text}
Keywords:"""

    try:
        raw = qwen_complete(
            prompt,
            max_tokens=32,
            temperature=0.0,
        )
        print("RAW:", repr(raw))
        print("SPLITLINES:", raw.splitlines())
    except Exception as e:
        print(
            "QWEN_COMPLETE ERROR:",
            type(e).__name__,
            repr(e),
        )
        continue

    try:
        kw = qwen_extract_keywords(text)
        print("KEYWORDS:", kw)
    except Exception as e:
        print(
            "EXTRACT ERROR:",
            type(e).__name__,
            repr(e),
        )

print("\nDONE")
