from eval_personamem_local import (
    QUESTION_PATH,
    CONTEXT_PATH,
    load_rows_with_context,
    context_to_turns,
)

from eval_personamem_local_full import LocalQwenClient
from hela_mem.profile_utils import gpt_personality_analysis
from hela_mem.local_qwen_bridge import get_llm

get_llm()
client = LocalQwenClient()

row, context = next(
    load_rows_with_context(
        str(QUESTION_PATH),
        str(CONTEXT_PATH),
    )
)

turns = context_to_turns(context)[:10]

dialogs = [
    {
        "user_input": u,
        "agent_response": a,
        "timestamp": "unknown",
    }
    for u, a in turns
]

result = gpt_personality_analysis(
    dialogs,
    client,
)

print("\n=== PROFILE SMOKE RESULT ===")
print("profile present:", bool(result["profile"].strip()))
print("private present:", bool(result["private"].strip()))
print(
    "assistant knowledge present:",
    bool(result["assistant_knowledge"].strip()),
)

print("\nPROFILE PREVIEW:")
print(result["profile"][:300])

print("\nPRIVATE PREVIEW:")
print(result["private"][:300])

print("\nPASS")

combined = (
    result["profile"]
    + "\n"
    + result["private"]
    + "\n"
    + result["assistant_knowledge"]
).lower()

for forbidden in [
    "april 1st, 2025",
    "new york",
    "likes pizza",
    "sci-fi movies",
    "dislikes rainy",
]:
    assert forbidden not in combined, (
        f"FEW-SHOT LEAK DETECTED: {forbidden}"
    )

print("NO FEW-SHOT LEAK: PASS")

combined = (
    result["profile"]
    + "\n"
    + result["private"]
    + "\n"
    + result["assistant_knowledge"]
).lower()

for forbidden in [
    "interstellar",
    "extensive knowledge of cooking",
    "cooking recipes and techniques",
]:
    assert forbidden not in combined, (
        f"ASSISTANT FEW-SHOT LEAK DETECTED: {forbidden}"
    )

print("NO ASSISTANT FEW-SHOT LEAK: PASS")
