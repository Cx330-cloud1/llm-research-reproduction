import os
import json
import shutil
import argparse
import tempfile
from pathlib import Path

import pandas as pd

# ============================================================
# PersonaMem adaptation settings
# ============================================================

os.environ["HEBBIAN_LOCAL_QWEN"] = "1"
os.environ.setdefault(
    "HEBBIAN_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# PersonaMem has no real timestamp information.
# Do not invent temporal ranking signal.
os.environ["HEBBIAN_USE_TIME_DECAY"] = "false"

os.environ.setdefault("HEBBIAN_USE_KEYWORD_MATCH", "true")
os.environ.setdefault("HEBBIAN_LEARNING_RATE", "0.02")
os.environ.setdefault("HEBBIAN_DECAY_RATE", "0.995")
os.environ.setdefault("HEBBIAN_ACTIVATION_ALPHA", "0.1")
os.environ.setdefault("HEBBIAN_SPREADING_THRESHOLD", "0.4")
os.environ.setdefault("HEBBIAN_MAX_FLIPPED", "3")
os.environ.setdefault("HEBBIAN_KEYWORD_WEIGHT", "0.7")
os.environ.setdefault("HEBBIAN_KNOWLEDGE_BUFFER_SIZE", "10")

# ============================================================
# Reuse the already verified PersonaMem adapter utilities
# ============================================================

from eval_personamem_local import (
    QUESTION_PATH,
    CONTEXT_PATH,
    load_rows_with_context,
    context_to_turns,
    format_retrieval,
    INSTRUCTION,
    generate_answers,
    extract_answer,
    report,
)

# ============================================================
# Official HeLa modules
# ============================================================

from hela_mem.hebbian_memory import HebbianMemoryGraph
from hela_mem.hebbian_retriever import HebbianRetriever
from hela_mem.hebbian_knowledge_memory import HebbianKnowledgeMemory

# Reuse OFFICIAL knowledge extraction workflow unchanged.
from hela_mem.encode_longmemeval import (
    process_incremental_buffer,
    BUFFER_SIZE,
)

# Local API replacement only.
from hela_mem.local_qwen_bridge import (
    get_llm,
    qwen_generate,
    _qwen_cache_get,
    _qwen_cache_put,
)

from hela_mem.personamem_structured_adapter import (
    parse_semantic_payload,
    render_profile_for_official_parser,
    render_assistant_for_official_parser,
)


# ============================================================
# OpenAI-compatible interface replacement
# ============================================================


from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams



# Original schema: retain compatibility with existing valid cache entries.
LEGACY_SEMANTIC_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "profile": {"type": "string"},
        "private_facts": {
            "type": "array",
            "items": {"type": "string"},
        },
        "assistant_knowledge": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "profile",
        "private_facts",
        "assistant_knowledge",
    ],
    "additionalProperties": False,
}

# Bounded schema: prevent unbounded repetitive list generation.
SEMANTIC_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "profile": {
            "type": "string",
            "maxLength": 2500,
        },
        "private_facts": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "string",
                "maxLength": 200,
            },
        },
        "assistant_knowledge": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "string",
                "maxLength": 200,
            },
        },
    },
    "required": [
        "profile",
        "private_facts",
        "assistant_knowledge",
    ],
    "additionalProperties": False,
}


def _guided_semantic_generate(messages, max_tokens):
    # Read old-schema cache entries first so valid results are preserved.
    legacy_payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "guided_backend": "guidance",
        "schema": LEGACY_SEMANTIC_JSON_SCHEMA,
        "chat_template": "default",
        "add_generation_prompt": True,
        "enable_thinking": False,
    }

    cached = _qwen_cache_get("guided_semantic", legacy_payload)

    if cached is not None:
        try:
            parse_semantic_payload(cached)
        except Exception as exc:
            print(
                "[SEMANTIC CACHE REJECT] legacy entry failed validation: "
                + str(exc)[:200]
            )
        else:
            return cached

    # New entries use a bounded schema and a matching cache key.
    effective_max_tokens = max(int(max_tokens), 3072)

    cache_payload = {
        "messages": messages,
        "max_tokens": effective_max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "guided_backend": "guidance",
        "schema": SEMANTIC_JSON_SCHEMA,
        "chat_template": "default",
        "add_generation_prompt": True,
        "enable_thinking": False,
    }

    cached = _qwen_cache_get("guided_semantic", cache_payload)

    if cached is not None:
        try:
            parse_semantic_payload(cached)
        except Exception as exc:
            print(
                "[SEMANTIC CACHE REJECT] bounded entry failed validation: "
                + str(exc)[:200]
            )
        else:
            return cached

    llm = get_llm()
    tokenizer = llm.get_tokenizer()

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    guided = GuidedDecodingParams(json=SEMANTIC_JSON_SCHEMA)

    params = SamplingParams(
        max_tokens=effective_max_tokens,
        temperature=0.0,
        top_p=1.0,
        guided_decoding=guided,
    )

    out = llm.generate([prompt], params, use_tqdm=False)

    if not out or not out[0].outputs:
        raise RuntimeError("Guided semantic generation returned no output")

    text = out[0].outputs[0].text.strip()

    # Never cache a result that fails semantic validation.
    try:
        parse_semantic_payload(text)
    except Exception as exc:
        print(
            "[STRUCTURED SEMANTIC REJECT] generated output: "
            + repr(text[-300:])
        )
        raise RuntimeError(
            "Generated semantic payload failed validation; not cached"
        ) from exc

    _qwen_cache_put("guided_semantic", cache_payload, text)
    return text

class LocalQwenClient:
    """
    Local Qwen3-8B-Base replacement for the OpenAI client expected
    by the official HeLa profile/knowledge code.

    Official HeLa modules remain unchanged.
    """

    @staticmethod
    def _actual_conversation(messages):
        for m in reversed(messages):
            if m.get("role") != "user":
                continue

            content = str(m.get("content", ""))

            if "Conversation:" in content:
                return content.rsplit(
                    "Conversation:",
                    1,
                )[1].strip()

        return ""

    def chat_completion(
        self,
        model,
        messages,
        temperature=0.7,
        max_tokens=2000,
    ):
        joined = "\n".join(
            str(m.get("content", ""))
            for m in messages
        )

        is_profile_analysis = (
            "Personality and User Data Analysis Task"
            in joined
        )

        is_assistant_analysis = (
            "Assistant Knowledge Extraction Task"
            in joined
        )

        effective_messages = messages

        # ----------------------------------------------------
        # Qwen-Base can copy facts from official few-shot
        # examples. Preserve the HeLa task semantics but remove
        # example text before sending the request to Qwen.
        # ----------------------------------------------------
        if is_profile_analysis:
            conversation = self._actual_conversation(
                messages
            )

            effective_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a personality and user data "
                        "analysis engine. Extract ONLY information "
                        "explicitly supported by the supplied ACTUAL "
                        "CONVERSATION. Never invent facts."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
Analyze ONLY the ACTUAL CONVERSATION below.

Output in EXACTLY this format:

【User Profile】
1. Core Psychological Traits:
   - [Trait]: [Positive/Negative/Neutral] (Evidence)

2. Content Preferences:
   - [Topic]: [Like/Dislike/Neutral] (Evidence)

3. Interaction Style:
   - [Style]: [Preference] (Evidence)

4. Value Alignment:
   - [Value]: [Strong/Weak] (Evidence)

【User Data】
- [Fact]: [Details]
- If no explicit factual user data exists, write "None".

Rules:
1. Use ONLY facts explicitly present in ACTUAL CONVERSATION.
2. Do not infer dates, locations, food preferences, movie preferences,
   events, or personal facts unless actually stated.
3. Keep the result concise.
4. Produce exactly one 【User Profile】 section and exactly one
   【User Data】 section.

ACTUAL CONVERSATION:
{conversation}
""".strip(),
                },
            ]

        elif is_assistant_analysis:
            conversation = self._actual_conversation(
                messages
            )

            effective_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant knowledge extraction "
                        "engine. Extract ONLY explicit statements "
                        "about the assistant from the supplied ACTUAL "
                        "CONVERSATION. Never invent facts."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
Analyze ONLY the ACTUAL CONVERSATION below.

Output exactly:

【Assistant Knowledge】
- [Fact]
- Or "None" if no explicit assistant trait or knowledge can be extracted.

Rules:
1. Do not use any external or example information.
2. Extract only statements supported by ACTUAL CONVERSATION.
3. Keep statements concise and factual.

ACTUAL CONVERSATION:
{conversation}
""".strip(),
                },
            ]


        # ----------------------------------------------------
        # PersonaMem/Qwen semantic extraction.
        #
        # IMPORTANT:
        # Qwen3-8B-Base tends to echo system prompts, so semantic
        # extraction uses ONE user message plus vLLM guided JSON.
        # ----------------------------------------------------
        if is_profile_analysis:
            conversation = self._actual_conversation(messages)

            effective_messages = [
                {
                    "role": "user",
                    "content": (
                        "Extract a concise user profile and explicit user "
                        "facts from the conversation below.\n\n"
                        "Do not copy raw dialogue lines.\n"
                        "Do not invent information.\n"
                        "The profile should summarize stable preferences, "
                        "traits, interests, and interaction style.\n"
                        "private_facts should contain only concise facts "
                        "explicitly supported by the conversation.\n"
                        "assistant_knowledge must be empty.\n\n"
                        "Conversation:\n"
                        + conversation
                    ),
                }
            ]

        elif is_assistant_analysis:
            conversation = self._actual_conversation(messages)

            effective_messages = [
                {
                    "role": "user",
                    "content": (
                        "Extract concise knowledge conveyed BY the assistant "
                        "from the conversation below.\n\n"
                        "Do not copy raw dialogue lines.\n"
                        "Do not invent information.\n"
                        "Do NOT describe the assistant's personality, preferences, "
                        "identity, hobbies, or traits.\n"
                        "Do NOT transfer any user attribute to the assistant.\n"
                        "Each item must summarize information, advice, suggestions, "
                        "or claims explicitly stated in an Assistant: utterance.\n"
                        "profile must be empty.\n"
                        "private_facts must be empty.\n"
                        "assistant_knowledge should contain only concise propositions "
                        "supported directly by Assistant: utterances.\n\n"
                        "Conversation:\n"
                        + conversation
                    ),
                }
            ]

        if is_profile_analysis or is_assistant_analysis:
            text = _guided_semantic_generate(
                effective_messages,
                max_tokens=max_tokens,
            )
        else:
            text = qwen_generate(
                effective_messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )

        # ----------------------------------------------------
        # Strict semantic boundary:
        # Qwen JSON -> validation -> official HeLa text format.
        # Invalid model output is NEVER written to memory.
        # ----------------------------------------------------
        if is_profile_analysis or is_assistant_analysis:
            try:
                payload = parse_semantic_payload(text)
            except ValueError as exc:
                print(
                    "[STRUCTURED SEMANTIC ERROR] "
                    + repr(text[:1000])
                )
                raise RuntimeError(
                    "Qwen semantic extraction failed structured validation"
                ) from exc

            if is_profile_analysis:
                text = render_profile_for_official_parser(payload)
            else:
                text = render_assistant_for_official_parser(payload)

        return text

def add_semantic_context(
    history,
    knowledge_memory,
    question,
    item_id,
    semantic_top_k=5,
):
    """
    Preserve HeLa's official semantic/profile/assistant memory layers,
    but expose them through the same RF-Mem generation interface.
    """

    # Semantic knowledge
    try:
        kb_results = knowledge_memory.search_knowledge(
            question,
            top_k=semantic_top_k,
        )
    except Exception as e:
        print(f"[{item_id}] semantic retrieval error: {e}")
        kb_results = []

    if kb_results:
        text = "<KNOWLEDGE BASE>\n"
        text += "\n".join(
            f"- {x['knowledge']}"
            for x in kb_results
        )
        history.append(text)

    # User profile
    profile = knowledge_memory.get_raw_user_profile(item_id)

    if profile:
        history.append(
            "<CHARACTER TRAITS>\n"
            "Characteristics of the user:\n"
            + profile
        )

    # Assistant knowledge
    ak = knowledge_memory.get_assistant_knowledge()

    if ak:
        text = (
            "<ASSISTANT KNOWLEDGE>\n"
            "Here are some of your character traits and knowledge:\n"
        )

        for item in ak:
            k = item.get("knowledge", "").strip()
            if k:
                text += f"- {k}\n"

        history.append(text)

    return kb_results


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_items",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--semantic_top_k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        default="results_hela_personamem_full_sanity1.csv",
    )

    args = parser.parse_args()

    # ========================================================
    # ONE shared Qwen instance for:
    # keyword extraction
    # profile extraction
    # private knowledge extraction
    # assistant knowledge extraction
    # profile merging
    # final answer generation
    # ========================================================

    llm = get_llm()

    local_client = LocalQwenClient()

    temp_root = tempfile.mkdtemp(
        prefix="hela_personamem_full_"
    )

    rows = []
    prompts = []
    histories = []

    try:

        iterator = load_rows_with_context(
            str(QUESTION_PATH),
            str(CONTEXT_PATH),
        )

        for idx, (row, context) in enumerate(iterator):

            if idx >= args.num_items:
                break

            qid = row["question_id"]
            question = row["user_question_or_message"]
            correct_answer = row["correct_answer"]
            all_options = row["all_options"]

            end_idx = int(
                row["end_index_in_shared_context"]
            )

            # Strict no-future-leak assertion.
            assert len(context) == end_idx, (
                qid,
                len(context),
                end_idx,
            )

            turns = context_to_turns(context)

            print(
                f"\n{'=' * 70}\n"
                f"[{idx + 1}/{args.num_items}] {qid}\n"
                f"messages={len(context)}, turns={len(turns)}\n"
                f"{'=' * 70}"
            )

            # ------------------------------------------------
            # Episodic Hebbian Memory
            # ------------------------------------------------

            mem_path = os.path.join(
                temp_root,
                f"{qid}_hebbian.json",
            )

            graph = HebbianMemoryGraph(
                file_path=mem_path,
            )

            # ------------------------------------------------
            # Official Semantic/Knowledge Memory
            # ------------------------------------------------

            kb_path = os.path.join(
                temp_root,
                f"{qid}_long_term.json",
            )

            knowledge_memory = HebbianKnowledgeMemory(
                file_path=kb_path,
            )

            retriever = HebbianRetriever(
                graph,
                profile_memory=knowledge_memory,
            )

            # PersonaMem contains no genuine timestamps.
            # "unknown" avoids injecting a fake date into profile extraction.
            neutral_timestamp = "unknown"

            incremental_buffer = []

            # ------------------------------------------------
            # Official encoding semantics
            # ------------------------------------------------

            for turn_idx, (user_text, assistant_text) in enumerate(
                turns
            ):

                retriever.process_conversation_turn(
                    user_text,
                    assistant_text,
                    timestamp=neutral_timestamp,
                )

                incremental_buffer.append(
                    {
                        "user_text": user_text,
                        "ai_text": assistant_text,
                        "timestamp": neutral_timestamp,
                    }
                )

                # EXACT official BUFFER_SIZE workflow.
                if len(incremental_buffer) >= BUFFER_SIZE:

                    print(
                        f"  knowledge extraction "
                        f"through turn {turn_idx + 1}"
                    )

                    process_incremental_buffer(
                        incremental_buffer,
                        knowledge_memory,
                        qid,
                        local_client,
                    )

                    incremental_buffer = []

            # Remaining turns
            if incremental_buffer:

                print(
                    "  knowledge extraction "
                    "for remaining turns"
                )

                process_incremental_buffer(
                    incremental_buffer,
                    knowledge_memory,
                    qid,
                    local_client,
                )

            # ------------------------------------------------
            # Official episodic retrieval
            # ------------------------------------------------

            episodic_results = graph.retrieve(
                question,
                top_k=args.top_k,
            )

            history = format_retrieval(
                episodic_results
            )

            # ------------------------------------------------
            # Official semantic/profile retrieval
            # ------------------------------------------------

            kb_results = add_semantic_context(
                history,
                knowledge_memory,
                question,
                qid,
                semantic_top_k=args.semantic_top_k,
            )

            # ------------------------------------------------
            # SAME PersonaMem question prompt as RF-Mem
            # ------------------------------------------------

            prompt = (
                question
                + "\n\n"
                + INSTRUCTION
                + "\n\n"
                + all_options
            )

            prompts.append(prompt)
            histories.append(history)

            rows.append(
                {
                    "score": False,
                    "persona_id": row["persona_id"],
                    "question_id": qid,
                    "question": question,
                    "question_type": row["question_type"],
                    "topic": row["topic"],
                    "correct_answer": correct_answer,
                    "model_response": "",
                    "predicted_answer": "",
                    "context_messages": len(context),
                    "num_turns": len(turns),
                    "episodic_retrieved": len(
                        episodic_results
                    ),
                    "semantic_retrieved": len(
                        kb_results
                    ),
                    "profile_present": bool(
                        knowledge_memory
                        .get_raw_user_profile(qid)
                    ),
                    "assistant_knowledge_count": len(
                        knowledge_memory
                        .get_assistant_knowledge()
                    ),
                    "knowledge_nodes": len(
                        knowledge_memory
                        .knowledge_graph.nodes
                    ),
                    "end_index_in_shared_context": end_idx,
                }
            )

        print(
            "\nAll memory construction finished:",
            len(rows),
        )

        # ====================================================
        # SAME Qwen generation settings as RF-Mem
        # ====================================================

        responses = generate_answers(
            llm,
            prompts,
            histories,
        )

        for i, response in enumerate(responses):

            score, predicted = extract_answer(
                response,
                rows[i]["correct_answer"],
            )

            rows[i]["score"] = bool(score)
            rows[i]["model_response"] = response
            rows[i]["predicted_answer"] = predicted

            print("\n------------------------------")
            print("Q:", rows[i]["question"])
            print("Response:", response)
            print("Predicted:", predicted)
            print("Correct:", rows[i]["correct_answer"])
            print("Score:", score)
            print(
                "Episodic retrieved:",
                rows[i]["episodic_retrieved"],
            )
            print(
                "Semantic retrieved:",
                rows[i]["semantic_retrieved"],
            )
            print(
                "Knowledge nodes:",
                rows[i]["knowledge_nodes"],
            )
            print(
                "Profile present:",
                rows[i]["profile_present"],
            )

        df = pd.DataFrame(rows)

        df.to_csv(
            args.output,
            index=False,
            encoding="utf-8-sig",
        )

        json_path = str(
            Path(args.output).with_suffix(".json")
        )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                rows,
                f,
                ensure_ascii=False,
                indent=2,
            )

        report(df)

        print("\nSaved:")
        print(args.output)
        print(json_path)

    finally:

        shutil.rmtree(
            temp_root,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
