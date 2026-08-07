from __future__ import annotations

import re
from typing import Any

from .common import clean_response, extract_field, parse_strategies


STRATEGIES = {
    "Question": "Ask for information that helps the user articulate the problem.",
    "Restatement or Paraphrasing": "Concisely rephrase the user's statements.",
    "Reflection of feelings": "Articulate and describe the user's feelings.",
    "Self-disclosure": "Share a similar experience or emotion to express empathy.",
    "Affirmation and Reassurance": "Affirm strengths and provide encouragement.",
    "Providing Suggestions": "Offer gentle, non-prescriptive suggestions.",
    "Information": "Provide useful facts, resources, or answers.",
    "Others": "Use other appropriate supportive conversational acts.",
}
STRATEGY_LIST = list(STRATEGIES)
STRATEGY_DEFINITIONS = "\n".join(f"{key}: {value}" for key, value in STRATEGIES.items())

PAPER_PROMPT_METHODS = {"Zero-shot", "Few-shot", "Zero-shot CoT", "Few-shot CoT"}
PAPER_DESCRIBED_METHODS = {
    "Self-consistency", "Self-Refine", "Mixed-Initiative", "ESCoT", "CogChain", "Cooper",
    "MAS(Chain)", "MAS(Debate)", "MultiAgentESC (Ours)",
}
ALL_METHODS = [
    "Zero-shot", "Few-shot", "Zero-shot CoT", "Few-shot CoT", "Self-consistency",
    "Self-Refine", "Mixed-Initiative", "ESCoT", "CogChain", "Cooper", "MAS(Chain)",
    "MAS(Debate)", "MultiAgentESC (Ours)",
]


def _examples_plain(examples: list[dict[str, str]]) -> str:
    return "\n\n".join(f"User: {item['post']}\nAssistant: {item['response']}" for item in examples)


def _examples_cot(examples: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"Context: User: {item['post']}\nStrategy: [{item['strategy']}]\n"
        f"Reasoning: This strategy addresses the user's expressed need.\nResponse: {item['response']}"
        for item in examples
    )


def _experience_text(experiences: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"{item['post']}\n[{item['strategy']}] {item['response']}" for item in experiences
    )


class MethodRunner:
    def __init__(self, client: Any, config: dict[str, Any], examples: list[dict[str, str]], retriever: Any):
        self.client = client
        self.config = config
        self.examples = examples
        self.retriever = retriever
        generation = config["generation"]
        self.temperature = float(generation.get("temperature", 0.0))
        self.max_tokens = int(generation.get("max_tokens", 400))
        self.sc_samples = int(generation.get("self_consistency_samples", 5))

    def _call(self, prompt: str, tag: str, trace: list[dict[str, str]], temperature: float | None = None) -> str:
        result = self.client.complete(
            prompt,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            tag=tag,
        )
        trace.append({"tag": tag, "output": result.text})
        return result.text

    @staticmethod
    def _strategy(text: str) -> str:
        strategies = parse_strategies(text, STRATEGY_LIST, 1)
        return strategies[0] if strategies else "None"

    def zero_shot(self, context: str, trace: list[dict[str, str]]) -> tuple[str, str]:
        prompt = f"""### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'.
Your task is to play a role as 'Assistant' and generate a response based on the given dialogue context.
### Dialogue context
{context}
Your answer must be fewer than 30 words and must follow this format:
Response: [response]"""
        raw = self._call(prompt, "zero_shot", trace)
        return clean_response(raw), "None"

    def few_shot(self, context: str, trace: list[dict[str, str]]) -> tuple[str, str]:
        prompt = f"""You are a psychological counseling expert. Generate an Assistant response.
The following examples use <context then response>.
### Examples
{_examples_plain(self.examples)}
### Dialogue context
{context}
Your answer must be fewer than 30 words and follow:
Response: [response]"""
        raw = self._call(prompt, "few_shot", trace)
        return clean_response(raw), "None"

    def cot(self, context: str, trace: list[dict[str, str]], few_shot: bool = False, temperature: float | None = None) -> tuple[str, str, str]:
        examples = f"\n### Examples\n{_examples_cot(self.examples)}" if few_shot else ""
        step = "\nLet's think step by step!" if not few_shot else ""
        prompt = f"""### Instruction
You are a psychological counseling expert. Select a suitable emotional support strategy and generate a constrained response.
### Dialogue context
{context}
### Strategy definitions
{STRATEGY_DEFINITIONS}
Do not overly favor any strategy.{examples}
Return exactly:
Strategy: [strategy]
Reasoning: [reasoning]
Response: [response under 30 words]{step}"""
        raw = self._call(prompt, "few_shot_cot" if few_shot else "zero_shot_cot", trace, temperature)
        return clean_response(raw), self._strategy(raw), raw

    def self_consistency(self, context: str, trace: list[dict[str, str]]) -> tuple[str, str]:
        candidates = []
        for index in range(self.sc_samples):
            response, strategy, _ = self.cot(context, trace, few_shot=False, temperature=self.temperature)
            candidates.append((strategy, response))
        options = "\n".join(f"{i+1}. [{s}] {r}" for i, (s, r) in enumerate(candidates))
        prompt = f"""Select the most consistent and contextually appropriate answer among the candidate reasoning outcomes.
### Context
{context}
### Candidates
{options}
Return:
Strategy: [strategy]
Response: [response]"""
        raw = self._call(prompt, "self_consistency_select", trace)
        response, strategy = clean_response(raw), self._strategy(raw)
        if not response:
            strategy, response = candidates[0]
        return response, strategy

    def self_refine(self, context: str, trace: list[dict[str, str]]) -> tuple[str, str]:
        initial, strategy, _ = self.cot(context, trace, few_shot=False)
        feedback_prompt = f"""Critique this emotional-support response for empathy, problem identification, usefulness, and contextual fit.
Context: {context}
Strategy: {strategy}
Response: {initial}
Return: Feedback: [specific feedback]"""
        feedback = self._call(feedback_prompt, "self_refine_feedback", trace)
        refine_prompt = f"""Refine the response using the feedback. Keep it under 30 words.
Context: {context}
Initial: [{strategy}] {initial}
Feedback: {feedback}
Return:
Strategy: [strategy]
Response: [refined response]"""
        raw = self._call(refine_prompt, "self_refine_revision", trace)
        return clean_response(raw), self._strategy(raw) or strategy

    def mixed_initiative(self, context: str, trace: list[dict[str, str]]) -> tuple[str, str]:
        prompt = f"""Analyze the user's background, emotion type, problem type, current situation, and whether the next turn should explore or offer action.
Then generate a mixed-initiative emotional-support response: respond empathetically while gently advancing the conversation.
Context: {context}
Strategies:
{STRATEGY_DEFINITIONS}
Return:
Strategy: [strategy]
Reasoning: [brief analysis]
Response: [under 30 words]"""
        raw = self._call(prompt, "mixed_initiative", trace)
        return clean_response(raw), self._strategy(raw)

    def escot(self, context: str, trace: list[dict[str, str]]) -> tuple[str, str]:
        prompt = f"""Use an emotional-support chain of thought: identify emotion, infer the support need, choose one ESConv strategy, then answer.
Context: {context}
Strategies:
{STRATEGY_DEFINITIONS}
Return:
Emotion: [emotion]
Need: [support need]
Strategy: [strategy]
Reasoning: [brief reasoning]
Response: [under 30 words]"""
        raw = self._call(prompt, "escot", trace)
        return clean_response(raw), self._strategy(raw)

    def cogchain(self, context: str, trace: list[dict[str, str]]) -> tuple[str, str]:
        prompt = f"""Apply a cognitive support chain: infer situation, likely cognition, emotion, coping goal, and a suitable support strategy.
Context: {context}
Strategies:
{STRATEGY_DEFINITIONS}
Return:
Situation: [situation]
Cognition: [likely cognition]
Emotion: [emotion]
Goal: [coping goal]
Strategy: [strategy]
Response: [under 30 words]"""
        raw = self._call(prompt, "cogchain", trace)
        return clean_response(raw), self._strategy(raw)

    def cooper(self, context: str, trace: list[dict[str, str]]) -> tuple[str, str]:
        emotion = self._call(f"Analyze the emotion and empathy need.\nContext: {context}", "cooper_empathy", trace)
        problem = self._call(f"Analyze the concrete problem and information need.\nContext: {context}", "cooper_problem", trace)
        action = self._call(f"Analyze safe, practical next steps without being prescriptive.\nContext: {context}", "cooper_action", trace)
        prompt = f"""Coordinate three specialist reports into one emotional-support response.
Context: {context}
Empathy specialist: {emotion}
Problem specialist: {problem}
Action specialist: {action}
Strategies:
{STRATEGY_DEFINITIONS}
Return:
Strategy: [strategy]
Response: [under 30 words]"""
        raw = self._call(prompt, "cooper_coordinator", trace)
        return clean_response(raw), self._strategy(raw)

    def mas_chain(self, context: str, trace: list[dict[str, str]]) -> tuple[str, str]:
        first = self._call(f"Analyze the user's emotional state, problem, and need.\nContext: {context}", "mas_chain_agent1", trace)
        second = self._call(f"Review and improve this analysis, then recommend a support strategy.\nContext: {context}\nAnalysis: {first}\nStrategies: {STRATEGY_DEFINITIONS}", "mas_chain_agent2", trace)
        raw = self._call(f"Generate the final supportive answer based on prior agents.\nContext: {context}\nAgent 1: {first}\nAgent 2: {second}\nReturn:\nStrategy: [strategy]\nResponse: [under 30 words]", "mas_chain_agent3", trace)
        return clean_response(raw), self._strategy(raw)

    def mas_debate(self, context: str, trace: list[dict[str, str]]) -> tuple[str, str]:
        candidates = []
        for index, focus in enumerate(("empathy", "problem identification", "useful suggestions"), start=1):
            raw = self._call(f"Generate a response prioritizing {focus}.\nContext: {context}\nReturn:\nStrategy: [strategy]\nResponse: [under 30 words]", f"mas_debate_agent{index}", trace)
            candidates.append(f"Candidate {index}: [{self._strategy(raw)}] {clean_response(raw)}")
        judge = self._call(f"Debate the merits of these candidates and select the best.\nContext: {context}\n" + "\n".join(candidates) + "\nReturn:\nStrategy: [strategy]\nResponse: [response]", "mas_debate_judge", trace)
        return clean_response(judge), self._strategy(judge)

    def _analysis(self, context: str, trace: list[dict[str, str]]) -> tuple[str, str, str]:
        emotion = self._call(f"""Infer the emotional state in the user's last utterance.
Context: {context}
Return:
Emotion: [emotion]
Reasoning: [reasoning]""", "dialogue_emotion", trace)
        cause = self._call(f"""Infer the specific event that caused the emotional state.
Context: {context}
Emotional state: {emotion}
Return:
Event: [event]
Reasoning: [reasoning]""", "dialogue_event", trace)
        intention = self._call(f"""Infer the user's intention for addressing the event.
Context: {context}
Emotional state: {emotion}
Event: {cause}
Return:
Intention: [intention]
Reasoning: [reasoning]""", "dialogue_intention", trace)
        return emotion, cause, intention

    def _is_complex(self, context: str, trace: list[dict[str, str]]) -> bool:
        raw = self._call(f"""Analyze whether this conversation reflects all three: current emotional state, reason for seeking support, and coping intention.
If all are present reply YES, otherwise NO.
Context: {context}
Return two parts: 1. YES or NO 2. explanation.""", "behavior_control", trace)
        return "yes" in raw.lower()

    def multiagentesc(self, record: dict[str, Any], trace: list[dict[str, str]], variant: str = "full") -> tuple[str, str]:
        context = record["context"]
        # Public code routes early or insufficiently complex targets to zero-shot.
        if int(record.get("turn_index", 99)) <= 5 or not self._is_complex(context, trace):
            return self.zero_shot(context, trace)

        use_analysis = variant != "w/o dialogue analysis"
        use_experience = variant != "w/o experience"
        use_group = variant != "w/o group discussion"
        if use_analysis:
            emotion, cause, intention = self._analysis(context, trace)
        else:
            emotion = cause = intention = "[module removed in this ablation]"

        retrieved = self.retriever.search(record["post"], int(self.config["retrieval"].get("top_k", 10))) if use_experience else []
        examples = _experience_text(retrieved) if retrieved else "[no retrieved experience]"
        state = f"Emotion:\n{emotion}\nEvent:\n{cause}\nIntention:\n{intention}"

        if not use_group:
            prompt = f"""A single expert must select strategies and generate one response without group discussion.
Context: {context}
{state}
Retrieved examples:
{examples}
Strategies:
{STRATEGY_DEFINITIONS}
Return:
Strategy: [strategy]
Reasoning: [reasoning]
Response: [under 30 words]"""
            raw = self._call(prompt, "multiagentesc_single_agent", trace)
            return clean_response(raw), self._strategy(raw)

        selected: list[str] = []
        for index in range(3):
            avoid = ", ".join(selected) if selected else "None"
            prompt = f"""Select one appropriate strategy. Differ from prior agents where reasonable.
Context: {context}
{state}
Retrieved examples:
{examples}
Already selected: {avoid}
Return:
Strategy: [strategy]
Reasoning: [reasoning]"""
            raw = self._call(prompt, f"strategy_agent_{index+1}", trace)
            strategy = self._strategy(raw)
            if strategy != "None" and strategy not in selected:
                selected.append(strategy)
        if not selected:
            return self.zero_shot(context, trace)

        candidates = []
        for index, strategy in enumerate(selected, start=1):
            matching = [item for item in retrieved if item["strategy"].lower() == strategy.lower()]
            prompt = f"""Generate an Assistant response using {strategy}.
Context: {context}
{state}
Examples:
{_experience_text(matching) if matching else examples}
Return:
Response: [{strategy}] [response under 30 words]"""
            raw = self._call(prompt, f"strategy_response_{index}", trace)
            candidates.append((strategy, clean_response(raw)))

        options = "\n".join(f"{i+1}. [{s}] {r}" for i, (s, r) in enumerate(candidates))
        opinions = []
        for index, (favored, _) in enumerate(candidates, start=1):
            raw = self._call(f"""Compare the candidate responses. You initially favor {favored}, but revise if another is better.
Context: {context}
{state}
Candidates:
{options}
Return:
Strategy: [chosen strategy]
Response: [chosen response]
Reasoning: [reasoning]""", f"response_debate_{index}", trace)
            opinions.append(raw)

        raw = self._call(f"""Reflect on the agents' opinions and choose the best response.
Context: {context}
Candidates:
{options}
Opinions:
{chr(10).join(opinions)}
Return:
Strategy: [strategy]
Response: [response]
Reasoning: [reasoning]""", "response_reflection_vote", trace)
        strategy = self._strategy(raw)
        response = clean_response(raw)
        if not response or strategy == "None":
            strategy, response = candidates[0]

        refined = self._call(f"""Check contextual consistency, strategy alignment, and emotional-support effectiveness. Preserve if adequate; otherwise refine under 30 words.
Context: {context}
Response: [{strategy}] {response}
Return:
Strategy: [{strategy}]
Response: [final response]
Reasoning: [reasoning]""", "response_self_reflection", trace)
        return clean_response(refined) or response, self._strategy(refined) or strategy

    def run(self, method: str, record: dict[str, Any], variant: str = "full") -> dict[str, Any]:
        trace: list[dict[str, str]] = []
        context = record["context"]
        if method == "Zero-shot":
            response, strategy = self.zero_shot(context, trace)
        elif method == "Few-shot":
            response, strategy = self.few_shot(context, trace)
        elif method == "Zero-shot CoT":
            response, strategy, _ = self.cot(context, trace, False)
        elif method == "Few-shot CoT":
            response, strategy, _ = self.cot(context, trace, True)
        elif method == "Self-consistency":
            response, strategy = self.self_consistency(context, trace)
        elif method == "Self-Refine":
            response, strategy = self.self_refine(context, trace)
        elif method == "Mixed-Initiative":
            response, strategy = self.mixed_initiative(context, trace)
        elif method == "ESCoT":
            response, strategy = self.escot(context, trace)
        elif method == "CogChain":
            response, strategy = self.cogchain(context, trace)
        elif method == "Cooper":
            response, strategy = self.cooper(context, trace)
        elif method == "MAS(Chain)":
            response, strategy = self.mas_chain(context, trace)
        elif method == "MAS(Debate)":
            response, strategy = self.mas_debate(context, trace)
        elif method == "MultiAgentESC (Ours)":
            response, strategy = self.multiagentesc(record, trace, variant)
        else:
            raise ValueError(f"unknown method: {method}")
        implementation = "paper-prompt" if method in PAPER_PROMPT_METHODS else "paper-described-reimplementation"
        if method == "MultiAgentESC (Ours)":
            implementation = "public-code-aligned-reimplementation"
        return {
            "prediction": response,
            "pred_strategy": strategy,
            "trace": trace,
            "implementation": implementation,
            "variant": variant,
        }
