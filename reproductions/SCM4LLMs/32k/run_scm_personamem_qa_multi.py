#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCM PersonaMem QA runner for isolated 128K / 1M adapters.

Run preflight first (no GPU model load):
  python -u run_scm_personamem_qa_multi.py \
    --dataset-dir adapters/scm_personamem_128k \
    --dataset-name PersonaMem-128K \
    --data-file personamem_scm_128k.jsonl \
    --expected-questions 2727 \
    --preflight-only

Then run:
  python -u run_scm_personamem_qa_multi.py \
    --dataset-dir adapters/scm_personamem_128k \
    --dataset-name PersonaMem-128K \
    --data-file personamem_scm_128k.jsonl \
    --expected-questions 2727

Use a separate directory for PersonaMem-1M.
"""

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch
import tiktoken
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True,
                    help="独立实验目录；128K和1M必须使用不同目录")
    ap.add_argument("--dataset-name", required=True,
                    help="例如 PersonaMem-128K / PersonaMem-1M")
    ap.add_argument("--data-file", default="personamem_scm.jsonl")
    ap.add_argument("--unique-file", default="unique_scm_memory_units.jsonl")
    ap.add_argument("--summaries-file", default="scm_summaries_qwen7b.jsonl")
    ap.add_argument("--topk-file", default="scm_top6_retrieval.jsonl")
    ap.add_argument("--controller-file", default="scm_controller_qwen7b.jsonl")
    ap.add_argument("--answers-file", default="personamem_scm_answers_qwen7b.jsonl")
    ap.add_argument("--metrics-file", default="personamem_scm_metrics_qwen7b.json")
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--expected-questions", type=int, default=None)
    ap.add_argument("--max-pre-turn-tokens", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-context-tokens", type=int, default=32768)
    ap.add_argument("--gpu-memory", default="18GiB")
    ap.add_argument("--cpu-memory", default="24GiB")
    ap.add_argument("--preflight-only", action="store_true")
    return ap.parse_args()


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"JSONL损坏: {path}:{line_no}: {exc}"
                ) from exc
    return rows


def load_unique_by_key(path, key, label):
    rows = load_jsonl(path)
    result = {}
    for i, row in enumerate(rows, 1):
        if key not in row:
            raise KeyError(f"{label}第{i}条缺少字段 {key}")
        value = str(row[key])
        if value in result:
            raise RuntimeError(f"{label}存在重复键: {value}")
        result[value] = row
    return result


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"缺少输入文件: {path}")


def parse_options(value):
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        raise TypeError(f"all_options必须是list或str，实际为{type(value)}")
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    raise ValueError(f"无法解析all_options: {value[:200]}")


def build_mcq_input(q):
    options = parse_options(q["all_options"])
    option_text = "\n".join(str(x) for x in options)
    return (
        f"{q['question']}\n\n"
        "Choose the single best answer from the options below.\n\n"
        f"{option_text}\n\n"
        "Your final answer must contain exactly one option label from (a), (b), (c), or (d).\n"
        "End your response using exactly this format:\n"
        "<final_answer>(a)</final_answer>\n"
        "Replace (a) with the correct option label."
    )


EN_HISTORY_AGENT_PROMPT = """The following is a conversation between a user and an AI assistant. Please answer the current question based on the history of the conversation:

Related conversation history:

{history_turn_text}

Previous conversation:

{pre_turn_text}

###

User: {input}

Assistant: """


EN_NO_HISTORY_AGENT_PROMPT = """The following is a conversation between a user and an AI assistant. Please answer the current question based on the history of the conversation:

Previous conversation:

{pre_turn_text}

###

User: {input}

Assistant: """


OPTION_RE = re.compile(r"\(([abcdABCD])\)")


def normalize_option(value):
    if not value:
        return None
    value = str(value).strip().lower()
    if value in {"a", "b", "c", "d"}:
        return f"({value})"
    match = OPTION_RE.fullmatch(value)
    return f"({match.group(1).lower()})" if match else None


def parse_answer(text):
    if text is None:
        return None, [], "empty"

    lower = text.lower()
    marker = "<final_answer>"
    region = text
    source = "full_response"
    pos = lower.rfind(marker)

    if pos >= 0:
        region = text[pos + len(marker):]
        end = region.lower().find("</final_answer>")
        if end >= 0:
            region = region[:end]
        source = "final_answer_region"

    found = [
        normalize_option(m.group(1))
        for m in OPTION_RE.finditer(region)
    ]
    found = [x for x in found if x is not None]
    unique_found = list(dict.fromkeys(found))

    if len(unique_found) == 1:
        return unique_found[0], unique_found, source

    if source != "full_response":
        found_all = [
            normalize_option(m.group(1))
            for m in OPTION_RE.finditer(text)
        ]
        found_all = [x for x in found_all if x is not None]
        unique_all = list(dict.fromkeys(found_all))
        if len(unique_all) == 1:
            return unique_all[0], unique_all, "whole_response_fallback"
        return None, unique_all, "ambiguous"

    return None, unique_found, "ambiguous"


def atomic_json_dump(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    args = parse_args()
    base = Path(args.dataset_dir).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"实验目录不存在: {base}")

    # Every input/output is rooted in the dataset-specific directory.
    paths = {
        "data": base / args.data_file,
        "unique": base / args.unique_file,
        "summaries": base / args.summaries_file,
        "topk": base / args.topk_file,
        "controller": base / args.controller_file,
        "answers": base / args.answers_file,
        "metrics": base / args.metrics_file,
    }
    for name in ("data", "unique", "summaries", "topk"):
        require_file(paths[name])

    print("===== SCM QA PREFLIGHT =====", flush=True)
    print("dataset:", args.dataset_name, flush=True)
    print("dataset_dir:", base, flush=True)
    print("model:", args.model, flush=True)

    questions_rows = load_jsonl(paths["data"])
    questions = {}
    for i, q in enumerate(questions_rows, 1):
        if "question_id" not in q:
            raise KeyError(f"问题数据第{i}条缺少question_id")
        qid = str(q["question_id"])
        if qid in questions:
            raise RuntimeError(f"问题数据存在重复question_id: {qid}")
        required = {
            "question_type", "question", "correct_answer", "all_options"
        }
        missing = required - set(q)
        if missing:
            raise KeyError(f"问题 {qid} 缺少字段: {sorted(missing)}")
        questions[qid] = q

    question_ids = set(questions)
    if args.expected_questions is not None:
        if len(questions) != args.expected_questions:
            raise RuntimeError(
                f"问题数量错误: {len(questions)} != {args.expected_questions}"
            )

    unique = load_unique_by_key(
        paths["unique"], "unit_hash", "unique memory units"
    )
    summaries = load_unique_by_key(
        paths["summaries"], "unit_hash", "summaries"
    )
    retrieval = load_unique_by_key(
        paths["topk"], "question_id", "retrieval"
    )
    # Controller cache is required only if at least one question needs it.
    # Its target set is known after loading the retrieval rows below.
    controller = {}
    if paths["controller"].exists():
        controller = load_unique_by_key(
            paths["controller"], "question_id", "controller"
        )

    if set(retrieval) != question_ids:
        raise RuntimeError(
            "检索问题ID集合不匹配: "
            f"missing={len(question_ids - set(retrieval))}, "
            f"extra={len(set(retrieval) - question_ids)}"
        )

    if not unique:
        raise RuntimeError("unique memory units为空")
    if set(summaries) != set(unique):
        raise RuntimeError(
            "摘要缓存与唯一记忆单元不匹配: "
            f"missing={len(set(unique) - set(summaries))}, "
            f"extra={len(set(summaries) - set(unique))}"
        )

    for h, row in unique.items():
        if not isinstance(row.get("raw_text"), str):
            raise TypeError(f"记忆单元raw_text不是字符串: {h}")
        if not row["raw_text"].strip():
            raise RuntimeError(f"记忆单元raw_text为空: {h}")
        if not isinstance(summaries[h].get("summary"), str):
            raise TypeError(f"summary不是字符串: {h}")
        if not summaries[h]["summary"].strip():
            raise RuntimeError(f"summary为空: {h}")

    controller_targets = {
        qid for qid, row in retrieval.items()
        if bool(row.get("needs_controller"))
    }
    if set(controller) != controller_targets:
        raise RuntimeError(
            "Controller缓存目标集合不匹配: "
            f"missing={len(controller_targets - set(controller))}, "
            f"extra={len(set(controller) - controller_targets)}"
        )

    # Validate every retrieval reference before loading the LLM.
    for qid, row in retrieval.items():
        previous_hash = str(row.get("previous_hash", ""))
        if previous_hash not in unique:
            raise RuntimeError(
                f"问题{qid}的previous_hash不存在: {previous_hash}"
            )

        if row.get("needs_controller"):
            ctrl_text = controller[qid].get("retrieve_history_text")
            # Empty text is a valid controller decision if it drops all candidates.
            if not isinstance(ctrl_text, str):
                raise RuntimeError(f"问题{qid}的Controller历史不是字符串")
        else:
            hashes = row.get("chronological_top_hashes")
            if not isinstance(hashes, list) or len(hashes) != 6:
                raise RuntimeError(
                    f"问题{qid}的Top-6列表异常: "
                    f"{None if hashes is None else len(hashes)}"
                )
            missing_hashes = [str(h) for h in hashes if str(h) not in unique]
            if missing_hashes:
                raise RuntimeError(
                    f"问题{qid}的Top-6包含未知hash: {missing_hashes[:3]}"
                )

    # Parse every question and validate the gold answer before model load.
    for qid, q in questions.items():
        options = parse_options(q["all_options"])
        if not options:
            raise RuntimeError(f"问题{qid}没有选项")
        gold = normalize_option(q["correct_answer"])
        if gold is None:
            raise RuntimeError(
                f"问题{qid}的correct_answer无法标准化: {q['correct_answer']!r}"
            )

    print("questions:", len(questions), flush=True)
    print("unique memory units:", len(unique), flush=True)
    print("summaries:", len(summaries), flush=True)
    print("retrieval rows:", len(retrieval), flush=True)
    print("controller targets:", len(controller_targets), flush=True)
    print("controller records:", len(controller), flush=True)

    # Existing answer cache is resumable, but duplicate or foreign IDs are fatal.
    done = {}
    if paths["answers"].exists():
        for i, row in enumerate(load_jsonl(paths["answers"]), 1):
            qid = str(row.get("question_id", ""))
            if qid not in question_ids:
                raise RuntimeError(
                    f"答案缓存含有当前数据集之外的question_id: {qid}"
                )
            if qid in done:
                raise RuntimeError(f"答案缓存存在重复question_id: {qid}")
            done[qid] = row

    remaining = [q for qid, q in questions.items() if qid not in done]
    print("answers already done:", len(done), flush=True)
    print("remaining:", len(remaining), flush=True)

    # Load tokenizer/config only; this is CPU-side preflight, not GPU inference.
    if not Path(args.model).exists():
        raise FileNotFoundError(f"本地Qwen模型不存在: {args.model}")
    config = AutoConfig.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )

    context_limit = args.max_context_tokens
    config_limit = getattr(config, "max_position_embeddings", None)
    if isinstance(config_limit, int) and config_limit > 0:
        context_limit = min(context_limit, config_limit)

    def get_previous_text(previous_hash):
        raw = unique[previous_hash]["raw_text"]
        if len(enc.encode(raw)) < args.max_pre_turn_tokens:
            return raw, "raw"
        return summaries[previous_hash]["summary"], "summary"

    def get_retrieved_history(qid):
        row = retrieval[qid]
        if row.get("needs_controller"):
            return controller[qid]["retrieve_history_text"], "controller"
        pieces = [
            unique[str(h)]["raw_text"]
            for h in row["chronological_top_hashes"]
        ]
        text = "\n\n".join(pieces)
        if text:
            text += "\n\n"
        return text, "raw_top6"

    try:
        enc = tiktoken.encoding_for_model("text-davinci-003")
    except Exception:
        enc = tiktoken.get_encoding("p50k_base")

    def build_final_prompt(q):
        qid = str(q["question_id"])
        row = retrieval[qid]
        previous_text, previous_mode = get_previous_text(
            str(row["previous_hash"])
        )
        retrieved_text, retrieval_mode = get_retrieved_history(qid)
        mcq_input = build_mcq_input(q)

        if retrieved_text:
            prompt = EN_HISTORY_AGENT_PROMPT.format(
                history_turn_text=retrieved_text,
                pre_turn_text=previous_text,
                input=mcq_input,
            )
        else:
            prompt = EN_NO_HISTORY_AGENT_PROMPT.format(
                pre_turn_text=previous_text,
                input=mcq_input,
            )

        return prompt, {
            "retrieval_mode": retrieval_mode,
            "previous_mode": previous_mode,
            "previous_hash": str(row["previous_hash"]),
        }

    prompt_lengths = []
    for qid, q in questions.items():
        prompt, _ = build_final_prompt(q)
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        token_count = len(
            tokenizer(chat_text, add_special_tokens=False)["input_ids"]
        )
        prompt_lengths.append(token_count)

    sorted_lengths = sorted(prompt_lengths)

    def percentile(p):
        return sorted_lengths[int(round((len(sorted_lengths) - 1) * p))]

    print("prompt_tokens:", {
        "min": sorted_lengths[0],
        "median": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": sorted_lengths[-1],
    }, flush=True)
    print("model_context_limit:", context_limit, flush=True)

    overflow = [
        n for n in prompt_lengths
        if n + args.max_new_tokens > context_limit
    ]
    if overflow:
        raise RuntimeError(
            f"Prompt超出模型上下文预算: {len(overflow)}条; "
            f"max_prompt={max(prompt_lengths)}, "
            f"max_new_tokens={args.max_new_tokens}, "
            f"context_limit={context_limit}"
        )

    if args.preflight_only:
        print("SCM_QA_PREFLIGHT_PASS", flush=True)
        print("GPU model load: SKIPPED", flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，停止QA推理")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: args.gpu_memory, "cpu": args.cpu_memory},
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    print("HF_DEVICE_MAP:", getattr(model, "hf_device_map", None), flush=True)

    def lm_head_pre_hook(module, hook_args):
        hidden = hook_args[0]
        if hidden.ndim == 3 and hidden.shape[1] > 1:
            hidden = hidden[:, -1:, :]
        return (hidden,)

    hook = model.lm_head.register_forward_pre_hook(lm_head_pre_hook)

    def generate(prompt):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt")
        input_device = model.get_input_embeddings().weight.device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = output[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(
            new_ids, skip_special_tokens=True
        ).strip()

    try:
        with paths["answers"].open("a", encoding="utf-8") as fout:
            for i, q in enumerate(remaining, 1):
                qid = str(q["question_id"])
                prompt, memory_meta = build_final_prompt(q)
                response = generate(prompt)
                parsed, options_found, parser_source = parse_answer(response)
                gold = normalize_option(q["correct_answer"])
                correct = parsed == gold

                record = {
                    "question_id": qid,
                    "question_type": q["question_type"],
                    "question": q["question"],
                    "correct_answer": gold,
                    "parsed_answer": parsed,
                    "correct": bool(correct),
                    "options_found": options_found,
                    "parser_source": parser_source,
                    "response": response,
                    "memory_meta": memory_meta,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
                done[qid] = record

                print(
                    f"[ANSWER {len(done)}/{len(questions)}] "
                    f"qid={qid} gold={gold} pred={parsed} "
                    f"correct={correct} found={options_found}",
                    flush=True,
                )
    finally:
        hook.remove()

    if set(done) != question_ids:
        missing = question_ids - set(done)
        extra = set(done) - question_ids
        raise RuntimeError(
            f"QA结果ID集合不完整: missing={len(missing)}, extra={len(extra)}"
        )

    records = [done[qid] for qid in questions]
    correct_total = sum(bool(x["correct"]) for x in records)
    unparsed = sum(x["parsed_answer"] is None for x in records)
    type_total = Counter()
    type_correct = Counter()
    type_unparsed = Counter()

    for row in records:
        qtype = row["question_type"]
        type_total[qtype] += 1
        if row["correct"]:
            type_correct[qtype] += 1
        if row["parsed_answer"] is None:
            type_unparsed[qtype] += 1

    overall_acc = correct_total / len(records)
    per_type = {}
    for qtype in sorted(type_total):
        total = type_total[qtype]
        per_type[qtype] = {
            "correct": type_correct[qtype],
            "total": total,
            "accuracy": type_correct[qtype] / total,
            "unparsed": type_unparsed[qtype],
        }

    metrics = {
        "method": "SCM",
        "dataset": args.dataset_name,
        "setting": "compatibility",
        "text_llm": "Qwen2.5-7B-Instruct",
        "embedding": "Contriever",
        "similar_top_k": 6,
        "max_history_tokens": 2500,
        "max_previous_turn_tokens": args.max_pre_turn_tokens,
        "max_new_tokens": args.max_new_tokens,
        "model_context_limit": context_limit,
        "questions": len(records),
        "correct": correct_total,
        "overall_accuracy": overall_acc,
        "unparsed": unparsed,
        "controller_triggered": len(controller_targets),
        "per_question_type": per_type,
        "prompt_token_stats": {
            "min": sorted_lengths[0],
            "median": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": sorted_lengths[-1],
        },
        "input_files": {
            k: str(v) for k, v in paths.items()
            if k not in {"answers", "metrics"}
        },
    }
    atomic_json_dump(metrics, paths["metrics"])

    print("===== FINAL METRICS =====", flush=True)
    print(
        f"overall: {correct_total}/{len(records)} "
        f"= {overall_acc * 100:.2f}%",
        flush=True,
    )
    print("unparsed:", unparsed, flush=True)
    print("metrics:", paths["metrics"], flush=True)
    print("PERSONAMEM_SCM_QA_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SCM_QA_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
