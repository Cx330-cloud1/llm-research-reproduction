#!/usr/bin/env python3
"""
SCM PersonaMem controller for deduplicated 128K / 1M adapters.

Important: retrieval candidates are history *positions*, not unique hashes.
The same deduplicated memory unit may occur at multiple positions in one
question's history. All controller decisions and chronological reconstruction
therefore use candidate indices. Hashes are used only to look up the content.
"""

import argparse
import json
from pathlib import Path

import torch
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_no}: {exc}"
                ) from exc
    return rows


def require_unique_dict(rows, key, label):
    result = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"Duplicate {key} in {label}: {value}")
        result[value] = row
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run SCM's retrieval controller for one PersonaMem dataset."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--expected-questions", type=int, required=True)
    parser.add_argument("--expected-targets", type=int, required=True)
    parser.add_argument(
        "--model",
        default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    base = Path(args.dataset_dir)
    questions_rows = read_jsonl(base / "question_metadata.jsonl")
    unit_rows = read_jsonl(base / "unique_scm_memory_units.jsonl")
    summary_rows = read_jsonl(base / "scm_summaries_qwen7b.jsonl")
    mapping_rows = read_jsonl(base / "question_scm_unit_hashes.jsonl")
    retrieval_rows = read_jsonl(base / "scm_top6_retrieval.jsonl")
    output_path = base / "scm_controller_qwen7b.jsonl"

    try:
        encoding = tiktoken.encoding_for_model("text-davinci-003")
    except Exception:
        encoding = tiktoken.get_encoding("p50k_base")

    def token_count(text):
        return len(encoding.encode(text))

    questions = require_unique_dict(
        questions_rows, "question_id", "question metadata"
    )
    units = require_unique_dict(unit_rows, "unit_hash", "unique units")
    summaries_by_hash = require_unique_dict(
        summary_rows, "unit_hash", "summary cache"
    )
    mappings = require_unique_dict(
        mapping_rows, "question_id", "question-to-unit mapping"
    )
    retrieval = require_unique_dict(
        retrieval_rows, "question_id", "retrieval cache"
    )

    summaries = {
        h: row["summary"] for h, row in summaries_by_hash.items()
    }

    if len(questions) != args.expected_questions:
        raise AssertionError(
            f"Question count mismatch: {len(questions)} != "
            f"{args.expected_questions}"
        )
    if set(units) != set(summaries):
        raise AssertionError(
            "Summary cache hash coverage differs from unique-unit hashes"
        )
    if set(questions) != set(mappings) or set(questions) != set(retrieval):
        raise AssertionError(
            "Question IDs differ across metadata, mapping, and retrieval files"
        )

    targets = {
        qid: row
        for qid, row in retrieval.items()
        if bool(row.get("needs_controller"))
    }
    if len(targets) != args.expected_targets:
        raise AssertionError(
            f"Controller target mismatch: {len(targets)} != "
            f"{args.expected_targets}"
        )

    repeated_hash_candidate_rows = 0

    # Validate that retrieval indices point to the expected hashes in the
    # question's original ordered history mapping. Indices must be unique;
    # hashes do NOT need to be unique.
    for qid, row in targets.items():
        candidate_indices = row["controller_top_indices"]
        candidate_hashes = row["controller_top_hashes"]
        chronological_indices = row["chronological_top_indices"]
        chronological_hashes = row["chronological_top_hashes"]
        previous_hash = row["previous_hash"]
        question_hashes = mappings[qid]["unit_hashes"]

        if not (
            len(candidate_indices)
            == len(candidate_hashes)
            == len(chronological_indices)
            == len(chronological_hashes)
            == 6
        ):
            raise AssertionError(f"{qid}: retrieval top-6 arrays must all have length 6")

        if len(set(candidate_indices)) != 6:
            raise AssertionError(
                f"{qid}: duplicate candidate indices; expected six distinct "
                "history positions"
            )

        if sorted(candidate_indices) != sorted(chronological_indices):
            raise AssertionError(
                f"{qid}: chronological indices do not match candidate indices"
            )

        if len(set(candidate_hashes)) < 6:
            repeated_hash_candidate_rows += 1

        for index, h in zip(candidate_indices, candidate_hashes):
            if not isinstance(index, int) or index < 0 or index >= len(question_hashes):
                raise AssertionError(f"{qid}: invalid candidate history index {index}")
            if question_hashes[index] != h:
                raise AssertionError(
                    f"{qid}: candidate index/hash mismatch at index {index}"
                )
            if h not in units or h not in summaries:
                raise AssertionError(f"{qid}: missing unit or summary for hash {h}")

        for index, h in zip(chronological_indices, chronological_hashes):
            if question_hashes[index] != h:
                raise AssertionError(
                    f"{qid}: chronological index/hash mismatch at index {index}"
                )

        if previous_hash not in units or previous_hash not in summaries:
            raise AssertionError(f"{qid}: missing previous-turn unit/summary")

        if "all_summary_total_budget" in row:
            if row["all_summary_total_budget"] > 2500:
                raise AssertionError(
                    f"{qid}: all-summary budget exceeds 2500 tokens"
                )

    # Resume support. Refuse legacy hash-keyed cache records because they
    # cannot safely represent repeated occurrences of the same hash.
    done = {}
    if output_path.exists():
        for row in read_jsonl(output_path):
            qid = str(row["question_id"])
            if qid not in targets:
                raise AssertionError(
                    f"Foreign question_id in controller cache: {qid}"
                )
            if qid in done:
                raise AssertionError(
                    f"Duplicate question_id in controller cache: {qid}"
                )
            if "controller_top_indices" not in row:
                raise RuntimeError(
                    f"Legacy controller cache detected for {qid}. "
                    f"Back up and remove {output_path}, then rerun."
                )
            expected_indices = targets[qid]["controller_top_indices"]
            if row["controller_top_indices"] != expected_indices:
                raise AssertionError(
                    f"{qid}: cached controller indices differ from retrieval"
                )
            if len(row.get("flags_controller_order", [])) != 6:
                raise AssertionError(f"{qid}: invalid cached controller flags")
            done[qid] = row

    remaining = [
        (qid, row)
        for qid, row in targets.items()
        if qid not in done
    ]

    print("===== CONTROLLER PREFLIGHT =====", flush=True)
    print("dataset_dir:", base.resolve(), flush=True)
    print("questions:", len(questions), flush=True)
    print("unique units:", len(units), flush=True)
    print("summaries:", len(summaries), flush=True)
    print("retrieval rows:", len(retrieval), flush=True)
    print("controller targets:", len(targets), flush=True)
    print("targets with repeated hashes:", repeated_hash_candidate_rows, flush=True)
    print("already done:", len(done), flush=True)
    print("remaining:", len(remaining), flush=True)

    if args.preflight_only:
        print("SCM_CONTROLLER_PREFLIGHT_PASS", flush=True)
        return

    if not remaining:
        if set(done) != set(targets):
            raise AssertionError("Controller cache coverage mismatch")
        print("SCM_CONTROLLER_CACHE_OK", flush=True)
        return

    judge_prompt = """给定[对话内容]和[用户问题]，请回答指令问题。

[对话内容]:
```
{content}
```

[用户问题]:
```
{query}
```

指令问题:
```
根据[对话内容]是否能够回答[用户问题]？如果能请回答 `(A)是`，否则请回答 `(B)否`。
```

现在请开始回答，输出必须严格按照以下格式输出：

[答案]: 最终的答案是：(A)是 / (B)否
"""

    print("===== LOAD QWEN =====", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "18GiB", "cpu": "24GiB"},
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    print("HF_DEVICE_MAP:", getattr(model, "hf_device_map", None), flush=True)

    # Keep the same memory-saving lm_head hook used in the existing
    # Qwen reproduction scripts.
    def lm_head_pre_hook(module, hook_args):
        hidden = hook_args[0]
        if hidden.ndim == 3 and hidden.shape[1] > 1:
            hidden = hidden[:, -1:, :]
        return (hidden,)

    hook = model.lm_head.register_forward_pre_hook(lm_head_pre_hook)

    def generate(prompt):
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(chat_text, return_tensors="pt")
        input_device = next(model.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_ids = output[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(
            new_ids, skip_special_tokens=True
        ).strip()

    # Identical text/query pairs can be judged once, but their decisions
    # remain stored independently by candidate position.
    judge_cache = {}

    def can_answer(content, query):
        key = (content, query)
        if key not in judge_cache:
            response = generate(
                judge_prompt.format(content=content, query=query)
            )
            # Preserve the decision convention of the original SCM adapter.
            judge_cache[key] = ("(B)" not in response, response)
        return judge_cache[key]

    def previous_turn_tokens(previous_hash):
        raw_text = units[previous_hash]["raw_text"]
        raw_tokens = token_count(raw_text)
        if raw_tokens < 500:
            return raw_tokens
        return token_count(summaries[previous_hash])

    def calculate_budget(candidate_hashes, flags, previous_hash):
        total = previous_turn_tokens(previous_hash)
        for h, flag in zip(candidate_hashes, flags):
            if flag == "drop":
                continue
            if flag == "summary":
                total += token_count(summaries[h])
            else:
                total += token_count(units[h]["raw_text"])
        return total > 2500, total

    try:
        with output_path.open("a", encoding="utf-8") as fout:
            for n, (qid, row) in enumerate(remaining, 1):
                query = questions[qid]["question"]

                # Arrays are in controller (similarity-rank) order.
                candidate_indices = row["controller_top_indices"]
                candidate_hashes = row["controller_top_hashes"]
                previous_hash = row["previous_hash"]

                # One flag per selected history POSITION, not per hash.
                flags = ["raw"] * len(candidate_indices)
                decisions = []

                too_long, total = calculate_budget(
                    candidate_hashes, flags, previous_hash
                )
                if not too_long:
                    raise AssertionError(
                        f"{qid}: controller triggered although raw budget fits"
                    )

                first_round = True
                rounds = 0

                while too_long:
                    rounds += 1

                    for position, (history_index, h) in enumerate(
                        zip(candidate_indices, candidate_hashes)
                    ):
                        if first_round:
                            if position < len(candidate_indices) - 1:
                                raw_ok, raw_output = can_answer(
                                    units[h]["raw_text"], query
                                )

                                if not raw_ok:
                                    choice = "drop"
                                    decisions.append({
                                        "round": rounds,
                                        "candidate_position": position,
                                        "history_index": history_index,
                                        "unit_hash": h,
                                        "stage": "raw",
                                        "model_output": raw_output,
                                        "choice": choice,
                                    })
                                else:
                                    summary_ok, summary_output = can_answer(
                                        summaries[h], query
                                    )
                                    choice = "summary" if summary_ok else "raw"
                                    decisions.append({
                                        "round": rounds,
                                        "candidate_position": position,
                                        "history_index": history_index,
                                        "unit_hash": h,
                                        "stage": "raw_then_summary",
                                        "raw_model_output": raw_output,
                                        "summary_model_output": summary_output,
                                        "choice": choice,
                                    })
                            else:
                                # Keep the lowest-ranked candidate in raw form,
                                # matching the existing SCM controller behavior.
                                choice = "raw"
                                decisions.append({
                                    "round": rounds,
                                    "candidate_position": position,
                                    "history_index": history_index,
                                    "unit_hash": h,
                                    "stage": "forced_last_candidate",
                                    "choice": choice,
                                })
                        else:
                            choice = (
                                "summary" if flags[position] == "raw"
                                else flags[position]
                            )
                            decisions.append({
                                "round": rounds,
                                "candidate_position": position,
                                "history_index": history_index,
                                "unit_hash": h,
                                "stage": "subsequent_round",
                                "choice": choice,
                            })

                        flags[position] = choice

                        if not first_round:
                            too_long, total = calculate_budget(
                                candidate_hashes, flags, previous_hash
                            )
                            if not too_long:
                                break

                    too_long, total = calculate_budget(
                        candidate_hashes, flags, previous_hash
                    )
                    first_round = False

                    if rounds > 10:
                        raise RuntimeError(
                            f"Controller did not converge: {qid}"
                        )

                # Reconstruct in chronological history-position order.
                # Map history index -> candidate-array position, not hash -> flag.
                candidate_position_by_history_index = {
                    history_index: position
                    for position, history_index in enumerate(candidate_indices)
                }

                pieces = []
                chronological_indices = row["chronological_top_indices"]
                chronological_hashes = row["chronological_top_hashes"]

                for history_index, h in zip(
                    chronological_indices, chronological_hashes
                ):
                    candidate_position = candidate_position_by_history_index[
                        history_index
                    ]
                    flag = flags[candidate_position]

                    if flag == "drop":
                        continue
                    if flag == "summary":
                        pieces.append(summaries[h])
                    else:
                        pieces.append(units[h]["raw_text"])

                retrieve_history_text = "\n\n".join(pieces)
                if retrieve_history_text:
                    retrieve_history_text += "\n\n"

                record = {
                    "question_id": qid,
                    "controller_top_indices": candidate_indices,
                    "controller_top_hashes": candidate_hashes,
                    "chronological_top_indices": chronological_indices,
                    "chronological_top_hashes": chronological_hashes,
                    "flags_controller_order": flags,
                    "final_budget_tokens": total,
                    "rounds": rounds,
                    "decisions": decisions,
                    "retrieve_history_text": retrieve_history_text,
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                done[qid] = record

                print(
                    f"[CONTROLLER] {n}/{len(remaining)} "
                    f"qid={qid} rounds={rounds} flags={flags} "
                    f"tokens={total}",
                    flush=True,
                )
    finally:
        hook.remove()

    final_rows = read_jsonl(output_path)
    final = require_unique_dict(
        final_rows, "question_id", "final controller cache"
    )

    if set(final) != set(targets):
        raise AssertionError(
            "Final controller coverage mismatch: "
            f"missing={len(set(targets) - set(final))}, "
            f"extra={len(set(final) - set(targets))}"
        )

    for qid, row in final.items():
        if row["final_budget_tokens"] > 2500:
            raise AssertionError(
                f"{qid}: final controller budget exceeds 2500"
            )
        if row["controller_top_indices"] != targets[qid]["controller_top_indices"]:
            raise AssertionError(f"{qid}: final candidate indices mismatch")

    print("FINAL_CONTROLLER_RECORDS:", len(final), flush=True)
    print("SCM_CONTROLLER_CACHE_OK", flush=True)


if __name__ == "__main__":
    main()
