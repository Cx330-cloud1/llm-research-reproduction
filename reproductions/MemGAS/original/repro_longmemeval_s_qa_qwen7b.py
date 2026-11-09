import os
import json
import hashlib
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

ROOT = Path("/root/autodl-tmp/paper_repro/MemGAS")
DATA_PATH = ROOT / "data/process_data/longmemeval_s.json"
MODEL_PATH = Path("/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
RETRIEVAL_DIR = ROOT / "retrieval_logs"
OUTPUT_DIR = ROOT / "generation_logs/longmemeval_s_qwen7b"
LOG_DIR = ROOT / "repro_logs"

METHODS = [
    "session_level",
    "turn_level",
    "summary_level",
    "keyword_level",
    "hybrid_level",
    "memgas",
]

TOPK = 3
MAX_NEW_TOKENS = 128
MAX_MEMORY_TOKENS = 7000
MAX_INPUT_TOKENS = 28000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(
                    f"Malformed JSONL: {path}:{line_no}: {e}"
                )
    return rows


def atomic_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def output_paths(method):
    name = (
        f"longmemeval_s-contriever-{method}"
        "-filter-qwen2.5-7b-topk_3.jsonl"
    )
    out = OUTPUT_DIR / name
    return out, out.with_suffix(".meta.json")


def load_existing_output(path, method):
    if not path.exists():
        return {}

    # If a crash left an incomplete final line, retain all earlier
    # complete JSONL records and remove only that incomplete tail.
    raw_lines = path.read_text(encoding="utf-8").splitlines(
        keepends=True
    )

    valid_lines = []
    rows = []

    for i, line in enumerate(raw_lines):
        try:
            row = json.loads(line)
        except Exception:
            if i == len(raw_lines) - 1:
                print(
                    f"[RESUME] Removing incomplete final line: {path}",
                    flush=True,
                )
                break
            raise RuntimeError(
                f"Corrupt non-final line {i + 1} in {path}"
            )

        valid_lines.append(line)
        rows.append(row)

    if len(valid_lines) != len(raw_lines):
        path.write_text("".join(valid_lines), encoding="utf-8")

    completed = {}

    for row in rows:
        qid = row.get("question_id")
        if not qid:
            raise RuntimeError(f"Missing question_id in {path}")

        if row.get("method") != method:
            raise RuntimeError(f"Method mismatch in {path}")

        if row.get("topk") != TOPK:
            raise RuntimeError(f"Top-k mismatch in {path}")

        if qid in completed:
            raise RuntimeError(f"Duplicate question_id {qid} in {path}")

        completed[qid] = row

    return completed


def validate_inputs(data, data_by_id, method):
    path = RETRIEVAL_DIR / (
        f"longmemeval_s-contriever-{method}.jsonl"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    rows = read_jsonl(path)

    if len(rows) != 500:
        raise ValueError(
            f"{method}: expected 500 rows, got {len(rows)}"
        )

    seen = set()

    for row in rows:
        cid = row["conversation_id"]

        if cid not in data_by_id:
            raise ValueError(f"{method}: unknown conversation {cid}")

        if cid in seen:
            raise ValueError(f"{method}: duplicate question {cid}")

        seen.add(cid)

        conv = data_by_id[cid]

        matches = [
            qa for qa in conv["qa"]
            if qa["question"] == row["question"]
        ]

        if len(matches) != 1:
            raise ValueError(
                f"{method}: question mismatch for {cid}"
            )

        qa = matches[0]

        if str(qa["answer"]) != str(row["answer"]):
            raise ValueError(
                f"{method}: gold answer mismatch for {cid}"
            )

        sid_map = {
            sid: "\n".join(session)
            for sid, session in zip(
                conv["sessions_ids"], conv["sessions"]
            )
        }

        for item in row["retrieval_results"]["ranked_items"]:
            if item["corpus_id"] not in sid_map:
                raise ValueError(
                    f"{method}: missing session ID "
                    f"{item['corpus_id']} for {cid}"
                )

    print(
        f"[VALID] {method}: {len(rows)} questions, "
        "all retrieved IDs mapped",
        flush=True,
    )

    return rows


def build_prompt(row, conv, tokenizer):
    sid_map = {
        sid: "\n".join(session)
        for sid, session in zip(
            conv["sessions_ids"], conv["sessions"]
        )
    }

    selected = row["retrieval_results"]["ranked_items"][:TOPK]
    memory_blocks = []
    selected_ids = []
    truncated_ids = []

    for rank, item in enumerate(selected, 1):
        sid = item["corpus_id"]
        text = sid_map[sid]

        token_ids = tokenizer.encode(
            text, add_special_tokens=False
        )

        if len(token_ids) > MAX_MEMORY_TOKENS:
            text = tokenizer.decode(
                token_ids[:MAX_MEMORY_TOKENS],
                skip_special_tokens=True,
            )
            truncated_ids.append(sid)

        timestamp = item.get("timestamp", "")
        memory_blocks.append(
            f"[Memory {rank} | Date: {timestamp}]\n{text}"
        )
        selected_ids.append(sid)

    history = "\n\n".join(memory_blocks)

    prompt = (
        "I will give you several history chats between you and a user. "
        "Please answer the question based on the relevant chat history."
        "\n\nHistory Chats:\n\n"
        f"{history}"
        f"\n\nCurrent Date: {row.get('question_date', '')}"
        f"\nQuestion: {row['question']}"
        "\nAnswer:"
    )

    # Safety check: top-3 sessions are individually capped, but also
    # verify the actual tokenized prompt fits the configured budget.
    messages = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if ids.shape[-1] > MAX_INPUT_TOKENS:
        raise RuntimeError(
            f"Prompt too long: {ids.shape[-1]} tokens "
            f"for question {row['conversation_id']}. "
            "No answer was generated; existing checkpoints are safe."
        )

    return prompt, selected_ids, truncated_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=METHODS + ["all"],
        default="all",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum NEW answers per method; useful for smoke tests.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    if not (MODEL_PATH / "config.json").exists():
        raise FileNotFoundError(MODEL_PATH / "config.json")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if len(data) != 500:
        raise ValueError(f"Expected 500 records, got {len(data)}")

    data_by_id = {x["conversation_id"]: x for x in data}

    if len(data_by_id) != 500:
        raise ValueError("Duplicate conversation IDs in processed data")

    methods = METHODS if args.method == "all" else [args.method]

    # Preflight all selected retrieval files BEFORE loading the model.
    retrieval_rows = {}
    manifests = {}

    for method in methods:
        path = RETRIEVAL_DIR / (
            f"longmemeval_s-contriever-{method}.jsonl"
        )
        retrieval_rows[method] = validate_inputs(
            data, data_by_id, method
        )

        manifests[method] = {
            "dataset_sha256": sha256(DATA_PATH),
            "retrieval_sha256": sha256(path),
            "model_config_sha256": sha256(
                MODEL_PATH / "config.json"
            ),
            "model_path": str(MODEL_PATH),
            "method": method,
            "topk": TOPK,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_memory_tokens": MAX_MEMORY_TOKENS,
            "prompt": "LongMemEval default non-CoT history-chat template",
            "dtype": "float16",
        }

        out, meta = output_paths(method)

        if meta.exists():
            previous = json.loads(meta.read_text(encoding="utf-8"))
            if previous != manifests[method]:
                raise RuntimeError(
                    f"Manifest mismatch for {method}: {meta}"
                )
        elif out.exists() and out.stat().st_size > 0:
            raise RuntimeError(
                f"Output exists without a manifest: {out}. "
                "Refusing unsafe resume."
            )
        else:
            atomic_json(meta, manifests[method])

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Loading local Qwen2.5-7B-Instruct...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("cuda:0")
    model.eval()

    for method in methods:
        rows = retrieval_rows[method]
        out_path, _ = output_paths(method)
        completed = load_existing_output(out_path, method)

        print(
            f"\n===== {method} =====\n"
            f"completed={len(completed)}/500",
            flush=True,
        )

        new_count = 0

        with open(out_path, "a", encoding="utf-8") as out:
            for row in tqdm(
                rows,
                desc=f"QA {method}",
                unit="question",
            ):
                qid = row["conversation_id"]

                if qid in completed:
                    continue

                if (
                    args.limit is not None
                    and new_count >= args.limit
                ):
                    break

                conv = data_by_id[qid]
                prompt, selected_ids, truncated_ids = build_prompt(
                    row, conv, tokenizer
                )

                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to("cuda:0")

                attention_mask = torch.ones_like(inputs)

                with torch.inference_mode():
                    output = model.generate(
                        input_ids=inputs,
                        attention_mask=attention_mask,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )

                generated = output[0, inputs.shape[-1]:]
                hypothesis = tokenizer.decode(
                    generated,
                    skip_special_tokens=True,
                ).strip()

                if not hypothesis:
                    raise RuntimeError(
                        f"Empty answer for {qid}; stopping safely."
                    )

                result = {
                    # LongMemEval official evaluator input fields:
                    "question_id": qid,
                    "hypothesis": hypothesis,

                    # Reproduction audit metadata:
                    "conversation_id": qid,
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "question_type": row["question_type"],
                    "question_date": row.get("question_date", ""),
                    "method": method,
                    "topk": TOPK,
                    "model": "Qwen2.5-7B-Instruct",
                    "retrieved_session_ids": selected_ids,
                    "truncated_session_ids": truncated_ids,
                }

                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                os.fsync(out.fileno())

                completed[qid] = result
                new_count += 1

                if new_count == 1 or new_count % 25 == 0:
                    print(
                        f"\n[{method}] new={new_count} "
                        f"total={len(completed)}/500 "
                        f"last_question={qid}",
                        flush=True,
                    )

        print(
            f"[DONE] {method}: "
            f"{len(completed)}/500 rows saved",
            flush=True,
        )

    del model
    torch.cuda.empty_cache()

    print("\nLONGMEMEVAL_S_QA_GENERATION_OK", flush=True)


if __name__ == "__main__":
    main()
