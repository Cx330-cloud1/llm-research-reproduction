#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path("/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
DATA_PATH = ROOT / "data/process_data/locomo10.json"
RETRIEVAL_PATH = ROOT / "retrieval_logs/locomo10-contriever-memgas.jsonl"
SUMMARY_PATH = ROOT / "multi_granularity_logs/locomo10-summary_level.jsonl"
KEYWORD_PATH = ROOT / "multi_granularity_logs/locomo10-keyword_level.jsonl"
OUT_DIR = ROOT / "repro_logs/locomo_memgas_qa_qwen7b"
FILTER_CACHE = OUT_DIR / "filter_cache.jsonl"
FINAL_RESULTS = OUT_DIR / "final_results.jsonl"
METRICS_PATH = OUT_DIR / "metrics.json"
METADATA_PATH = OUT_DIR / "run_metadata.json"

PAPER_BASELINE = "9000a92824febdc0bc14e68de066981e8284237b"
TOPK = 3

PROMPT_G = """
You are an intelligent dialog bot. You will be shown History Dialogs. Please read, memorize, and understand the given Dialogs, then generate one concise, coherent and helpful response for the Question.

History Dialogs: {retrieved_texts}

Question Date: {question_date}
Question: {question}
"""

PROMPT_MULTIGRAN = """
You are an intelligent dialog bot. You will be shown History Dialogs and corresponding multi-granular information.
Filter the History Dialogs, summaries, and keywords to extract only the parts directly relevant to the Question. Preserve original tokens, do not paraphrase. Remove irrelevant turns, redundant info, and non-essential details.

History Dialogs: {retrieved_texts}

Question Date: {question_date}
Question: {question}
Answer:
"""


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"[WARN] skip malformed JSONL line {path}:{i}",
                    flush=True,
                )
    return rows


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_latest_by_key(path, key="sample_key"):
    out = {}
    for row in load_jsonl(path):
        if key in row:
            out[row[key]] = row
    return out


def load_multigran(path):
    out = {}
    for row in load_jsonl(path):
        if not isinstance(row, dict):
            raise RuntimeError(f"non-dict row in {path}")

        for k, v in row.items():
            if k in out:
                raise RuntimeError(
                    f"duplicate multigran key: {k}"
                )
            out[k] = v

    return out


def sample_key(idx, sample):
    payload = {
        "idx": idx,
        "conversation_id": sample["conversation_id"],
        "question": sample["question"],
        "question_date": sample.get("question_date"),
        "answer": sample["answer"],
    }

    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


make_sample_key = sample_key


def flatten_official_qa(data):
    out = []

    for entry in data:
        for qa in entry["qa"]:
            out.append({
                "conversation_id":
                    entry["conversation_id"],

                "question_type":
                    qa.get("question_type"),

                "question":
                    qa["question"],

                "answer":
                    qa["answer"],

                "question_date":
                    qa.get("question_date"),
            })

    return out


def verify_retrieval_alignment(
    flat,
    retrieved,
):
    if len(flat) != len(retrieved):
        raise RuntimeError(
            f"count mismatch: "
            f"data={len(flat)} "
            f"retrieval={len(retrieved)}"
        )

    for i, (a, b) in enumerate(
        zip(flat, retrieved)
    ):
        for field in [
            "conversation_id",
            "question",
            "answer",
            "question_date",
        ]:
            if a.get(field) != b.get(field):
                raise RuntimeError(
                    "retrieval/data mismatch "
                    f"idx={i} "
                    f"field={field}: "
                    f"{a.get(field)!r} "
                    f"!= {b.get(field)!r}"
                )


def build_maps(
    data,
    summaries,
    keywords,
):
    conv2sessions = {}
    conv2summary = {}
    conv2keyword = {}

    for entry in data:
        cid = entry["conversation_id"]

        conv2sessions[cid] = dict(
            zip(
                entry["sessions_ids"],
                entry["sessions"],
            )
        )

        conv2summary[cid] = {}
        conv2keyword[cid] = {}

        for sid in entry["sessions_ids"]:
            key = (
                f"convid-{cid}-"
                f"sessid-{sid}"
            )

            if (
                key not in summaries
                or key not in keywords
            ):
                raise RuntimeError(
                    f"missing multigran key: {key}"
                )

            conv2summary[cid][sid] = (
                summaries[key]
            )

            conv2keyword[cid][sid] = (
                keywords[key]
            )

    return (
        conv2sessions,
        conv2summary,
        conv2keyword,
    )


def build_filter_prompt(
    sample,
    conv2sessions,
    conv2summary,
    conv2keyword,
    topk=TOPK,
):
    cid = sample["conversation_id"]

    ranked = (
        sample["retrieval_results"]
        ["ranked_items"][:topk]
    )

    if len(ranked) < topk:
        raise RuntimeError(
            f"need top-{topk}, "
            f"got {len(ranked)} "
            f"for {cid}"
        )

    retrieved_texts = ""

    for item in ranked:
        sid = item["corpus_id"]

        session = (
            conv2sessions[cid][sid]
        )

        summary = (
            conv2summary[cid][sid]
        )

        keyword = (
            conv2keyword[cid][sid]
        )

        # Preserve official
        # generation_multigran.py behavior:
        # the list object is directly rendered.
        retrieved_texts += (
            f"\n### Session Date: "
            f"{item['timestamp']}\n"

            f"Session Content:\n"
            f"{session}\n\n"

            f"Session Summary:\n"
            f"{summary}\n"

            f"Session Keyword:\n"
            f"{keyword}\n"
        )

    return PROMPT_MULTIGRAN.format(
        retrieved_texts=retrieved_texts,
        question=sample["question"],
        question_date=sample[
            "question_date"
        ],
    )


def build_answer_prompt(
    sample,
    filtered_response,
    topk=TOPK,
):
    ranked = (
        sample["retrieval_results"]
        ["ranked_items"][:topk]
    )

    if not ranked:
        raise RuntimeError(
            "empty retrieval result"
        )

    # Compatibility fix:
    # official generation_multigran.py
    # inserts exactly the SAME filtered
    # response once for every top-k item.
    #
    # We intentionally insert it once.
    retrieved_texts = (
        f"\n### Session Date: "
        f"{ranked[0]['timestamp']}\n"

        f"Session Content:\n"
        f"{filtered_response}\n"
    )

    return PROMPT_G.format(
        retrieved_texts=retrieved_texts,
        question=sample["question"],
        question_date=sample[
            "question_date"
        ],
    )


def get_answer(ans):
    # Preserve official post-processing,
    # including Python str.strip behavior.

    strip_words = [
        "\nDialogs:",
        "\n[bot]:",
        "\nAssistant:",
        "\nReview:",
        "\n",
        "[bot]:",
    ]

    cut_words = [
        "\n[human]:",
        "\nQuestion:",
        "\nQ:",
    ]

    for word in strip_words:
        ans = (
            ans.strip(word)
            if ans is not None
            else ""
        )

    for word in cut_words:
        if word in ans:
            ans = ans.split(word)[0]

    return ans


def load_inputs():
    data = json.load(
        DATA_PATH.open(
            encoding="utf-8"
        )
    )

    retrieved = load_jsonl(
        RETRIEVAL_PATH
    )

    summaries = load_multigran(
        SUMMARY_PATH
    )

    keywords = load_multigran(
        KEYWORD_PATH
    )

    verify_retrieval_alignment(
        flatten_official_qa(data),
        retrieved,
    )

    maps = build_maps(
        data,
        summaries,
        keywords,
    )

    return (
        data,
        retrieved,
        *maps,
    )


def git_head():
    try:
        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=ROOT,
            text=True,
        ).strip()

    except Exception:
        return None


class QwenGreedy:

    def __init__(self):
        import torch

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        self.torch = torch

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                MODEL_PATH,
                local_files_only=True,
            )
        )

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                local_files_only=True,
            )
        )

        self.model.eval()

        # do_sample=False.
        # These fields do not affect greedy
        # generation; clearing them only removes
        # irrelevant Transformers warnings.
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

        vals = []

        for v in [
            getattr(
                self.tokenizer,
                "model_max_length",
                None,
            ),
            getattr(
                self.model.config,
                "max_position_embeddings",
                None,
            ),
        ]:
            if (
                isinstance(v, int)
                and 0 < v < 1_000_000
            ):
                vals.append(v)

        self.context_limit = (
            min(vals)
            if vals
            else None
        )


    def generate(
        self,
        prompt,
        max_new_tokens,
    ):
        chat_text = (
            self.tokenizer
            .apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        inputs = self.tokenizer(
            chat_text,
            return_tensors="pt",
        ).to(
            self.model.device
        )

        n_in = int(
            inputs["input_ids"]
            .shape[1]
        )

        if (
            self.context_limit
            and
            n_in + max_new_tokens
            > self.context_limit
        ):
            raise RuntimeError(
                "QWEN_CONTEXT_LIMIT "
                f"input_tokens={n_in} "
                f"max_new_tokens="
                f"{max_new_tokens} "
                f"context="
                f"{self.context_limit}"
            )

        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=
                    max_new_tokens,
                do_sample=False,
                pad_token_id=
                    self.tokenizer
                    .eos_token_id,
            )

        new = output[0][n_in:]

        text = (
            self.tokenizer
            .decode(
                new,
                skip_special_tokens=True,
            )
            .strip()
        )

        return (
            text,
            {
                "input_tokens":
                    n_in,

                "output_tokens":
                    int(new.shape[0]),
            },
        )


def write_metadata(
    args,
    n,
):
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    meta = {
        "label":
            "MemGAS original-task "
            "compatibility reproduction "
            "— LoCoMo QA",

        "paper_baseline_commit":
            PAPER_BASELINE,

        "current_git_head":
            git_head(),

        "dataset":
            "locomo10",

        "questions":
            n,

        "retriever":
            "contriever",

        "method":
            "memgas",

        "topk":
            TOPK,

        "paper_generator":
            "gpt-4o-mini",

        "compat_generator":
            str(MODEL_PATH),

        "max_new_tokens":
            args.max_new_tokens,

        "multi_granularity_generator":
            "Qwen2.5-7B-Instruct",

        "filtered_response_"
        "duplication_bug_preserved":
            False,
    }

    METADATA_PATH.write_text(
        json.dumps(
            meta,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def inspect(args):

    (
        data,
        retrieved,
        _,
        summaries,
        keywords,
    ) = load_inputs()

    print(
        "===== LOCOMO MEMGAS "
        "QA INSPECT ====="
    )

    print(
        "conversations:",
        len(data),
    )

    print(
        "sessions:",
        sum(
            len(x["sessions"])
            for x in data
        ),
    )

    print(
        "questions:",
        sum(
            len(x["qa"])
            for x in data
        ),
    )

    print(
        "retrieval_rows:",
        len(retrieved),
    )

    print(
        "summary_keys:",
        sum(
            len(x)
            for x in
            summaries.values()
        ),
    )

    print(
        "keyword_keys:",
        sum(
            len(x)
            for x in
            keywords.values()
        ),
    )

    print(
        "topk:",
        TOPK,
    )

    print(
        "generator:",
        MODEL_PATH,
    )

    print(
        "max_new_tokens:",
        args.max_new_tokens,
    )

    print(
        "LOCOMO_MEMGAS_QA_"
        "INSPECT_OK"
    )


def generate(args):

    (
        _,
        retrieved,
        conv2sessions,
        conv2summary,
        conv2keyword,
    ) = load_inputs()

    write_metadata(
        args,
        len(retrieved),
    )

    filters = load_latest_by_key(
        FILTER_CACHE
    )

    finals = load_latest_by_key(
        FINAL_RESULTS
    )

    print(
        "===== RESUME ====="
    )

    print(
        "filter_cached:",
        len(filters),
    )

    print(
        "final_cached:",
        len(finals),
    )

    print(
        "total:",
        len(retrieved),
    )

    model = QwenGreedy()

    print(
        "qwen_context_limit:",
        model.context_limit,
    )

    start = max(
        args.start,
        0,
    )

    stop = len(
        retrieved
    )

    if args.limit is not None:
        stop = min(
            stop,
            start + args.limit,
        )

    for idx in range(
        start,
        stop,
    ):
        sample = retrieved[idx]

        key = sample_key(
            idx,
            sample,
        )

        if key in finals:
            print(
                f"[{idx+1}/"
                f"{len(retrieved)}] "
                "FINAL HIT",
                flush=True,
            )

            continue

        if key in filters:
            filtered = (
                filters[key]
                ["filtered_response"]
            )

            print(
                f"[{idx+1}/"
                f"{len(retrieved)}] "
                "FILTER HIT",
                flush=True,
            )

        else:
            prompt = (
                build_filter_prompt(
                    sample,
                    conv2sessions,
                    conv2summary,
                    conv2keyword,
                )
            )

            (
                filtered,
                meta,
            ) = model.generate(
                prompt,
                args.max_new_tokens,
            )

            row = {
                "sample_key":
                    key,

                "global_idx":
                    idx,

                "conversation_id":
                    sample[
                        "conversation_id"
                    ],

                "question":
                    sample["question"],

                "filtered_response":
                    filtered,

                "generation_meta":
                    meta,
            }

            append_jsonl(
                FILTER_CACHE,
                row,
            )

            filters[key] = row

            print(
                f"[{idx+1}/"
                f"{len(retrieved)}] "
                "FILTER MISS "
                f"in="
                f"{meta['input_tokens']} "
                f"out="
                f"{meta['output_tokens']}",
                flush=True,
            )

        prompt = (
            build_answer_prompt(
                sample,
                filtered,
            )
        )

        (
            response,
            meta,
        ) = model.generate(
            prompt,
            args.max_new_tokens,
        )

        row = dict(
            sample
        )

        row.update({
            "sample_key":
                key,

            "global_idx":
                idx,

            "filtered_response":
                filtered,

            "response":
                response,

            "answer_generation_meta":
                meta,

            "compatibility": {
                "generator":
                    "Qwen2.5-7B-"
                    "Instruct",

                "paper_generator":
                    "gpt-4o-mini",

                "filtered_response_"
                "duplication_bug_"
                "preserved":
                    False,
            },
        })

        append_jsonl(
            FINAL_RESULTS,
            row,
        )

        finals[key] = row

        print(
            f"[{idx+1}/"
            f"{len(retrieved)}] "
            "ANSWER OK "
            f"in="
            f"{meta['input_tokens']} "
            f"out="
            f"{meta['output_tokens']} "
            f"completed="
            f"{len(finals)}/"
            f"{len(retrieved)}",
            flush=True,
        )

    print(
        "LOCOMO_MEMGAS_QA_"
        "GENERATION_STAGE_OK"
    )

    print(
        "completed_final:",
        len(
            load_latest_by_key(
                FINAL_RESULTS
            )
        ),
    )

    print(
        "total_expected:",
        len(retrieved),
    )


def metrics():

    (
        _,
        retrieved,
        _,
        _,
        _,
    ) = load_inputs()

    finals = load_latest_by_key(
        FINAL_RESULTS
    )

    ordered = []
    missing = []

    for idx, sample in enumerate(
        retrieved
    ):
        key = sample_key(
            idx,
            sample,
        )

        if key in finals:
            ordered.append(
                finals[key]
            )

        else:
            missing.append(
                idx
            )

    if missing:
        raise RuntimeError(
            "incomplete generation: "
            f"complete="
            f"{len(ordered)} "
            f"total="
            f"{len(retrieved)} "
            f"missing_first20="
            f"{missing[:20]}"
        )

    sys.path.insert(
        0,
        str(
            ROOT
            / "src/generation"
        ),
    )

    from metrics import (
        evaluate_match,
        evaluate_sim,
    )

    preds = [
        get_answer(
            x["response"]
        )
        for x in ordered
    ]

    refs = [
        str(
            x["answer"]
        )
        for x in ordered
    ]

    result = {
        "n":
            len(ordered),

        "match":
            evaluate_match(
                preds,
                refs,
                truncate_pred=False,
            ),
    }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "===== MATCH ====="
    )

    print(
        json.dumps(
            result["match"],
            ensure_ascii=False,
            indent=2,
        )
    )

    try:
        sim = evaluate_sim(
            preds,
            refs,
            truncate_pred=False,
        )

    except Exception as e:
        result[
            "similarity_metrics_error"
        ] = repr(e)

        METRICS_PATH.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "LOCOMO_MEMGAS_QA_"
            "SIM_METRICS_FAILED"
        )

        print(
            repr(e)
        )

        raise

    result["similarity"] = sim

    result[
        "similarity_percent"
    ] = {
        k: v * 100
        for k, v in sim.items()
    }

    METRICS_PATH.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "===== SIMILARITY (%) ====="
    )

    print(
        json.dumps(
            result[
                "similarity_percent"
            ],
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "LOCOMO_MEMGAS_QA_"
        "METRICS_OK"
    )


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--stage",
        choices=[
            "inspect",
            "generate",
            "metrics",
            "all",
        ],
        default="all",
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=4000,
    )

    p.add_argument(
        "--start",
        type=int,
        default=0,
    )

    p.add_argument(
        "--limit",
        type=int,
    )

    args = p.parse_args()

    if args.stage == "inspect":
        inspect(args)
        return

    if args.stage in [
        "generate",
        "all",
    ]:
        generate(args)

    if args.stage in [
        "metrics",
        "all",
    ]:
        metrics()


if __name__ == "__main__":
    main()
