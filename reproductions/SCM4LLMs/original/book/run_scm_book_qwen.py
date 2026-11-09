#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import chardet
import tiktoken
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Paths / fixed configuration
# ============================================================

OFFICIAL_REPO = Path(
    "/root/autodl-tmp/paper_repro/SCM4LLMs"
)

OFFICIAL_COMMIT = (
    "26216044f5d4caccc0303d86667b491ead11d878"
)

MODEL_PATH = Path(
    "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
)

OUT_ROOT = Path(
    "/root/autodl-tmp/paper_repro/"
    "adapters/scm_book_repro/output"
)


# Preserve official SCM book token-budget accounting.
TOKENIZER_MODEL = "text-davinci-003"

BOOK_SPLIT_MAX_TOKENS = 3000
PREVIOUS_RAW_THRESHOLD = 500

HIER_GROUP_TOKENS = 2300
OFFICIAL_CONTEXT_LIMIT = 4000

# Official text-davinci-003 API code used max_tokens <= 1024.
OFFICIAL_API_MAX_NEW_TOKENS = 1024


enc = tiktoken.encoding_for_model(
    TOKENIZER_MODEL
)


# ============================================================
# Official tools.py logic used by Book task
# ============================================================

LANG_EN = "English"
LANG_ZH = "Chinese"


def detect_language(text: str) -> str:
    total_count = len(text)
    en_count = len(
        re.findall(
            r"[a-zA-Z]",
            text,
        )
    )

    if (
        total_count > 0
        and en_count / total_count > 0.6
    ):
        return LANG_EN

    return LANG_ZH


def detect_encode_type(
    file_path: Path
) -> str:

    data = file_path.read_bytes()

    result = chardet.detect(
        data
    )

    encoding = result[
        "encoding"
    ]

    if not encoding:
        raise RuntimeError(
            "chardet could not detect "
            f"encoding: {file_path}"
        )

    return encoding


def toklen(
    text: str
) -> int:

    return len(
        enc.encode(text)
    )


# ============================================================
# Import ONLY official book prompts.
#
# We deliberately DO NOT import core.api / core.book because
# they initialize the historical OpenAI API layer.
# ============================================================

sys.path.insert(
    0,
    str(OFFICIAL_REPO),
)

from prompts.book import (  # noqa: E402
    zh_start_prompt,
    en_start_prompt,
    zh_agent_scm_prompt,
    en_agent_scm_prompt,
    zh_agent_no_scm_prompt,
    en_agent_no_scm_prompt,
    zh_hierarchical_summarization_prompt,
    en_hierarchical_summarization_prompt,
)


def choose_language_template(
    lang2template,
    text,
):
    lang = detect_language(
        text
    )

    if lang not in lang2template:
        lang = LANG_EN

    return lang2template[
        lang
    ]


# ============================================================
# Official BookSpliter logic
# utils/spliter.py
# ============================================================

def split_book_official(
    book_file: Path
):

    encoding = (
        detect_encode_type(
            book_file
        )
    )

    print(
        "[SPLIT] detected "
        f"encoding={encoding}",
        flush=True,
    )

    if (
        "gb"
        in encoding.lower()
    ):
        encoding = "gbk"


    with open(
        book_file,
        "r",
        encoding=encoding,
    ) as f:

        book = f.read()


    lang = detect_language(
        book
    )

    if lang == LANG_ZH:
        separator = "。"
    else:
        separator = ". "


    split_book = []

    current_tokens = 0
    current_block = ""


    # Keep official behavior:
    # token count does NOT include separator.
    for sentence in book.split(
        separator
    ):

        count = toklen(
            sentence
        )

        if (
            current_tokens
            + count
            > BOOK_SPLIT_MAX_TOKENS
        ):

            split_book.append(
                current_block
            )

            current_block = (
                sentence
                + separator
            )

            current_tokens = (
                count
            )

        else:

            current_tokens += (
                count
            )

            current_block += (
                sentence
                + separator
            )


    if current_block:
        split_book.append(
            current_block
        )


    return (
        split_book,
        lang,
        encoding,
    )


# ============================================================
# Qwen compatibility generator
# ============================================================

class QwenGenerator:

    def __init__(
        self
    ):

        print(
            "[MODEL] loading "
            f"{MODEL_PATH}",
            flush=True,
        )


        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                str(
                    MODEL_PATH
                ),
                local_files_only=True,
                trust_remote_code=True,
            )
        )


        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                str(
                    MODEL_PATH
                ),
                torch_dtype=(
                    torch.bfloat16
                ),
                device_map="auto",
                max_memory={
                    0: "18GiB",
                    "cpu": "24GiB",
                },
                low_cpu_mem_usage=True,
                local_files_only=True,
                trust_remote_code=True,
            )
        )


        self.model.eval()


        print(
            "[MODEL] "
            "HF_DEVICE_MAP:",
            getattr(
                self.model,
                "hf_device_map",
                None,
            ),
            flush=True,
        )


        # Same low-memory lm_head hook
        # already tested in our earlier
        # Qwen compatibility runs.
        def lm_head_pre_hook(
            module,
            args,
        ):

            hidden = args[0]

            if (
                hidden.ndim == 3
                and hidden.shape[1] > 1
            ):

                hidden = (
                    hidden[
                        :,
                        -1:,
                        :
                    ]
                )

            return (
                hidden,
            )


        self._hook = (
            self.model
            .lm_head
            .register_forward_pre_hook(
                lm_head_pre_hook
            )
        )


    @staticmethod
    def apply_official_stop_strings(
        text: str
    ) -> str:

        # Official text-davinci code:
        #
        # stop=[
        #   "\n\n\n",
        #   "###"
        # ]
        #
        # Here we apply the equivalent
        # truncation after Qwen decode.

        positions = []

        for marker in (
            "\n\n\n",
            "###",
        ):

            pos = text.find(
                marker
            )

            if pos >= 0:
                positions.append(
                    pos
                )


        if positions:

            text = text[
                :min(positions)
            ]


        return text.strip()


    def generate(
        self,
        prompt: str,
    ) -> str:

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]


        chat_text = (
            self.tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )


        inp = self.tokenizer(
            chat_text,
            return_tensors="pt",
        )


        device = (
            next(
                self.model
                .parameters()
            ).device
        )


        inp = {
            k: v.to(device)
            for k, v
            in inp.items()
        }


        with torch.inference_mode():

            out = (
                self.model
                .generate(
                    **inp,
                    max_new_tokens=(
                        OFFICIAL_API_MAX_NEW_TOKENS
                    ),
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=(
                        self.tokenizer
                        .eos_token_id
                    ),
                )
            )


        new_ids = out[
            0,
            inp[
                "input_ids"
            ].shape[1]:
        ]


        text = (
            self.tokenizer
            .decode(
                new_ids,
                skip_special_tokens=True,
            )
            .strip()
        )


        return (
            self
            .apply_official_stop_strings(
                text
            )
        )


    def close(
        self
    ):
        self._hook.remove()


# ============================================================
# Cache helpers
# ============================================================

def sha256_text(
    text: str
) -> str:

    return (
        hashlib
        .sha256(
            text.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


class JsonlCache:

    def __init__(
        self,
        path: Path,
    ):

        self.path = path
        self.rows = {}


        if path.exists():

            with open(
                path,
                encoding="utf-8",
            ) as f:

                for line in f:

                    line = (
                        line.strip()
                    )

                    if not line:
                        continue

                    row = json.loads(
                        line
                    )

                    self.rows[
                        row[
                            "cache_key"
                        ]
                    ] = row


    def get(
        self,
        key,
    ):

        row = self.rows.get(
            key
        )

        if row is None:
            return None

        return row[
            "output"
        ]


    def put(
        self,
        key,
        stage,
        prompt,
        output,
        meta=None,
    ):

        if key in self.rows:
            return


        row = {
            "cache_key":
                key,

            "stage":
                stage,

            "prompt_sha256":
                sha256_text(
                    prompt
                ),

            "output":
                output,

            "meta":
                meta or {},
        }


        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        with open(
            self.path,
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


        self.rows[
            key
        ] = row


# ============================================================
# Official prompt construction
# ============================================================

def first_prompt(
    text
):

    template = (
        choose_language_template(
            {
                LANG_EN:
                    en_start_prompt,

                LANG_ZH:
                    zh_start_prompt,
            },
            text,
        )
    )


    return template.format(
        text=text
    )


def scm_prompt(
    current_text,
    previous_content,
):

    template = (
        choose_language_template(
            {
                LANG_EN:
                    en_agent_scm_prompt,

                LANG_ZH:
                    zh_agent_scm_prompt,
            },
            current_text,
        )
    )


    return template.format(
        previous_content=(
            previous_content
        ),
        current_text=(
            current_text
        ),
    )


def no_scm_prompt(
    text
):

    template = (
        choose_language_template(
            {
                LANG_EN:
                    en_agent_no_scm_prompt,

                LANG_ZH:
                    zh_agent_no_scm_prompt,
            },
            text,
        )
    )


    return template.format(
        text=text
    )


def hierarchical_prompt(
    paragraph_summaries,
    max_tokens,
):

    template = (
        choose_language_template(
            {
                LANG_EN:
                    en_hierarchical_summarization_prompt,

                LANG_ZH:
                    zh_hierarchical_summarization_prompt,
            },
            paragraph_summaries,
        )
    )


    return template.format(
        paragraph_summaries=(
            paragraph_summaries.strip()
        ),
        max_tokens=max_tokens,
    )


# ============================================================
# Exact SummaryBot.group_strings
# ============================================================

def group_strings(
    content_lst,
    summary_token_length_lst,
    group_tokens=2300,
):

    result = []

    curr_group = []
    curr_len = 0


    for i in range(
        len(content_lst)
    ):

        if (
            summary_token_length_lst[i]
            > group_tokens
        ):

            if curr_group:
                result.append(
                    curr_group
                )

            result.append(
                [
                    content_lst[i]
                ]
            )

            curr_group = []
            curr_len = 0


        elif (
            curr_len
            + summary_token_length_lst[i]
            > group_tokens
        ):

            result.append(
                curr_group
            )

            curr_group = [
                content_lst[i]
            ]

            curr_len = (
                summary_token_length_lst[i]
            )


        else:

            curr_group.append(
                content_lst[i]
            )

            curr_len += (
                summary_token_length_lst[i]
            )


    if curr_group:
        result.append(
            curr_group
        )


    return [
        "\n\n"
        .join(group)
        .strip()

        for group
        in result
    ]


# ============================================================
# Exact recursive final-summary structure
# ============================================================

def divide_conquer_summary(
    content_lst,
    generator,
    cache,
    book_slug,
    mode,
    depth=1,
):

    summary_token_length_lst = [
        toklen(txt)
        for txt
        in content_lst
    ]


    total_tokens = sum(
        summary_token_length_lst
    )


    print(
        "[FINAL] "
        f"mode={mode} "
        f"depth={depth} "
        f"count={len(content_lst)} "
        f"total_tokens={total_tokens}",
        flush=True,
    )


    groups = group_strings(
        content_lst,
        summary_token_length_lst,
        group_tokens=(
            HIER_GROUP_TOKENS
        ),
    )


    single_summary_max_tokens = (
        1000
    )


    if len(groups) == 1:
        single_summary_max_tokens = (
            1500
        )


    group_summaries = []


    for i, text in enumerate(
        groups
    ):

        token_count = toklen(
            text
        )


        current_tokens = max(
            token_count,
            HIER_GROUP_TOKENS,
        )


        tgt_tokens = min(
            OFFICIAL_CONTEXT_LIMIT
            - current_tokens
            - 400,

            single_summary_max_tokens,
        )


        prompt = hierarchical_prompt(
            text,
            max_tokens=tgt_tokens,
        )


        key = (
            f"{book_slug}|"
            f"{mode}|"
            f"hier|"
            f"depth={depth}|"
            f"group={i}|"
            f"{sha256_text(prompt)}"
        )


        output = cache.get(
            key
        )


        if output is None:

            output = (
                generator
                .generate(
                    prompt
                )
            )


            cache.put(
                key=key,
                stage=(
                    "hierarchical_summary"
                ),
                prompt=prompt,
                output=output,
                meta={
                    "depth":
                        depth,

                    "group_index":
                        i,

                    "group_count":
                        len(groups),

                    "input_token_count_p50k":
                        token_count,

                    "prompt_target_max_characters":
                        tgt_tokens,
                },
            )


            cache_state = (
                "MISS"
            )

        else:

            cache_state = (
                "HIT"
            )


        print(
            "[FINAL] "
            f"{cache_state} "
            f"mode={mode} "
            f"depth={depth} "
            f"group={i+1}/"
            f"{len(groups)} "
            f"out_p50k="
            f"{toklen(output)}",
            flush=True,
        )


        group_summaries.append(
            output
        )


    if (
        len(group_summaries)
        > 1
    ):

        return (
            divide_conquer_summary(
                group_summaries,
                generator,
                cache,
                book_slug,
                mode,
                depth=depth + 1,
            )
        )


    return group_summaries[
        0
    ]


# ============================================================
# Run one book / one condition
# ============================================================

def run_book(
    book_file: Path,
    mode: str,
    generator: QwenGenerator,
):

    assert mode in {
        "scm",
        "no_scm",
    }


    book_name = (
        book_file.name
    )

    book_slug = (
        book_name
        .replace(
            "/",
            "_"
        )
    )


    run_dir = (
        OUT_ROOT
        / book_slug
        / mode
    )


    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    split_file = (
        run_dir
        / "split.json"
    )

    chunk_file = (
        run_dir
        / "chunk_summaries.jsonl"
    )

    cache_file = (
        run_dir
        / "generation_cache.jsonl"
    )

    final_file = (
        run_dir
        / "final_summary.txt"
    )

    result_file = (
        run_dir
        / "result.json"
    )


    print()
    print(
        "=" * 80
    )

    print(
        "[BOOK]",
        book_name,
    )

    print(
        "[MODE]",
        mode,
    )

    print(
        "=" * 80,
        flush=True,
    )


    (
        chunks,
        lang,
        encoding,
    ) = split_book_official(
        book_file
    )


    token_lengths = [
        toklen(x)
        for x
        in chunks
    ]


    split_payload = {
        "book_file":
            str(book_file),

        "book_name":
            book_name,

        "language":
            lang,

        "encoding":
            encoding,

        "tokenizer_model":
            TOKENIZER_MODEL,

        "split_max_tokens":
            BOOK_SPLIT_MAX_TOKENS,

        "chunks":
            chunks,

        "chunk_token_lengths_p50k":
            token_lengths,
    }


    split_file.write_text(
        json.dumps(
            split_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print(
        "[SPLIT] "
        f"chunks={len(chunks)} "
        f"min={min(token_lengths)} "
        f"max={max(token_lengths)} "
        f"sum={sum(token_lengths)}",
        flush=True,
    )


    # --------------------------------------------------------
    # Resume already-finished chunks
    # --------------------------------------------------------

    completed = {}


    if chunk_file.exists():

        with open(
            chunk_file,
            encoding="utf-8",
        ) as f:

            for line in f:

                line = (
                    line.strip()
                )

                if not line:
                    continue

                row = json.loads(
                    line
                )

                completed[
                    int(
                        row[
                            "chunk_index"
                        ]
                    )
                ] = row


    cache = JsonlCache(
        cache_file
    )


    histories = []


    with open(
        chunk_file,
        "a",
        encoding="utf-8",
    ) as fout:


        for i, text in enumerate(
            chunks
        ):


            # -----------------------------------------------
            # Resume
            # -----------------------------------------------

            if i in completed:

                row = completed[
                    i
                ]


                expected = (
                    sha256_text(
                        text
                    )
                )


                if (
                    row[
                        "paragraph_sha256"
                    ]
                    != expected
                ):

                    raise RuntimeError(
                        "resume mismatch "
                        f"{book_name} "
                        f"{mode} "
                        f"chunk={i}"
                    )


                summary = row[
                    "summary"
                ]


                histories.append({
                    "paragraph":
                        text,

                    "summary":
                        summary,

                    "content_tokens_length":
                        toklen(text),

                    "summary_tokens_length":
                        toklen(summary),
                })


                print(
                    "[CHUNK] HIT "
                    f"{i+1}/"
                    f"{len(chunks)} "
                    f"mode={mode} "
                    "summary_p50k="
                    f"{toklen(summary)}",
                    flush=True,
                )


                continue


            # -----------------------------------------------
            # Build official prompt
            # -----------------------------------------------

            if mode == "scm":

                if i == 0:

                    prompt = (
                        first_prompt(
                            text
                        )
                    )

                    previous_mode = (
                        "none"
                    )


                else:

                    prev = histories[
                        -1
                    ]


                    if (
                        prev[
                            "content_tokens_length"
                        ]
                        < PREVIOUS_RAW_THRESHOLD
                    ):

                        previous_content = (
                            prev[
                                "paragraph"
                            ]
                        )

                        previous_mode = (
                            "raw_previous_paragraph"
                        )


                    else:

                        previous_content = (
                            prev[
                                "summary"
                            ]
                        )

                        previous_mode = (
                            "previous_summary"
                        )


                    prompt = scm_prompt(
                        text,
                        previous_content,
                    )


            else:

                # Official book_summary.py:
                #
                # if scm is False:
                #     get_paragragh_prompt(text)
                #
                # This applies to every chunk,
                # including the first chunk.

                prompt = no_scm_prompt(
                    text
                )

                previous_mode = (
                    "no_scm"
                )


            key = (
                f"{book_slug}|"
                f"{mode}|"
                f"chunk={i}|"
                f"{sha256_text(prompt)}"
            )


            summary = cache.get(
                key
            )


            if summary is None:

                summary = (
                    generator
                    .generate(
                        prompt
                    )
                )


                cache.put(
                    key=key,
                    stage=(
                        "chunk_summary"
                    ),
                    prompt=prompt,
                    output=summary,
                    meta={
                        "chunk_index":
                            i,

                        "mode":
                            mode,

                        "previous_mode":
                            previous_mode,

                        "paragraph_tokens_p50k":
                            toklen(text),
                    },
                )


                cache_state = (
                    "MISS"
                )


            else:

                cache_state = (
                    "HIT"
                )


            row = {
                "chunk_index":
                    i,

                "paragraph_sha256":
                    sha256_text(
                        text
                    ),

                "paragraph_tokens_p50k":
                    toklen(text),

                "summary_tokens_p50k":
                    toklen(summary),

                "previous_mode":
                    previous_mode,

                "summary":
                    summary,
            }


            fout.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            fout.flush()


            histories.append({
                "paragraph":
                    text,

                "summary":
                    summary,

                "content_tokens_length":
                    toklen(text),

                "summary_tokens_length":
                    toklen(summary),
            })


            print(
                "[CHUNK] "
                f"{cache_state} "
                f"{i+1}/"
                f"{len(chunks)} "
                f"mode={mode} "
                "paragraph_p50k="
                f"{toklen(text)} "
                "summary_p50k="
                f"{toklen(summary)}",
                flush=True,
            )


    sub_summaries = [
        x[
            "summary"
        ]
        for x
        in histories
    ]


    # --------------------------------------------------------
    # Final hierarchical summary
    # --------------------------------------------------------

    if (
        final_file.exists()
        and final_file
        .read_text(
            encoding="utf-8"
        )
        .strip()
    ):

        final_summary = (
            final_file
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )


        print(
            "[FINAL] existing "
            "final_summary.txt "
            "-> reuse",
            flush=True,
        )


    else:

        final_summary = (
            divide_conquer_summary(
                content_lst=(
                    sub_summaries
                ),
                generator=generator,
                cache=cache,
                book_slug=book_slug,
                mode=mode,
                depth=1,
            )
        )


        final_file.write_text(
            final_summary
            + "\n",
            encoding="utf-8",
        )


    result = {
        "method":
            "SCM",

        "task":
            "official_book_summarization",

        "setting":
            "Qwen2.5-7B-Instruct compatibility",

        "official_repo":
            str(
                OFFICIAL_REPO
            ),

        "official_baseline_commit":
            OFFICIAL_COMMIT,

        "book_file":
            str(
                book_file
            ),

        "book_name":
            book_name,

        "mode":
            mode,

        "generation_backbone":
            str(
                MODEL_PATH
            ),

        "generation": {
            "do_sample":
                False,

            "max_new_tokens":
                OFFICIAL_API_MAX_NEW_TOKENS,

            "legacy_stop_strings_applied_post_decode":
                [
                    "\\n\\n\\n",
                    "###",
                ],
        },

        "token_budget_compatibility": {
            "tokenizer_model":
                TOKENIZER_MODEL,

            "book_split_max_tokens":
                BOOK_SPLIT_MAX_TOKENS,

            "previous_raw_threshold":
                PREVIOUS_RAW_THRESHOLD,

            "hierarchical_group_tokens":
                HIER_GROUP_TOKENS,

            "official_context_limit":
                OFFICIAL_CONTEXT_LIMIT,
        },

        "language":
            lang,

        "encoding":
            encoding,

        "num_chunks":
            len(chunks),

        "book_chunk_tokens_p50k_sum":
            sum(
                toklen(x)
                for x
                in chunks
            ),

        "chunk_summary_tokens_p50k_sum":
            sum(
                toklen(x)
                for x
                in sub_summaries
            ),

        "final_summary_tokens_p50k":
            toklen(
                final_summary
            ),

        "final_summary":
            final_summary,

        "compatibility_notes": [
            (
                "Original book data and "
                "SCM/no-SCM book algorithm "
                "are preserved."
            ),
            (
                "Official book prompts are "
                "imported from prompts/book.py."
            ),
            (
                "Legacy OpenAI generation is "
                "replaced by local "
                "Qwen2.5-7B-Instruct."
            ),
            (
                "text-davinci-003 tiktoken "
                "accounting is retained for "
                "splitter and SummaryBot budgets."
            ),
            (
                "Qwen generation uses "
                "deterministic greedy decoding "
                "for reproducibility."
            ),
        ],
    }


    result_file.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print(
        "[BOOK_DONE] "
        f"{book_name} "
        f"mode={mode} "
        f"chunks={len(chunks)} "
        "final_p50k="
        f"{toklen(final_summary)}",
        flush=True,
    )


    return result


# ============================================================
# Discover official 10 books
# ============================================================

def discover_all_books():

    books = []


    for subdir in (
        "ChineseBook",
        "EnglishBook",
    ):

        books.extend(
            sorted(
                (
                    OFFICIAL_REPO
                    / "data"
                    / "book"
                    / subdir
                )
                .glob(
                    "*.txt"
                )
            )
        )


    return books


# ============================================================
# Main
# ============================================================

def main():

    parser = (
        argparse
        .ArgumentParser()
    )


    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run all 10 official books"
        ),
    )


    parser.add_argument(
        "--book",
        type=str,
        default=None,
        help=(
            "Run one explicit "
            "book path"
        ),
    )


    parser.add_argument(
        "--mode",
        choices=[
            "scm",
            "no_scm",
            "both",
        ],
        default="both",
    )


    args = parser.parse_args()


    if not OFFICIAL_REPO.exists():

        raise FileNotFoundError(
            OFFICIAL_REPO
        )


    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            MODEL_PATH
        )


    if (
        args.all
        and args.book
    ):

        raise ValueError(
            "Use either --all "
            "or --book, not both."
        )


    if args.all:

        books = (
            discover_all_books()
        )


        if len(books) != 10:

            raise RuntimeError(
                "Expected 10 official "
                "books, found "
                f"{len(books)}:\n"
                f"{books}"
            )


    elif args.book:

        p = Path(
            args.book
        )


        if not p.is_absolute():

            p = (
                OFFICIAL_REPO
                / p
            )


        books = [
            p
        ]


    else:

        # Default smoke:
        # shortest official book.

        books = [
            OFFICIAL_REPO
            / "data"
            / "book"
            / "EnglishBook"
            / (
                "The_Old_Man_"
                "and_the_Sea.txt"
            )
        ]


    for p in books:

        if not p.exists():

            raise FileNotFoundError(
                p
            )


    if args.mode == "both":

        modes = [
            "scm",
            "no_scm",
        ]

    else:

        modes = [
            args.mode
        ]


    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    metadata = {
        "task":
            (
                "SCM official Book "
                "Summarization compatibility "
                "reproduction"
            ),

        "official_baseline_commit":
            OFFICIAL_COMMIT,

        "books":
            [
                str(x)
                for x
                in books
            ],

        "modes":
            modes,

        "model":
            str(
                MODEL_PATH
            ),

        "tokenizer_budget_model":
            TOKENIZER_MODEL,

        "started_at_unix":
            time.time(),
    }


    (
        OUT_ROOT
        / "last_run_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    generator = (
        QwenGenerator()
    )


    results = []


    try:

        for book in books:

            for mode in modes:

                results.append(
                    run_book(
                        book_file=book,
                        mode=mode,
                        generator=generator,
                    )
                )

    finally:

        generator.close()


    print()
    print(
        "=" * 80
    )

    print(
        "===== SCM BOOK "
        "COMPATIBILITY RUN "
        "COMPLETE ====="
    )


    for r in results:

        print(
            f"{r['book_name']} "
            f"| {r['mode']} "
            "| chunks="
            f"{r['num_chunks']} "
            "| final_p50k="
            f"{r['final_summary_tokens_p50k']}"
        )


    print(
        "OUTPUT_ROOT:",
        OUT_ROOT,
    )

    print(
        "SCM_BOOK_QWEN_OK"
    )


if __name__ == "__main__":
    main()
