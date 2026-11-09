#!/usr/bin/env python3

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path


REPO = Path('/root/autodl-tmp/paper_repro/SCM4LLMs')
MODEL = Path('/root/autodl-tmp/models/Qwen2.5-7B-Instruct')
OUT = Path('/root/autodl-tmp/paper_repro/adapters/scm_meeting_repro/output')

COMMIT = '26216044f5d4caccc0303d86667b491ead11d878'

MEETING_MAX_TOKENS = 2800
PREVIOUS_RAW_THRESHOLD = 500
FINAL_GROUP_TOKENS = 2300


def sha256_text(s):
    return hashlib.sha256(
        s.encode('utf-8')
    ).hexdigest()


def build_theme_introduction(dialogues):
    introduction = dialogues[0]['content']
    total = len(dialogues)

    if len(introduction) < 100:
        i = 1

        while (
            len(introduction) < 1800
            and i < total
        ):
            introduction += (
                f" {dialogues[i]['content']}"
            )
            i += 1

    return introduction[:1200]


def split_meeting(
    dialogues,
    encode,
    max_tokens=MEETING_MAX_TOKENS,
):
    meetings = [
        f'[{u["role"]}]: {u["content"]}'
        for u in dialogues
    ]

    meetings_str = '\n\n'.join(
        meetings
    )

    separator = '。'

    split_parts = []
    current_tokens = 0
    current_block = ''

    for sentence in meetings_str.split(
        separator
    ):
        count = len(
            encode(sentence)
        )

        if (
            current_tokens + count
            > max_tokens
        ):
            # Preserve official splitter behavior.
            split_parts.append(
                current_block
            )

            current_block = (
                sentence
                + separator
            )

            current_tokens = count

        else:
            current_tokens += count

            current_block += (
                sentence
                + separator
            )

    if current_block:
        split_parts.append(
            current_block
        )

    return split_parts


def previous_content(
    paragraph,
    summary,
    encode,
):
    if (
        len(encode(paragraph))
        < PREVIOUS_RAW_THRESHOLD
    ):
        return paragraph, 'raw'

    return summary, 'summary'


def group_strings(
    content_lst,
    summary_token_length_lst,
    group_tokens=FINAL_GROUP_TOKENS,
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
                [content_lst[i]]
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
        '\n\n'.join(
            group
        ).strip()
        for group in result
    ]


def apply_historical_stop(text):
    """
    Preserve the historical Davinci-style
    post-generation stop behavior used in
    our SCM compatibility reproduction.

    We explicitly record every stop hit,
    because Qwen may itself generate ###.
    """
    cut = len(text)
    hit = None

    for mark in (
        '\n\n\n',
        '###',
    ):
        p = text.find(mark)

        if (
            p >= 0
            and p < cut
        ):
            cut = p
            hit = mark

    return (
        text[:cut].strip(),
        hit,
    )


def load_json(path):
    with open(
        path,
        encoding='utf-8',
    ) as f:
        return json.load(f)


def safe_load_jsonl(
    path,
    key_field,
):
    """
    Tolerates a partially written final line
    after an unexpected shutdown.
    """
    out = {}

    if not path.exists():
        return out

    with open(
        path,
        encoding='utf-8',
    ) as f:
        for lineno, line in enumerate(
            f,
            1,
        ):
            if not line.strip():
                continue

            try:
                x = json.loads(
                    line
                )

            except json.JSONDecodeError:
                print(
                    '[WARN] skip malformed '
                    f'JSONL line: '
                    f'{path}:{lineno}',
                    flush=True,
                )
                continue

            if key_field in x:
                out[
                    x[key_field]
                ] = x

    return out


def append_jsonl(
    path,
    obj,
):
    """
    Open/write/fsync every record.
    This is intentionally conservative for
    AutoDL interruption / shutdown recovery.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        'a',
        encoding='utf-8',
    ) as f:
        f.write(
            json.dumps(
                obj,
                ensure_ascii=False,
            )
            + '\n'
        )

        f.flush()
        os.fsync(
            f.fileno()
        )


class GenCache:

    def __init__(
        self,
        path,
    ):
        self.path = path

        self.data = (
            safe_load_jsonl(
                path,
                'key',
            )
        )

    def get(
        self,
        prompt,
    ):
        return self.data.get(
            sha256_text(
                prompt
            )
        )

    def put(
        self,
        prompt,
        kind,
        raw_output,
        output,
        stop_hit,
    ):
        k = sha256_text(
            prompt
        )

        if k in self.data:
            return self.data[k]

        rec = {
            'key': k,
            'kind': kind,
            'raw_output':
                raw_output,
            'output':
                output,
            'stop_hit':
                stop_hit,
        }

        append_jsonl(
            self.path,
            rec,
        )

        self.data[k] = rec

        return rec


class Qwen:

    def __init__(
        self,
        cache_path,
    ):
        import torch
        import tiktoken

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        self.torch = torch

        self.cache = GenCache(
            cache_path
        )

        # Historical text-davinci-003 completion limits
        # are defined in this tokenizer space, not in
        # the Qwen tokenizer space.
        self.compat_enc = (
            tiktoken.encoding_for_model(
                'text-davinci-003'
            )
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL,
                local_files_only=True,
                trust_remote_code=True,
            )
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                max_memory={
                    0: '18GiB',
                    'cpu': '24GiB',
                },
                local_files_only=True,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).eval()
        )

    def _compat_output(
        self,
        prompt,
        raw,
    ):
        # Official call_text_davinci_003:
        #
        # prompt_tokens = p50k(prompt)
        # assert prompt_tokens < 3999
        # max_tokens = min(
        #     4000 - prompt_tokens,
        #     1024,
        # )
        #
        # The Qwen tokenizer is different, so
        # max_new_tokens=1024 alone does NOT
        # reproduce this constraint.

        prompt_tokens = len(
            self.compat_enc.encode(
                prompt
            )
        )

        if prompt_tokens >= 3999:
            raise RuntimeError(
                'OFFICIAL_PROMPT_LIMIT '
                f'p50k_prompt_tokens='
                f'{prompt_tokens}'
            )

        compat_max_tokens = min(
            4000 - prompt_tokens,
            1024,
        )

        # Historical stop strings.
        out, stop_hit = (
            apply_historical_stop(
                raw
            )
        )

        ids = self.compat_enc.encode(
            out
        )

        compat_truncated = (
            len(ids)
            > compat_max_tokens
        )

        if compat_truncated:
            out = (
                self.compat_enc.decode(
                    ids[
                        :compat_max_tokens
                    ]
                )
                .strip()
            )

        output_p50k_tokens = len(
            self.compat_enc.encode(
                out
            )
        )

        return (
            out,
            stop_hit,
            compat_truncated,
            compat_max_tokens,
            output_p50k_tokens,
        )


    def ask(
        self,
        prompt,
        kind,
    ):
        cached = self.cache.get(
            prompt
        )

        if cached:
            raw = cached.get(
                'raw_output',
                cached.get(
                    'output',
                    '',
                ),
            )

            (
                out,
                stop_hit,
                compat_truncated,
                compat_max_tokens,
                output_p50k_tokens,
            ) = self._compat_output(
                prompt,
                raw,
            )

            meta = dict(
                cached
            )

            meta.update({
                'output':
                    out,

                'stop_hit':
                    stop_hit,

                'compat_truncated':
                    compat_truncated,

                'compat_max_tokens':
                    compat_max_tokens,

                'output_p50k_tokens':
                    output_p50k_tokens,
            })

            print(
                '[GEN] HIT '
                f'{kind} '
                f'stop={stop_hit} '
                f'p50k='
                f'{output_p50k_tokens}/'
                f'{compat_max_tokens} '
                f'truncated='
                f'{compat_truncated}',
                flush=True,
            )

            return (
                out,
                meta,
            )

        text = (
            self.tokenizer
            .apply_chat_template(
                [
                    {
                        'role': 'user',
                        'content': prompt,
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        z = self.tokenizer(
            text,
            return_tensors='pt',
        )

        dev = next(
            self.model.parameters()
        ).device

        z = {
            k: v.to(dev)
            for k, v
            in z.items()
        }

        n = (
            z['input_ids']
            .shape[1]
        )

        with (
            self.torch
            .inference_mode()
        ):
            y = self.model.generate(
                **z,
                max_new_tokens=1024,
                do_sample=False,
                use_cache=True,
                pad_token_id=(
                    self.tokenizer
                    .eos_token_id
                ),
            )[0, n:]

        raw = (
            self.tokenizer.decode(
                y,
                skip_special_tokens=True,
            )
            .strip()
        )

        (
            out,
            stop_hit,
            compat_truncated,
            compat_max_tokens,
            output_p50k_tokens,
        ) = self._compat_output(
            prompt,
            raw,
        )

        rec0 = self.cache.put(
            prompt,
            kind,
            raw,
            out,
            stop_hit,
        )

        rec = dict(
            rec0
        )

        rec.update({
            'output':
                out,

            'stop_hit':
                stop_hit,

            'compat_truncated':
                compat_truncated,

            'compat_max_tokens':
                compat_max_tokens,

            'output_p50k_tokens':
                output_p50k_tokens,
        })

        print(
            '[GEN] MISS '
            f'{kind} '
            f'out_chars={len(out)} '
            f'stop={stop_hit} '
            f'p50k='
            f'{output_p50k_tokens}/'
            f'{compat_max_tokens} '
            f'truncated='
            f'{compat_truncated}',
            flush=True,
        )

        return out, rec


def free_gpu():
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception:
        pass


def load_official_prompts():
    import sys

    sys.path.insert(
        0,
        str(REPO),
    )

    from prompts.meeting import (
        zh_theme_prompt,
        zh_start_prompt,
        zh_memory_prompt,
        zh_hierarchical_summarization_prompt,
    )

    return {
        'theme':
            zh_theme_prompt,

        'start':
            zh_start_prompt,

        'memory':
            zh_memory_prompt,

        'hier':
            zh_hierarchical_summarization_prompt,
    }


def recursive_final_summary(
    qwen,
    summaries,
    encode,
    hier_prompt,
    depth=1,
    trace=None,
):
    if trace is None:
        trace = []

    lens = [
        len(encode(x))
        for x in summaries
    ]

    groups = group_strings(
        summaries,
        lens,
        FINAL_GROUP_TOKENS,
    )

    if len(groups) == 1:
        single_summary_max_tokens = 1500
    else:
        single_summary_max_tokens = 1000

    group_summaries = []

    level = {
        'depth':
            depth,

        'input_count':
            len(summaries),

        'groups':
            len(groups),

        'input_tokens':
            sum(lens),

        'outputs':
            [],
    }

    for i, text in enumerate(
        groups
    ):
        token_count = len(
            encode(text)
        )

        current_tokens = max(
            token_count,
            FINAL_GROUP_TOKENS,
        )

        target = min(
            4000
            - current_tokens
            - 400,
            single_summary_max_tokens,
        )

        prompt = (
            hier_prompt.format(
                paragraph_summaries=(
                    text.strip()
                ),
                max_tokens=target,
            )
        )

        out, meta = qwen.ask(
            prompt,
            f'final_depth_{depth}',
        )

        group_summaries.append(
            out
        )

        level[
            'outputs'
        ].append({
            'group':
                i + 1,

            'input_tokens':
                token_count,

            'target':
                target,

            'output_tokens':
                len(encode(out)),

            'stop_hit':
                meta.get(
                    'stop_hit'
                ),
        })

    trace.append(
        level
    )

    if (
        len(group_summaries)
        > 1
    ):
        new_total = sum(
            len(encode(x))
            for x
            in group_summaries
        )

        # Do not silently modify the
        # official recursive algorithm.
        # Only stop if Qwen creates an
        # actual non-shrinking loop.
        if (
            len(group_summaries)
            >= len(summaries)
            and
            new_total
            >= sum(lens)
        ):
            raise RuntimeError(
                'OFFICIAL_NONTERMINATION_RISK '
                f'depth={depth} '
                f'input_count='
                f'{len(summaries)} '
                f'output_count='
                f'{len(group_summaries)} '
                f'input_tokens='
                f'{sum(lens)} '
                f'output_tokens='
                f'{new_total}'
            )

        return (
            recursive_final_summary(
                qwen,
                group_summaries,
                encode,
                hier_prompt,
                depth + 1,
                trace,
            )
        )

    return (
        group_summaries[0],
        trace,
    )


def run_one(
    meeting,
    mode,
    qwen,
    encode,
    prompts,
):
    mid = meeting[
        'meeting_id'
    ]

    dialogues = meeting[
        'dialogues'
    ]

    # Official theme extraction.
    intro = (
        build_theme_introduction(
            dialogues
        )
    )

    theme_prompt = (
        prompts[
            'theme'
        ].format(
            input=intro
        )
    )

    theme, theme_meta = (
        qwen.ask(
            theme_prompt,
            'theme',
        )
    )

    theme = (
        theme.strip()
        .rstrip('。')
        .rstrip('.')
    )

    # Official 2800-token splitter.
    parts = split_meeting(
        dialogues,
        encode,
        MEETING_MAX_TOKENS,
    )

    summaries = []
    chunk_meta = []

    for i, text in enumerate(
        parts
    ):
        if mode == 'scm':

            if i == 0:
                prompt = (
                    prompts[
                        'start'
                    ].format(
                        theme=theme,
                        input=text,
                    )
                )

                prev_mode = None

            else:
                pre, prev_mode = (
                    previous_content(
                        parts[i - 1],
                        summaries[i - 1],
                        encode,
                    )
                )

                prompt = (
                    prompts[
                        'memory'
                    ].format(
                        theme=theme,
                        previous_content=pre,
                        current_text=text,
                    )
                )

        else:
            # Official no-SCM behavior:
            # summarize each paragraph
            # independently.
            prompt = (
                prompts[
                    'start'
                ].format(
                    theme=theme,
                    input=text,
                )
            )

            prev_mode = None

        summary, meta = (
            qwen.ask(
                prompt,
                f'{mode}_chunk',
            )
        )

        summaries.append(
            summary
        )

        chunk_meta.append({
            'chunk_index':
                i + 1,

            'chunk_tokens':
                len(encode(text)),

            'summary_tokens':
                len(encode(summary)),

            'previous_mode':
                prev_mode,

            'stop_hit':
                meta.get(
                    'stop_hit'
                ),
        })

        print(
            '[MEETING] '
            f'{mid} '
            f'mode={mode} '
            f'chunk='
            f'{i+1}/{len(parts)} '
            f'prev={prev_mode}',
            flush=True,
        )

    final_summary, final_trace = (
        recursive_final_summary(
            qwen,
            summaries,
            encode,
            prompts['hier'],
        )
    )

    return {
        'key':
            f'{mid}|{mode}',

        'meeting_id':
            mid,

        'mode':
            mode,

        'official_commit':
            COMMIT,

        'compatibility_setting':
            True,

        'backbone':
            str(MODEL),

        'theme':
            theme,

        'source_token_count_reported':
            meeting.get(
                'token_count'
            ),

        'source_turn_count_reported':
            meeting.get(
                'turn_count'
            ),

        'chunk_count':
            len(parts),

        'chunk_meta':
            chunk_meta,

        'chunk_summaries':
            summaries,

        'final_trace':
            final_trace,

        'final_summary':
            final_summary,

        'gold_summary':
            meeting.get(
                'summary'
            ),

        'theme_stop_hit':
            theme_meta.get(
                'stop_hit'
            ),
    }


def prepare_blind(
    results,
    official_ids,
):
    by = {
        (
            x['meeting_id'],
            x['mode'],
        ): x

        for x in (
            results.values()
        )
    }

    blind = []
    key = []

    for idx, mid in enumerate(
        official_ids,
        1,
    ):
        if (
            (mid, 'scm')
            not in by
            or
            (mid, 'no_scm')
            not in by
        ):
            continue

        scm = by[
            (mid, 'scm')
        ]

        noscm = by[
            (mid, 'no_scm')
        ]

        # Deterministic A/B blinding.
        scm_is_a = (
            int(
                sha256_text(mid)[0],
                16,
            )
            % 2
            == 0
        )

        if scm_is_a:
            A = scm[
                'final_summary'
            ]

            B = noscm[
                'final_summary'
            ]

            am = 'scm'
            bm = 'no_scm'

        else:
            A = noscm[
                'final_summary'
            ]

            B = scm[
                'final_summary'
            ]

            am = 'no_scm'
            bm = 'scm'

        pair_id = (
            f'meeting{idx:02d}'
        )

        blind.append({
            'pair_id':
                pair_id,

            'meeting_id':
                mid,

            # Official repository's
            # meeting markdown includes
            # the reference summary.
            'reference_summary':
                scm['gold_summary'],

            'A':
                A,

            'B':
                B,
        })

        key.append({
            'pair_id':
                pair_id,

            'meeting_id':
                mid,

            'A_mode':
                am,

            'B_mode':
                bm,
        })

    return blind, key


def inspect_only(
    data,
    official_ids,
    encode,
):
    records = {
        x['meeting_id']: x
        for x in data
    }

    missing = [
        x
        for x in official_ids
        if x not in records
    ]

    extra = [
        x
        for x in records
        if x not in official_ids
    ]

    print(
        '===== SCM MEETING '
        'INSPECT ====='
    )

    print(
        'data_records:',
        len(data),
    )

    print(
        'official_ids:',
        len(official_ids),
    )

    print(
        'missing_official_ids:',
        len(missing),
    )

    print(
        'extra_data_ids:',
        len(extra),
    )

    vals = []
    utterances = 0

    for mid in official_ids:
        x = records[mid]

        texts = [
            f'[{u["role"]}]: '
            f'{u["content"]}'
            for u in x[
                'dialogues'
            ]
        ]

        n = len(
            encode(
                '\n\n'.join(
                    texts
                )
            )
        )

        vals.append(n)

        utterances += len(
            x['dialogues']
        )

        print(
            f'{mid} | '
            f'utterances='
            f'{len(x["dialogues"])} '
            f'| p50k_tokens={n}'
        )

    print(
        'meetings:',
        len(official_ids),
    )

    print(
        'utterances_total:',
        utterances,
    )

    print(
        'p50k_tokens_min:',
        min(vals),
    )

    print(
        'p50k_tokens_max:',
        max(vals),
    )

    print(
        'p50k_tokens_total:',
        sum(vals),
    )

    print(
        'SCM_MEETING_INSPECT_OK'
    )


def main():
    import tiktoken

    ap = argparse.ArgumentParser()

    ap.add_argument(
        '--stage',
        choices=[
            'inspect',
            'all',
        ],
        default='inspect',
    )

    ap.add_argument(
        '--mode',
        choices=[
            'both',
            'scm',
            'no_scm',
        ],
        default='both',
    )

    args = ap.parse_args()

    data_path = (
        REPO
        / 'data/meeting/vc_summary.json'
    )

    ids_path = (
        REPO
        / 'data/meeting/meeting_ids.txt'
    )

    data = load_json(
        data_path
    )

    official_ids = (
        ids_path
        .read_text(
            encoding='utf-8'
        )
        .split()
    )

    records = {
        x['meeting_id']: x
        for x in data
    }

    enc = (
        tiktoken
        .encoding_for_model(
            'text-davinci-003'
        )
    )

    encode = enc.encode

    missing = [
        x
        for x in official_ids
        if x not in records
    ]

    if missing:
        raise RuntimeError(
            'missing official '
            f'meeting ids: {missing}'
        )

    if args.stage == 'inspect':
        inspect_only(
            data,
            official_ids,
            encode,
        )
        return

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        OUT
        / 'meeting_generation_cache_qwen7b.jsonl'
    )

    results_path = (
        OUT
        / 'meeting_summaries_qwen7b.jsonl'
    )

    metrics_path = (
        OUT
        / 'meeting_generation_metrics.json'
    )

    blind_path = (
        OUT
        / 'meeting_pairwise_blind.json'
    )

    key_path = (
        OUT
        / 'meeting_pairwise_key.json'
    )

    results = (
        safe_load_jsonl(
            results_path,
            'key',
        )
    )

    print(
        f'REPO={REPO}'
    )

    print(
        f'COMMIT={COMMIT}'
    )

    print(
        f'MODEL={MODEL}'
    )

    print(
        f'OUTPUT={OUT}'
    )

    print(
        f'CACHED_COMPLETED_PAIRS='
        f'{len(results)}'
    )

    qwen = Qwen(
        cache_path
    )

    prompts = (
        load_official_prompts()
    )

    if args.mode == 'both':
        modes = [
            'scm',
            'no_scm',
        ]
    else:
        modes = [
            args.mode
        ]

    for mid in official_ids:

        for mode in modes:

            k = (
                f'{mid}|{mode}'
            )

            if k in results:
                print(
                    f'[PAIR] HIT {k}',
                    flush=True,
                )
                continue

            print(
                f'\n[PAIR] START {k}',
                flush=True,
            )

            rec = run_one(
                records[mid],
                mode,
                qwen,
                encode,
                prompts,
            )

            append_jsonl(
                results_path,
                rec,
            )

            results[k] = rec

            print(
                f'[PAIR] DONE {k} '
                f'chunks='
                f'{rec["chunk_count"]}',
                flush=True,
            )

    # Reload from disk so mode-specific
    # runs can later combine cleanly.
    results = (
        safe_load_jsonl(
            results_path,
            'key',
        )
    )

    all_recs = list(
        results.values()
    )

    blind, key = prepare_blind(
        results,
        official_ids,
    )

    stop_chunk = sum(
        1
        for r in all_recs
        for x in r.get(
            'chunk_meta',
            [],
        )
        if x.get(
            'stop_hit'
        )
    )

    stop_final = sum(
        1
        for r in all_recs
        for lev in r.get(
            'final_trace',
            [],
        )
        for x in lev.get(
            'outputs',
            [],
        )
        if x.get(
            'stop_hit'
        )
    )

    stop_theme = sum(
        1
        for r in all_recs
        if r.get(
            'theme_stop_hit'
        )
    )

    metrics = {
        'official_commit':
            COMMIT,

        'backbone':
            str(MODEL),

        'compatibility_setting':
            True,

        'historical_generation_replaced_with_qwen':
            True,

        'historical_stop_postprocessing': [
            '\\n\\n\\n',
            '###',
        ],

        'davinci_compatible_output_token_cap_p50k':
            1024,

        'meeting_max_tokens':
            MEETING_MAX_TOKENS,

        'previous_raw_threshold':
            PREVIOUS_RAW_THRESHOLD,

        'final_group_tokens':
            FINAL_GROUP_TOKENS,

        'official_meetings':
            len(official_ids),

        'completed_pairs':
            len(all_recs),

        'completed_scm':
            sum(
                r.get('mode')
                == 'scm'
                for r in all_recs
            ),

        'completed_no_scm':
            sum(
                r.get('mode')
                == 'no_scm'
                for r in all_recs
            ),

        'meetings_with_both':
            len(blind),

        'stop_hits': {
            'theme_records':
                stop_theme,

            'chunk_summaries':
                stop_chunk,

            'final_hierarchical':
                stop_final,
        },

        'note': (
            'Generation only. '
            'Final quality must be '
            'evaluated separately. '
            'This is a Qwen compatibility '
            'reproduction, not strict '
            'historical OpenAI numeric replay.'
        ),
    }

    with open(
        metrics_path,
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        blind_path,
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(
            blind,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        key_path,
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(
            key,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        '\n===== SCM MEETING '
        'COMPATIBILITY GENERATION ====='
    )

    print(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        )
    )

    if (
        len(blind)
        == len(official_ids)
    ):
        print(
            'SCM_MEETING_QWEN_OK'
        )

    else:
        print(
            'SCM_MEETING_PARTIAL'
        )

    del qwen
    free_gpu()


if __name__ == '__main__':
    main()
