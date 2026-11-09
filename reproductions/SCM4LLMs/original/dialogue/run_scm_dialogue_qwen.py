#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import tiktoken
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

REPO = Path('/root/autodl-tmp/paper_repro/SCM4LLMs')
MODEL = Path('/root/autodl-tmp/models/Qwen2.5-7B-Instruct')
CONTRIEVER = Path('/root/autodl-tmp/paper_repro/MemGAS/facebook/contriever')
OUT = Path('/root/autodl-tmp/paper_repro/adapters/scm_dialogue_repro/output')

COMMIT = '26216044f5d4caccc0303d86667b491ead11d878'

TOPK = 6
MAX_HIST = 2500
MAX_PREV = 500
SUM_TH = 300

P50K = tiktoken.encoding_for_model('text-davinci-003')

sys.path.insert(0, str(REPO))

from prompts.dialogue import (
    en_turn_summarization_prompt,
    zh_turn_summarization_prompt,
    en_history_agent_prompt,
    zh_history_agent_prompt,
    en_no_history_agent_prompt,
    zh_no_history_agent_prompt,
    judge_answerable_prompt,
)


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def tok(s):
    return len(P50K.encode(s))


def loadj(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def appendj(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a', encoding='utf-8') as f:
        f.write(json.dumps(x, ensure_ascii=False) + '\n')


def loadmap(p, key):
    d = {}
    if p.exists():
        with open(p, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    x = json.loads(line)
                    d[x[key]] = x
    return d


def datasets(lang):
    ds = {
        x['id']: x['dialogue']
        for x in loadj(REPO / 'data' / 'dialogue' / f'{lang}.json')
    }

    qs = loadj(
        REPO
        / 'annotation_data'
        / 'dialogue'
        / f'dialogue_{lang}_questions.json'
    )

    miss = sorted({
        q['id']
        for q in qs
        if q['id'] not in ds
    })

    if miss:
        raise RuntimeError(f'missing dialogue ids: {miss}')

    return ds, qs


def turn_rows(did, seq, lang):
    if len(seq) % 2:
        raise RuntimeError(f'odd dialogue length: {did}')

    out = []

    for j, i in enumerate(range(0, len(seq), 2), 1):
        u = seq[i]
        a = seq[i + 1]

        if lang == 'en':
            plain = f'User: {u}\n\nAssistant: {a}'
            raw = f'[Turn {j}]\n\nUser: {u}\n\nAssistant: {a}'
        else:
            plain = f'用户：{u}\n\n助手：{a}'
            raw = f'[第{j}轮]\n\n用户：{u}\n\n助手：{a}'

        out.append({
            'did': did,
            'idx': j,
            'plain': plain,
            'raw': raw,
            'key': h(raw),
        })

    return out


class GenCache:
    def __init__(self, p):
        self.p = p
        self.d = loadmap(p, 'key')

    def get(self, prompt):
        return self.d.get(h(prompt))

    def put(self, prompt, output, kind):
        k = h(prompt)

        if k not in self.d:
            x = {
                'key': k,
                'kind': kind,
                'output': output,
            }
            appendj(self.p, x)
            self.d[k] = x


class Qwen:
    def __init__(self, cache):
        self.c = cache

        self.t = AutoTokenizer.from_pretrained(
            MODEL,
            local_files_only=True,
            trust_remote_code=True,
        )

        self.m = AutoModelForCausalLM.from_pretrained(
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

    @torch.inference_mode()
    def ask(self, prompt, kind):
        x = self.c.get(prompt)

        if x:
            print(
                f'[GEN] HIT {kind} out={tok(x["output"])}',
                flush=True,
            )
            return x['output']

        text = self.t.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        z = self.t(text, return_tensors='pt')
        dev = next(self.m.parameters()).device
        z = {k: v.to(dev) for k, v in z.items()}
        n = z['input_ids'].shape[1]

        y = self.m.generate(
            **z,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.t.eos_token_id,
        )[0, n:]

        s = self.t.decode(
            y,
            skip_special_tokens=True,
        ).strip()

        cut = len(s)
        for mark in ('\n\n\n', '###'):
            p = s.find(mark)
            if p >= 0:
                cut = min(cut, p)

        s = s[:cut].strip()

        self.c.put(
            prompt,
            s,
            kind,
        )

        print(
            f'[GEN] MISS {kind} out={tok(s)}',
            flush=True,
        )

        return s


class Contriever:
    def __init__(self):
        self.t = AutoTokenizer.from_pretrained(
            CONTRIEVER,
            local_files_only=True,
        )

        self.m = AutoModel.from_pretrained(
            CONTRIEVER,
            local_files_only=True,
        ).eval()

        self.dev = torch.device(
            'cuda'
            if torch.cuda.is_available()
            else 'cpu'
        )

        self.m.to(self.dev)

    @torch.inference_mode()
    def enc(self, texts, bs=32):
        vs = []

        for i in range(0, len(texts), bs):
            b = texts[i:i + bs]

            z = self.t(
                b,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt',
            )

            z = {
                k: v.to(self.dev)
                for k, v in z.items()
            }

            o = self.m(
                **z
            ).last_hidden_state

            m = (
                z['attention_mask']
                .unsqueeze(-1)
                .to(o.dtype)
            )

            v = (
                (o * m).sum(1)
                / m.sum(1).clamp(min=1)
            )

            vs.append(
                v.float().cpu()
            )

            print(
                f'[EMBED] '
                f'{min(i + len(b), len(texts))}'
                f'/{len(texts)}',
                flush=True,
            )

        return torch.cat(vs)


def free_gpu():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_summaries(
    langs,
    qwen,
    path,
):
    cache = loadmap(
        path,
        'key',
    )

    for lang in langs:
        ds, qs = datasets(lang)

        ids = sorted({
            q['id']
            for q in qs
        })

        for di, did in enumerate(
            ids,
            1,
        ):
            rows = turn_rows(
                did,
                ds[did],
                lang,
            )

            print(
                f'[SUMMARY] {lang} '
                f'{di}/{len(ids)} '
                f'{did} '
                f'turns={len(rows)}',
                flush=True,
            )

            for r in rows:
                if r['key'] in cache:
                    continue

                tmpl = (
                    en_turn_summarization_prompt
                    if lang == 'en'
                    else zh_turn_summarization_prompt
                )

                p = tmpl.format(
                    input=r['plain']
                )

                if tok(p) > SUM_TH:
                    summary = qwen.ask(
                        p,
                        'summary',
                    )
                else:
                    summary = r['raw']

                x = {
                    'key': r['key'],
                    'did': did,
                    'idx': r['idx'],
                    'lang': lang,
                    'summary': summary,
                    'content_tokens': tok(
                        r['raw']
                    ),
                    'summary_tokens': tok(
                        summary
                    ),
                }

                appendj(
                    path,
                    x,
                )

                cache[r['key']] = x


def prepare_embeddings(
    langs,
    path,
):
    if path.exists():
        c = torch.load(
            path,
            map_location='cpu',
        )
    else:
        c = {
            'v': {},
        }

    keys = []
    texts = []

    for lang in langs:
        ds, qs = datasets(lang)

        ids = sorted({
            q['id']
            for q in qs
        })

        for did in ids:
            for r in turn_rows(
                did,
                ds[did],
                lang,
            ):
                k = 't:' + r['key']

                if k not in c['v']:
                    keys.append(k)
                    texts.append(
                        r['raw']
                    )

        for q in qs:
            k = (
                'q:'
                + h(q['question'])
            )

            if k not in c['v']:
                keys.append(k)
                texts.append(
                    q['question']
                )

    print(
        f'[EMBED] '
        f'missing={len(keys)} '
        f'existing={len(c["v"])}',
        flush=True,
    )

    if keys:
        enc = Contriever()

        vecs = enc.enc(
            texts
        )

        for k, v in zip(
            keys,
            vecs,
        ):
            c['v'][k] = v

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            c,
            path,
        )

        del enc
        free_gpu()

    print(
        f'[EMBED] '
        f'total={len(c["v"])} '
        f'saved={path}',
        flush=True,
    )


def cos(a, b):
    a = a.float()
    b = b.float()

    d = (
        a.norm()
        * b.norm()
    )

    if d.item() == 0:
        return 0.0

    return float(
        torch.dot(a, b)
        / d
    )


def hydrate(
    did,
    seq,
    lang,
    sums,
    emb,
):
    out = []

    for r in turn_rows(
        did,
        seq,
        lang,
    ):
        s = sums[
            r['key']
        ]

        r.update({
            'summary':
                s['summary'],

            'content_tokens':
                s['content_tokens'],

            'summary_tokens':
                s['summary_tokens'],

            'embedding':
                emb['v'][
                    't:' + r['key']
                ],
        })

        out.append(r)

    return out


def prev(turns):
    t = turns[-1]

    if (
        t['content_tokens']
        < MAX_PREV
    ):
        return (
            t['raw'],
            'raw',
        )

    return (
        t['summary'],
        'summary',
    )


def total(
    lengths,
    turns,
):
    t = turns[-1]

    if (
        t['content_tokens']
        < MAX_PREV
    ):
        p = t[
            'content_tokens'
        ]
    else:
        p = t[
            'summary_tokens'
        ]

    return (
        sum(lengths)
        + p
    )


def binary(
    qwen,
    prompt,
):
    s = qwen.ask(
        prompt,
        'controller',
    )

    return not (
        '(B)否' in s
        or '(B)' in s
    )


def choose(
    qwen,
    query,
    t,
):
    raw_prompt = (
        judge_answerable_prompt.format(
            content=t['raw'],
            query=query,
        )
    )

    if not binary(
        qwen,
        raw_prompt,
    ):
        return 'drop'

    summary_prompt = (
        judge_answerable_prompt.format(
            content=t['summary'],
            query=query,
        )
    )

    if binary(
        qwen,
        summary_prompt,
    ):
        return 'summary'

    return 'raw'


def retrieve(
    qwen,
    query,
    turns,
    qvec,
):
    cand = turns[:-1]

    if not cand:
        return '', {
            'controller_used': False,
            'topk_indices': [],
            'topk_scores': [],
            'flags': {},
            'rounds': 0,
            'selected_tokens': 0,
        }

    sims = np.array(
        [
            cos(
                qvec,
                t['embedding'],
            )
            for t in cand
        ],
        dtype=np.float32,
    )

    idx = np.argsort(
        sims
    )[
        -min(
            TOPK,
            len(cand),
        ):
    ].tolist()

    vals = [
        float(
            sims[i]
        )
        for i in idx
    ]

    lens = [
        turns[i][
            'content_tokens'
        ]
        for i in idx
    ]

    flags = {
        i: 'raw'
        for i in idx
    }

    used = (
        total(
            lens,
            turns,
        )
        > MAX_HIST
    )

    first = True
    rounds = 0

    while (
        total(
            lens,
            turns,
        )
        > MAX_HIST
    ):
        rounds += 1

        before = (
            tuple(lens),
            tuple(
                flags[i]
                for i in idx
            ),
        )

        for pos, i in enumerate(
            idx
        ):
            if first:
                if pos < len(idx) - 1:
                    ch = choose(
                        qwen,
                        query,
                        turns[i],
                    )
                else:
                    ch = 'raw'

            else:
                ch = (
                    'summary'
                    if flags[i] == 'raw'
                    else flags[i]
                )

            flags[i] = ch

            if ch == 'drop':
                lens[pos] = 0

            elif ch == 'summary':
                lens[pos] = (
                    turns[i][
                        'summary_tokens'
                    ]
                )

            if (
                not first
                and total(
                    lens,
                    turns,
                )
                <= MAX_HIST
            ):
                break

        if first:
            first = False

        after = (
            tuple(lens),
            tuple(
                flags[i]
                for i in idx
            ),
        )

        if (
            after == before
            and total(
                lens,
                turns,
            )
            > MAX_HIST
        ):
            raise RuntimeError(
                'OFFICIAL_NONTERMINATION_RISK '
                f'total={total(lens, turns)}'
            )

    pieces = []

    for i, _ in sorted(
        zip(
            idx,
            vals,
        ),
        key=lambda x: x[0],
    ):
        if flags[i] == 'drop':
            continue

        if flags[i] == 'summary':
            pieces.append(
                turns[i][
                    'summary'
                ]
            )
        else:
            pieces.append(
                turns[i][
                    'raw'
                ]
            )

    text = ''.join(
        x + '\n\n'
        for x in pieces
    )

    meta = {
        'controller_used':
            used,

        'topk_indices':
            idx,

        'topk_scores':
            vals,

        'flags': {
            str(k): v
            for k, v
            in flags.items()
        },

        'rounds':
            rounds,

        'selected_tokens':
            total(
                lens,
                turns,
            ),
    }

    return (
        text,
        meta,
    )


def run_qa(
    langs,
    qwen,
    sum_path,
    emb_path,
    res_path,
):
    sums = loadmap(
        sum_path,
        'key',
    )

    emb = torch.load(
        emb_path,
        map_location='cpu',
    )

    done = loadmap(
        res_path,
        'question_id',
    )

    for lang in langs:
        ds, qs = datasets(lang)

        by = defaultdict(list)

        for q in qs:
            by[
                q['id']
            ].append(q)

        for di, did in enumerate(
            sorted(by),
            1,
        ):
            turns = hydrate(
                did,
                ds[did],
                lang,
                sums,
                emb,
            )

            pretxt, premode = prev(
                turns
            )

            print(
                f'[QA] {lang} '
                f'{di}/{len(by)} '
                f'{did} '
                f'turns={len(turns)} '
                f'prev={premode}',
                flush=True,
            )

            for q in by[did]:
                qid = q[
                    'question_id'
                ]

                if qid in done:
                    print(
                        f'[QA] HIT {qid}',
                        flush=True,
                    )
                    continue

                qvec = emb[
                    'v'
                ][
                    'q:'
                    + h(
                        q['question']
                    )
                ]

                hist, meta = retrieve(
                    qwen,
                    q['question'],
                    turns,
                    qvec,
                )

                if hist:
                    tmpl = (
                        en_history_agent_prompt
                        if lang == 'en'
                        else zh_history_agent_prompt
                    )

                    p = tmpl.format(
                        history_turn_text=hist,
                        pre_turn_text=pretxt,
                        input=q['question'],
                    )

                else:
                    tmpl = (
                        en_no_history_agent_prompt
                        if lang == 'en'
                        else zh_no_history_agent_prompt
                    )

                    p = tmpl.format(
                        pre_turn_text=pretxt,
                        input=q['question'],
                    )

                ans = qwen.ask(
                    p,
                    'qa',
                )

                x = {
                    'question_id':
                        qid,

                    'dialogue_id':
                        did,

                    'lang':
                        lang,

                    'mode':
                        q.get(
                            'mode'
                        ),

                    'question':
                        q['question'],

                    'gold_answer':
                        q.get(
                            'answer'
                        ),

                    'qwen_scm_answer':
                        ans,

                    'official_scm_davinci_answer':
                        q.get(
                            'scm_davinci_answer'
                        ),

                    'official_scm_turbo_answer':
                        q.get(
                            'scm_turbo_answer'
                        ),

                    'previous_turn_mode':
                        premode,

                    'retrieval':
                        meta,

                    'prompt_p50k':
                        tok(p),
                }

                appendj(
                    res_path,
                    x,
                )

                done[qid] = x

                print(
                    f'[QA] MISS {qid} '
                    f'controller={meta["controller_used"]} '
                    f'prompt={tok(p)} '
                    f'out={tok(ans)}',
                    flush=True,
                )

    rows = list(
        done.values()
    )

    m = {
        'official_commit':
            COMMIT,

        'backbone':
            str(MODEL),

        'embedding_model':
            str(CONTRIEVER),

        'compatibility_setting':
            True,

        'top_k':
            TOPK,

        'max_history_tokens':
            MAX_HIST,

        'max_previous_turn_tokens':
            MAX_PREV,

        'questions_total':
            len(rows),

        'questions_by_lang': {
            k: sum(
                r['lang'] == k
                for r in rows
            )
            for k in (
                'zh',
                'en',
            )
        },

        'questions_by_mode': {
            k: sum(
                r.get('mode') == k
                for r in rows
            )
            for k in (
                'single-turn',
                'multi-turn',
            )
        },

        'controller_used_total':
            sum(
                bool(
                    r['retrieval'][
                        'controller_used'
                    ]
                )
                for r in rows
            ),

        'note':
            (
                'Generation only; '
                'no automatic QA accuracy claimed. '
                'Evaluate against gold separately.'
            ),
    }

    with open(
        OUT
        / 'dialogue_generation_metrics.json',
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(
            m,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        '\n===== SCM DIALOGUE '
        'COMPATIBILITY GENERATION COMPLETE ====='
    )

    print(
        json.dumps(
            m,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        'SCM_DIALOGUE_QWEN_OK'
    )


def inspect(langs):
    for lang in langs:
        ds, qs = datasets(lang)

        ids = sorted({
            q['id']
            for q in qs
        })

        turns = sum(
            len(ds[i]) // 2
            for i in ids
        )

        print(
            f'[INSPECT] {lang}: '
            f'dialogues={len(ids)} '
            f'turns={turns} '
            f'questions={len(qs)} '
            f'single='
            f'{sum(q.get("mode") == "single-turn" for q in qs)} '
            f'multi='
            f'{sum(q.get("mode") == "multi-turn" for q in qs)}'
        )

    print(
        'SCM_DIALOGUE_INSPECT_OK'
    )


def main():
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
        '--lang',
        choices=[
            'zh',
            'en',
            'both',
        ],
        default='both',
    )

    a = ap.parse_args()

    langs = (
        ['zh', 'en']
        if a.lang == 'both'
        else [a.lang]
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f'REPO={REPO}\n'
        f'COMMIT={COMMIT}\n'
        f'MODEL={MODEL}\n'
        f'CONTRIEVER={CONTRIEVER}\n'
        f'STAGE={a.stage}\n'
        f'LANGS={langs}'
    )

    inspect(
        langs
    )

    if a.stage == 'inspect':
        return

    gen_path = (
        OUT
        / 'generation_cache_qwen7b.jsonl'
    )

    sum_path = (
        OUT
        / 'turn_summaries_qwen7b.jsonl'
    )

    emb_path = (
        OUT
        / 'contriever_embeddings.pt'
    )

    res_path = (
        OUT
        / 'dialogue_answers_qwen7b.jsonl'
    )

    # Phase 1: Qwen turn summaries
    q = Qwen(
        GenCache(
            gen_path
        )
    )

    prepare_summaries(
        langs,
        q,
        sum_path,
    )

    del q
    free_gpu()

    # Phase 2: Contriever raw-turn/query embeddings
    prepare_embeddings(
        langs,
        emb_path,
    )

    # Phase 3: Qwen QA + controller
    q = Qwen(
        GenCache(
            gen_path
        )
    )

    run_qa(
        langs,
        q,
        sum_path,
        emb_path,
        res_path,
    )

    del q
    free_gpu()


if __name__ == '__main__':
    main()
