#!/usr/bin/env python3

import json
import random
from pathlib import Path

ROOT = Path(
    "/root/autodl-tmp/paper_repro/"
    "adapters/scm_book_repro/output"
)

OUT = Path(
    "/root/autodl-tmp/paper_repro/"
    "adapters/scm_book_repro/evaluation"
)

SEED = 42

BOOKS = [
    "三体.txt",
    "三国演义.txt",
    "哈利.波特中文版1.txt",
    "平凡的世界.txt",
    "西游记.txt",
    "Gone_With_The_Wind.txt",
    "Jane_Eyre.txt",
    "The_Great_Gatsby.txt",
    "The_Life_and_Adventures_of_Robinson_Crusoe.txt",
    "The_Old_Man_and_the_Sea.txt",
]


def read_summary(book, mode):
    p = ROOT / book / mode / "final_summary.txt"

    if not p.exists():
        raise FileNotFoundError(p)

    s = p.read_text(
        encoding="utf-8"
    ).strip()

    if not s:
        raise RuntimeError(
            f"empty summary: {p}"
        )

    return s


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = random.Random(SEED)

    blind = []
    key = []

    for i, book in enumerate(
        BOOKS,
        1,
    ):
        scm = read_summary(
            book,
            "scm",
        )

        no_scm = read_summary(
            book,
            "no_scm",
        )

        if rng.random() < 0.5:
            A = scm
            B = no_scm
            mapping = {
                "A": "scm",
                "B": "no_scm",
            }
        else:
            A = no_scm
            B = scm
            mapping = {
                "A": "no_scm",
                "B": "scm",
            }

        pair_id = f"book_{i:02d}"

        blind.append({
            "pair_id": pair_id,
            "book": book,

            "summary_A": A,
            "summary_B": B,

            "coverage_winner": "",
            "coherence_winner": "",

            "coverage_comment": "",
            "coherence_comment": "",
        })

        key.append({
            "pair_id": pair_id,
            "book": book,
            "mapping": mapping,
        })

    blind_path = (
        OUT
        / "book_pairwise_blind.json"
    )

    key_path = (
        OUT
        / "book_pairwise_key.json"
    )

    blind_path.write_text(
        json.dumps(
            blind,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    key_path.write_text(
        json.dumps(
            key,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"pairs={len(blind)}"
    )

    print(
        f"blind={blind_path}"
    )

    print(
        f"key={key_path}"
    )

    print(
        "BOOK_BLIND_EVAL_PREP_OK"
    )


if __name__ == "__main__":
    main()
