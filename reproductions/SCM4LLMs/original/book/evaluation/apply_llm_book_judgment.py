#!/usr/bin/env python3

import json
from collections import Counter
from pathlib import Path

ROOT = Path(
    "/root/autodl-tmp/paper_repro/"
    "adapters/scm_book_repro/evaluation"
)

BLIND = ROOT / "book_pairwise_blind.json"
KEY = ROOT / "book_pairwise_key.json"

OUT = ROOT / "book_pairwise_llm_judged.json"
METRICS = ROOT / "book_pairwise_llm_metrics.json"


JUDGMENTS = {
    "book_01": {
        "coverage": "A",
        "coherence": "B",
        "coverage_comment":
            "A covers major early and late arcs, while B is concentrated mainly on the Cheng Xin era.",
        "coherence_comment":
            "B follows a tighter and more continuous late-story progression."
    },

    "book_02": {
        "coverage": "A",
        "coherence": "B",
        "coverage_comment":
            "A covers substantially more major characters, campaigns, and events through reunification.",
        "coherence_comment":
            "B is less list-like and presents the overall historical arc more smoothly."
    },

    "book_03": {
        "coverage": "A",
        "coherence": "B",
        "coverage_comment":
            "A includes more school events and more of the Philosopher's Stone plot.",
        "coherence_comment":
            "B is more compact and reads as a cleaner continuous narrative."
    },

    "book_04": {
        "coverage": "A",
        "coherence": "B",
        "coverage_comment":
            "A includes more character, village, work, and social-change events.",
        "coherence_comment":
            "B avoids the duplicated compressed final-summary section and reads more consistently."
    },

    "book_05": {
        "coverage": "B",
        "coherence": "B",
        "coverage_comment":
            "B spans Sun Wukong's origin, representative pilgrimage episodes, and the ending more broadly.",
        "coherence_comment":
            "B provides a clearer beginning-to-end account of the pilgrimage."
    },

    "book_06": {
        "coverage": "B",
        "coherence": "B",
        "coverage_comment":
            "B contains more concrete plot developments and includes the novel's concluding emotional arc.",
        "coherence_comment":
            "B gives a more connected narrative instead of a primarily thematic overview."
    },

    "book_07": {
        "coverage": "A",
        "coherence": "A",
        "coverage_comment":
            "A covers Gateshead, Lowood, Thornfield, Bertha Mason, the Rivers family, Morton, and reunion with Rochester.",
        "coherence_comment":
            "A has the clearer life-stage chronology and avoids the irrelevant preface material in B."
    },

    "book_08": {
        "coverage": "A",
        "coherence": "B",
        "coverage_comment":
            "A explicitly includes Myrtle's death, Gatsby's pursuit of Daisy, conflict with Tom, and Gatsby's death.",
        "coherence_comment":
            "B presents the novel's themes and Gatsby-Daisy arc more smoothly as a unified summary."
    },

    "book_09": {
        "coverage": "B",
        "coherence": "B",
        "coverage_comment":
            "B covers more of Crusoe's pre-island life, island survival, Friday, mutiny, plantation, and later plans.",
        "coherence_comment":
            "B presents these stages in a clearer chronological progression."
    },

    "book_10": {
        "coverage": "A",
        "coherence": "A",
        "coverage_comment":
            "A includes the 84-day drought, Manolin relationship, marlin struggle, shark attacks, and return with the skeleton.",
        "coherence_comment":
            "A gives a precise and continuous account of Santiago's journey and its central themes."
    },
}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    blind = load(BLIND)
    key = load(KEY)

    key_map = {
        x["pair_id"]: x["mapping"]
        for x in key
    }

    if len(blind) != 10:
        raise RuntimeError(
            f"expected 10 blind pairs, got {len(blind)}"
        )

    coverage = Counter()
    coherence = Counter()

    judged = []

    for row in blind:
        pid = row["pair_id"]

        if pid not in JUDGMENTS:
            raise RuntimeError(
                f"missing judgment: {pid}"
            )

        if pid not in key_map:
            raise RuntimeError(
                f"missing key: {pid}"
            )

        j = JUDGMENTS[pid]

        row["coverage_winner"] = j["coverage"]
        row["coherence_winner"] = j["coherence"]
        row["coverage_comment"] = j["coverage_comment"]
        row["coherence_comment"] = j["coherence_comment"]

        mapping = key_map[pid]

        coverage_method = mapping[
            j["coverage"]
        ]

        coherence_method = mapping[
            j["coherence"]
        ]

        coverage[coverage_method] += 1
        coherence[coherence_method] += 1

        row["resolved_coverage_winner"] = (
            coverage_method
        )

        row["resolved_coherence_winner"] = (
            coherence_method
        )

        judged.append(row)

    metrics = {
        "evaluation_type":
            "LLM-based compatibility evaluation",

        "judge":
            "ChatGPT",

        "blind_pairwise":
            True,

        "books":
            len(judged),

        "coverage": {
            "scm_wins":
                coverage["scm"],

            "no_scm_wins":
                coverage["no_scm"],

            "scm_win_rate":
                coverage["scm"] / len(judged),

            "no_scm_win_rate":
                coverage["no_scm"] / len(judged),
        },

        "coherence": {
            "scm_wins":
                coherence["scm"],

            "no_scm_wins":
                coherence["no_scm"],

            "scm_win_rate":
                coherence["scm"] / len(judged),

            "no_scm_win_rate":
                coherence["no_scm"] / len(judged),
        },

        "important_note":
            (
                "This is an LLM-based compatibility evaluation. "
                "It must not be reported as the original paper's "
                "human blind evaluation."
            ),
    }

    OUT.write_text(
        json.dumps(
            judged,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    METRICS.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "===== SCM BOOK LLM BLIND EVALUATION ====="
    )

    print(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"JUDGED: {OUT}")
    print(f"METRICS: {METRICS}")

    print(
        "SCM_BOOK_LLM_EVAL_OK"
    )


if __name__ == "__main__":
    main()
