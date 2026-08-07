from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.common import MockClient, clean_response
from scripts.dataset_utils import extract_targets, load_dialogues
from scripts.metrics import automatic_metrics, exact_sign_test
from scripts.run_table3 import judge_prompt, parse_scores


ROOT = Path(__file__).resolve().parents[1]


class OfflineTests(unittest.TestCase):
    def test_public_first_100_target_count(self) -> None:
        dialogues = load_dialogues(ROOT / "data/ESConv.json")
        targets = extract_targets(dialogues, 100, "first", 42)
        self.assertEqual(len(targets), 1210)
        self.assertEqual(len({target.target_id for target in targets}), 1210)

    def test_metrics_identity(self) -> None:
        scores = automatic_metrics(["hello there"], ["hello there"])
        self.assertAlmostEqual(scores["B-1"], 100.0)
        self.assertAlmostEqual(scores["F1"], 100.0)
        self.assertAlmostEqual(scores["R-L"], 100.0)

    def test_clean_response(self) -> None:
        self.assertEqual(clean_response("Response: [Question] What would help?"), "What would help?")

    def test_judge_parser(self) -> None:
        text = "\n".join(f"{name}: 1, 2, 3, 4, 5; reasons" for name in ["Fluency", "Identification", "Comforting", "Suggestion", "Overall"])
        parsed = parse_scores(text)
        self.assertEqual(parsed["Overall"], [1, 2, 3, 4, 5])

    def test_mock_judge_end_to_end(self) -> None:
        prompt = judge_prompt("User: I feel worried.", dict(zip("ABCDE", ["Supportive reply"] * 5)))
        scores = parse_scores(MockClient().complete(prompt).text)
        self.assertEqual(scores["Fluency"], [4, 4, 4, 4, 4])

    def test_configs_use_local_judge(self) -> None:
        for name in ("config.smoke.json", "config.full.json"):
            config = json.loads((ROOT / name).read_text(encoding="utf-8"))
            self.assertIn("127.0.0.1:11434", config["judge"]["base_url"])
            self.assertEqual(config["judge"]["type"], "local-llm-approximation")

    def test_sign_test(self) -> None:
        self.assertLess(exact_sign_test(20, 5), 0.01)


if __name__ == "__main__":
    unittest.main()
