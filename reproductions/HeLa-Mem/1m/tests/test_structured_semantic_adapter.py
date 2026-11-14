import unittest

from hela_mem.personamem_structured_adapter import (
    parse_semantic_payload,
    validate_semantic_facts,
)


class TestStructuredSemanticAdapter(unittest.TestCase):

    def test_valid_json(self):
        raw = '''
        {
          "profile": "Creative and reflective.",
          "private_facts": ["User enjoys producing music."],
          "assistant_knowledge": []
        }
        '''
        x = parse_semantic_payload(raw)
        self.assertEqual(
            x["private_facts"],
            ["User enjoys producing music."]
        )

    def test_markdown_json(self):
        raw = '''```json
        {
          "profile": "",
          "private_facts": ["User released a remix album."],
          "assistant_knowledge": []
        }
        ```'''
        x = parse_semantic_payload(raw)
        self.assertEqual(len(x["private_facts"]), 1)

    def test_reject_prompt_leak(self):
        bad = [
            "ACTUAL CONVERSATION:",
            "section.",
            "Time:unknown",
            "User: I went hiking.",
            "Assistant: That sounds difficult.",
            "Extract only statements supported by ACTUAL CONVERSATION.",
        ]
        for fact in bad:
            with self.subTest(fact=fact):
                with self.assertRaises(ValueError):
                    validate_semantic_facts([fact])

    def test_reject_free_text_echo(self):
        raw = '''
        section.
        ACTUAL CONVERSATION:
        User: I like music.
        Assistant: Interesting.
        Time:unknown
        '''
        with self.assertRaises(ValueError):
            parse_semantic_payload(raw)


if __name__ == "__main__":
    unittest.main()
