import unittest

from hela_mem.personamem_structured_adapter import (
    render_profile_for_official_parser,
    render_assistant_for_official_parser,
)


class TestStructuredSemanticRender(unittest.TestCase):

    def test_profile_render(self):
        payload = {
            "profile": "Creative and reflective.",
            "private_facts": [
                "User enjoys producing music.",
                "User disliked hiking dates."
            ],
            "assistant_knowledge": []
        }

        text = render_profile_for_official_parser(payload)

        self.assertEqual(text.count("【User Profile】"), 1)
        self.assertEqual(text.count("【User Data】"), 1)
        self.assertIn("Creative and reflective.", text)
        self.assertIn("User enjoys producing music.", text)
        self.assertNotIn("ACTUAL CONVERSATION", text)

    def test_assistant_render(self):
        payload = {
            "profile": "",
            "private_facts": [],
            "assistant_knowledge": [
                "The assistant suggested lower-pressure social settings."
            ]
        }

        text = render_assistant_for_official_parser(payload)

        self.assertEqual(text.count("【Assistant Knowledge】"), 1)
        self.assertIn(
            "The assistant suggested lower-pressure social settings.",
            text,
        )
        self.assertNotIn("ACTUAL CONVERSATION", text)

    def test_empty_lists_render_none(self):
        payload = {
            "profile": "",
            "private_facts": [],
            "assistant_knowledge": []
        }

        p = render_profile_for_official_parser(payload)
        a = render_assistant_for_official_parser(payload)

        self.assertIn("【User Data】\nNone", p)
        self.assertIn("【Assistant Knowledge】\nNone", a)


if __name__ == "__main__":
    unittest.main()
