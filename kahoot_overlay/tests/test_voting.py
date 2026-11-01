import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from voting import (
    extract_first_json_object,
    final_confidence,
    parse_model_response,
    parse_stage1_extraction,
    vote,
)


class TestVotingAndParsing(unittest.TestCase):
    def test_unanimity_three_tries(self):
        tries = [
            parse_model_response(
                '{"found": true, "type": "single", "think": "2+2 is 4", "answer": ["red"], "pos": ["top-left"], "conf": 90}',
                try_index=0,
                is_temp_zero=True,
            ),
            parse_model_response(
                '{"found": true, "type": "single", "think": "4 is red", "answer": ["red"], "pos": ["top-left"], "conf": 85}',
                try_index=1,
                is_temp_zero=False,
            ),
            parse_model_response(
                '{"found": true, "type": "single", "think": "red triangle", "answer": ["red"], "pos": ["top-left"], "conf": 95}',
                try_index=2,
                is_temp_zero=False,
            ),
        ]
        res = vote(tries, configured_tries=3, early_exited=False)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["answer"], ["red"])
        self.assertEqual(res["pos"], ["top-left"])
        # avg_conf = 90, agreement = 3/3 = 1.0 -> 90%
        self.assertEqual(res["confidence"], 90)
        self.assertEqual(res["agree_text"], "3 of 3 tries agree")

    def test_never_100_percent_confidence(self):
        tries = [
            parse_model_response(
                '{"found": true, "type": "single", "answer": ["blue"], "pos": ["top-right"], "conf": 100}',
                try_index=i,
                is_temp_zero=(i == 0),
            )
            for i in range(3)
        ]
        res = vote(tries, configured_tries=3)
        self.assertEqual(res["confidence"], 99)
        self.assertLess(res["confidence"], 100)

    def test_single_try_clamped_to_95(self):
        t = parse_model_response(
            '{"found": true, "type": "single", "answer": ["green"], "pos": ["bottom-right"], "conf": 100}',
            try_index=0,
            is_temp_zero=True,
        )
        res = vote([t], configured_tries=1)
        self.assertEqual(res["confidence"], 95)
        self.assertEqual(res["agree_text"], "single try")

    def test_split_vote_majority_two_of_three(self):
        tries = [
            parse_model_response(
                '{"found": true, "type": "single", "answer": ["yellow"], "pos": ["bottom-left"], "conf": 90}',
                try_index=0,
                is_temp_zero=True,
            ),
            parse_model_response(
                '{"found": true, "type": "single", "answer": ["yellow"], "pos": ["bottom-left"], "conf": 84}',
                try_index=1,
                is_temp_zero=False,
            ),
            parse_model_response(
                '{"found": true, "type": "single", "answer": ["blue"], "pos": ["top-right"], "conf": 70}',
                try_index=2,
                is_temp_zero=False,
            ),
        ]
        res = vote(tries, configured_tries=3, early_exited=True)
        self.assertEqual(res["answer"], ["yellow"])
        # avg_conf of winning tries = (90 + 84)/2 = 87. agreement = 2/3. 87 * (2/3) = 58
        self.assertEqual(res["confidence"], 58)
        self.assertEqual(res["agree_text"], "2 of 3 tries agree (early)")

    def test_tie_break_by_higher_conf_sum(self):
        tries = [
            parse_model_response(
                '{"found": true, "type": "single", "answer": ["red"], "pos": ["top-left"], "conf": 60}',
                try_index=0,
                is_temp_zero=True,
            ),
            parse_model_response(
                '{"found": true, "type": "single", "answer": ["blue"], "pos": ["top-right"], "conf": 92}',
                try_index=1,
                is_temp_zero=False,
            ),
        ]
        res = vote(tries, configured_tries=2)
        self.assertEqual(res["answer"], ["blue"])
        # avg_conf = 92, agreement = 1/2 = 0.5 -> 46%
        self.assertEqual(res["confidence"], 46)

    def test_tie_break_by_temp_zero_when_conf_equal(self):
        tries = [
            parse_model_response(
                '{"found": true, "type": "single", "answer": ["red"], "pos": ["top-left"], "conf": 80}',
                try_index=0,
                is_temp_zero=True,
            ),
            parse_model_response(
                '{"found": true, "type": "single", "answer": ["blue"], "pos": ["top-right"], "conf": 80}',
                try_index=1,
                is_temp_zero=False,
            ),
        ]
        res = vote(tries, configured_tries=2)
        self.assertEqual(res["answer"], ["red"])
        self.assertEqual(res["confidence"], 40)

    def test_tolerant_parser_code_fences_and_normalization(self):
        raw = """```json
        Here is the result:
        {"found": true, "type": "multi", "think": "Both Red and Green apply", "answer": ["GREEN", "Red", "purple"], "pos": ["bottom-right", "top-left"], "conf": 125}
        ```"""
        parsed = parse_model_response(raw, try_index=0, is_temp_zero=True)
        self.assertIsNotNone(parsed)
        # Sorted tuple for non-order types; unknown 'purple' dropped
        self.assertEqual(parsed["answer"], ("green", "red"))
        self.assertEqual(parsed["pos"], ("bottom-right", "top-left"))
        self.assertEqual(parsed["conf"], 100)

    def test_order_question_preserves_sequence(self):
        raw = '{"found": true, "type": "order", "answer": ["yellow", "red", "blue", "green"], "pos": ["bottom-left", "top-left", "top-right", "bottom-right"], "conf": 88}'
        parsed = parse_model_response(raw, try_index=0, is_temp_zero=True)
        self.assertEqual(parsed["answer"], ("yellow", "red", "blue", "green"))

    def test_invalid_and_no_question_handling(self):
        self.assertIsNone(parse_model_response("not json at all"))
        self.assertIsNone(parse_model_response('{"found": true, "answer": ["orange"]}'))

        nf1 = parse_model_response('{"found": false}')
        nf2 = parse_model_response('{"found": false}')
        valid1 = parse_model_response('{"found": true, "answer": ["red"], "pos": ["top-left"], "conf": 50}')
        res = vote([nf1, nf2, valid1], configured_tries=3)
        self.assertEqual(res["status"], "no_question")
        self.assertFalse(res["found"])

    def test_stage1_extraction_parser(self):
        s1 = parse_stage1_extraction(
            '{"found": true, "type": "truefalse", "question": "The sun is a star", "options": {"blue": "True", "red": "False"}, "positions": {"blue": "top-right", "red": "top-left"}}'
        )
        self.assertIsNotNone(s1)
        self.assertTrue(s1["found"])
        self.assertEqual(s1["options"]["blue"], "True")
        self.assertEqual(s1["options"]["red"], "False")


if __name__ == "__main__":
    unittest.main()