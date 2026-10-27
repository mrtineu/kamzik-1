"""Pure-helper tests; no GPU, torch or vLLM needed.  python -m unittest thinkingcap/test_pipeline.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline as p  # noqa: E402

MATH_ROW = {'prompt': [{'role': 'user', 'content': 'Tom has 3 apples and buys 4 more.\n\n' + p.MATH_INSTRUCTION}]}
QA = ('Where would you keep a sandwich cold?\n\nA. oven\nB. refrigerator\nC. desert\nD. car\nE. bank\n\n'
      + p.CHOICE_INSTRUCTION)


def samples(*pairs):
    return [{'n_tokens': n, 'correct': c, 'text': str(n), 'token_ids': [0] * n} for n, c in pairs]


class Helpers(unittest.TestCase):
    def test_question_body_strips_instruction(self):
        self.assertEqual(p.question_body(MATH_ROW['prompt'][0]['content']), 'Tom has 3 apples and buys 4 more.')
        self.assertTrue(p.question_body(QA).endswith('E. bank'))
        self.assertEqual(p.question_body('no instruction'), 'no instruction')

    def test_overlap_uses_question_body_only(self):
        words = ' '.join(f'w{i}' for i in range(20))
        rows = [{'prompt': [{'content': words + '\n\n' + p.MATH_INSTRUCTION}]},
                {'prompt': [{'content': 'something else entirely\n\n' + p.MATH_INSTRUCTION}]}]
        self.assertEqual(p.overlapping(rows, ['prefix ' + words.upper()], 13), [0])
        self.assertEqual(p.overlapping(rows, [' '.join(f'w{i}' for i in range(12))], 13), [])
        # the shared instruction alone must not flag anything
        self.assertEqual(p.overlapping(rows, ['x ' + p.MATH_INSTRUCTION + ' y'], 13), [])

    def test_limited_takes_first_per_source(self):
        rows = [{'source': s, 'i': i} for i, s in enumerate('aabab')]
        self.assertEqual([r['i'] for r in p.limited(rows, 1)], [0, 2])
        self.assertIs(p.limited(rows, None), rows)

    def test_reference_length_needs_enough_correct(self):
        self.assertEqual(p.reference_length(samples((100, True), (300, True), (50, False), (900, False)), 2), 200)
        self.assertIsNone(p.reference_length(samples((100, True), (50, False)), 2))

    def test_median_trace_and_shortest(self):
        s = samples((300, True), (100, True), (200, True), (10, False))
        self.assertEqual(p.median_trace(s)['n_tokens'], 200)
        self.assertEqual(p.median_trace(samples((300, True), (100, True)))['n_tokens'], 100)
        self.assertEqual(p.shortest_correct(s)['n_tokens'], 100)
        self.assertIsNone(p.shortest_correct(samples((5, False))))

    def test_accept_requires_strictly_shorter(self):
        self.assertTrue(p.accept({'n_tokens': 199}, 200))
        self.assertFalse(p.accept({'n_tokens': 200}, 200))
        self.assertFalse(p.accept(None, 200))
        self.assertFalse(p.accept({'n_tokens': 1}, None))

    def test_parse_coaching(self):
        good = '- Skip restating the problem.\n- Do the subtraction once.\n- Stop after the first check.'
        self.assertEqual(len(p.parse_coaching(good)), 3)
        self.assertEqual(len(p.parse_coaching('<think>x</think>\nSure:\n' + good)), 3)
        self.assertIsNone(p.parse_coaching('- one\n- two'))
        self.assertIsNone(p.parse_coaching(good + '\n- four'))
        self.assertIsNone(p.parse_coaching('- ok\n- x\n- fine'))

    def test_math_leak(self):
        self.assertTrue(p.coaching_leaks(['the total is 1,250 so stop'], '1250', 'math', ''))
        self.assertTrue(p.coaching_leaks(['you get -7 quickly'], '-7', 'math', ''))
        self.assertFalse(p.coaching_leaks(['you get 12507 quickly', 'use 12.5 once'], '12', 'math', ''))
        self.assertFalse(p.coaching_leaks(['skip step 3'], '42', 'math', ''))

    def test_choice_leak(self):
        leaks = ['Go straight to option B.', 'The answer is C', 'Pick (D) faster', 'B fits, stop there',
                 'Stop weighing Refrigerator against oven', 'Answer: E']
        for text in leaks:
            self.assertTrue(p.coaching_leaks([text], 'B', 'multiple_choice', QA), text)
        clean = ['A quick read of the question is enough.', 'Do not re-read every option twice.',
                 'Commit after one pass over the choices.']
        self.assertFalse(p.coaching_leaks(clean, 'B', 'multiple_choice', QA))
        numeric = 'Q?\n\n1. red\n2. blue\n\n' + p.CHOICE_INSTRUCTION
        self.assertTrue(p.coaching_leaks(['choice 2 is right'], '2', 'multiple_choice', numeric))
        self.assertFalse(p.coaching_leaks(['skip step 2 entirely'], '2', 'multiple_choice', numeric))

    def test_coached_messages_append_only(self):
        messages = p.coached_messages(MATH_ROW['prompt'], ['a', 'b', 'c'])
        self.assertTrue(messages[0]['content'].startswith(MATH_ROW['prompt'][0]['content'] + '\n\n'))
        self.assertTrue(messages[0]['content'].endswith(p.COACH_HEADER + '\n- a\n- b\n- c'))
        self.assertEqual(MATH_ROW['prompt'][0]['content'].count('Brief coaching'), 0)

    def test_plan_steps_cover_every_example_each_epoch(self):
        steps = p.plan_steps(70, 32, 2, 1)
        self.assertEqual([len(s) for s in steps], [32, 32, 6, 32, 32, 6])
        self.assertEqual(sorted(i for s in steps[:3] for i in s), list(range(70)))
        self.assertNotEqual(steps[0], steps[3])
        self.assertEqual(steps, p.plan_steps(70, 32, 2, 1))

    def test_checkpoint_steps(self):
        self.assertEqual(p.checkpoint_steps(100, 4), [25, 50, 75, 100])
        self.assertEqual(p.checkpoint_steps(6, 4), [2, 3, 5, 6])
        self.assertEqual(p.checkpoint_steps(2, 4), [1, 2])

    def test_micro_batches_respect_budget(self):
        lengths = [5000, 4000, 3000, 2000, 1000, 9000, 100]
        batches = p.micro_batches(range(len(lengths)), lengths, 10000)
        self.assertEqual(sorted(i for b in batches for i in b), list(range(len(lengths))))
        for b in batches:
            self.assertTrue(len(b) == 1 or len(b) * max(lengths[i] for i in b) <= 10000)
        self.assertEqual(p.micro_batches([0], [20000], 10000), [[0]])

    def test_lr_schedule(self):
        lrs = [p.lr_at(s, 100, 1.0, 0.03) for s in range(100)]
        self.assertAlmostEqual(lrs[0], 1 / 3)
        self.assertAlmostEqual(lrs[2], 1.0)
        self.assertTrue(all(a >= b for a, b in zip(lrs[2:], lrs[3:])))
        self.assertLess(lrs[-1], 0.01)


if __name__ == '__main__':
    unittest.main()
