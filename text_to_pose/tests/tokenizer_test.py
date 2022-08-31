import unittest

import torch

from ...shared.tokenizers import HamNoSysTokenizer


class TokenizerTestCase(unittest.TestCase):

    def test_expected_token_id(self):
        tokenizer = HamNoSysTokenizer()
        tokenized = tokenizer(["\ue000\ue071"])
        self.assertEqual(tokenized['tokens_ids'][0][0], 1)
        self.assertEqual(tokenized['tokens_ids'][0][1], 13)
        self.assertEqual(tokenized['tokens_ids'][0][2], 97)

    def test_multiple_sentence(self):
        expected = {
            'tokens_ids':
                torch.tensor([[1, 13, 97, 0, 0, 0, 0, 0, 0], [1, 13, 97, 30, 42, 58, 120, 178, 192]],
                             dtype=torch.int32),
            'positions':
                torch.tensor([[0, 1, 2, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32),
            'attention_mask':
                torch.tensor([[False, False, False, True, True, True, True, True, True],
                              [False, False, False, False, False, False, False, False, False]],
                             dtype=torch.bool)
        }

        tokenizer = HamNoSysTokenizer()
        tokenized = tokenizer(["\ue000\ue071", "\ue000\ue071\ue012\ue029\ue03f\ue089\ue0c6\ue0d8"])

        for key, value in expected.items():
            self.assertTrue(torch.all(torch.eq(value, tokenized[key])))


if __name__ == '__main__':
    unittest.main()
