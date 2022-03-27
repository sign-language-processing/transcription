import unittest

import torch

from text_to_pose.tokenizers.hamnosys.hamnosys_tokenizer import HamNoSysTokenizer


class TokenizerTestCase(unittest.TestCase):
    def test_multiple_sentence(self):
        expected = {
            'tokens_ids': torch.tensor([[1, 11, 95, 0, 0, 0, 0, 0, 0],
                                        [1, 11, 95, 28, 40, 56, 118, 176, 190]], dtype=torch.int32),
            'positions': torch.tensor([[0, 1, 2, 0, 0, 0, 0, 0, 0],
                                       [0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32),
            'attention_mask': torch.tensor([[False, False, False, True, True, True, True, True, True],
                                            [False, False, False, False, False, False, False, False, False]],
                                           dtype=torch.bool)
        }

        tokenizer = HamNoSysTokenizer()
        tokenized = tokenizer([
            "\ue000\ue071",
            "\ue000\ue071\ue012\ue029\ue03f\ue089\ue0c6\ue0d8"
        ])

        for key, value in expected.items():
            self.assertTrue(torch.all(torch.eq(value, tokenized[key])))


if __name__ == '__main__':
    unittest.main()
