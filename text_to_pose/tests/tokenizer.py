from typing import List

import torch


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0

    def __len__(self):
        return 5

    def __call__(self, texts: List[str], device=None):
        desired_shape = (len(texts), 3)
        return {
            "tokens_ids": torch.tensor([[4, 1, 3]], dtype=torch.int, device=device).expand(desired_shape),
            "positions": torch.tensor([[1, 2, 3]], dtype=torch.int, device=device).expand(desired_shape),
            "attention_mask": torch.tensor([[0, 0, 0]], dtype=torch.bool, device=device).expand(desired_shape),
        }
