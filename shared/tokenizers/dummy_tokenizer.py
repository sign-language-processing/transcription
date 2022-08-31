from typing import List

import torch

from .base_tokenizer import BaseTokenizer


class DummyTokenizer(BaseTokenizer):

    def __init__(self, **kwargs):
        super().__init__(['a'], **kwargs)

    def tokens_to_text(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def __call__(self, texts: List[str], is_tokenized=None, device=None):
        desired_shape = (len(texts), 3)
        return {
            "tokens_ids": torch.tensor([[4, 1, 3]], dtype=torch.long, device=device).expand(desired_shape),
            "positions": torch.tensor([[1, 2, 3]], dtype=torch.int, device=device).expand(desired_shape),
            "attention_mask": torch.tensor([[0, 0, 0]], dtype=torch.bool, device=device).expand(desired_shape),
        }
