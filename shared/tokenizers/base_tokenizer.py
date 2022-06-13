from __future__ import annotations

from typing import List

import torch

from ...shared.collator import zero_pad_collator


class BaseTokenizer:

    def __init__(self, tokens: List[str], starting_index=None):
        if starting_index is None:
            starting_index = 4

        self.pad_token = "[PAD]"
        self.bos_token = "[CLS]"
        self.eos_token = "[SEP]"
        self.unk_token = "[UNK]"

        self.i2s = {(i + starting_index): c for i, c in enumerate(tokens)}
        self.i2s[0] = self.pad_token
        self.i2s[1] = self.bos_token
        self.i2s[2] = self.eos_token
        self.i2s[3] = self.unk_token
        self.s2i = {c: i for i, c in self.i2s.items()}

        self.pad_token_id = self.s2i[self.pad_token]
        self.bos_token_id = self.s2i[self.bos_token]
        self.eos_token_id = self.s2i[self.eos_token]
        self.unk_token_id = self.s2i[self.unk_token]

    def __len__(self):
        return len(self.i2s)

    def text_to_tokens(self, text: str) -> List[str]:
        raise NotImplementedError()

    def tokens_to_text(self, tokens: List[str]) -> str:
        raise NotImplementedError()

    def tokenize(self, text: str):
        return [self.bos_token_id] + [self.s2i[c] for c in self.text_to_tokens(text)]

    def detokenize(self, tokens: List[int]):
        if len(tokens) == 0:
            return ""
        if tokens[0] == self.bos_token_id:
            tokens = tokens[1:]

        try:
            padding_index = tokens.index(self.pad_token_id)
            tokens = tokens[:padding_index]
        except ValueError:
            pass

        return self.tokens_to_text([self.i2s[t] for t in tokens])

    def __call__(self, texts: List[str] | torch.Tensor, is_tokenized=False, device=None):
        if not is_tokenized:
            all_tokens = [self.tokenize(text) for text in texts]
        else:
            all_tokens = texts.tolist()

        tokens_batch = zero_pad_collator([{
            "tokens_ids": torch.tensor(tokens, dtype=torch.long, device=device),
            "attention_mask": torch.ones(len(tokens), dtype=torch.bool, device=device),
            "positions": torch.arange(0, len(tokens), dtype=torch.int, device=device)
        } for tokens in all_tokens])
        # In transformers, 1 is mask, not 0
        tokens_batch["attention_mask"] = torch.logical_not(tokens_batch["attention_mask"])

        return tokens_batch
