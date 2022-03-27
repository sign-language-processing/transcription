from pathlib import Path
from typing import List

import torch
from fontTools.ttLib import TTFont


class HamNoSysTokenizer:

    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1

        self.font_path = Path(__file__).parent.joinpath("HamNoSysUnicode.ttf")

        with TTFont(self.font_path) as font:
            tokens = [chr(key) for key in font["cmap"].getBestCmap().keys()]

        self.i2s = {(i + 2): c for i, c in enumerate(tokens)}
        self.s2i = {c: i for i, c in self.i2s.items()}

    def __len__(self):
        return len(self.i2s) + 2

    def tokenize(self, text: str):
        return [self.bos_token_id] + [self.s2i[c] for c in text]

    def __call__(self, texts: List[str], device=None):
        all_tokens = [self.tokenize(text) for text in texts]
        max_tokens = max([len(t) for t in all_tokens])

        shape = (len(texts), max_tokens)

        tokens_ids = torch.zeros(shape, dtype=torch.int, device=device)
        attention_mask = torch.ones(shape, dtype=torch.bool, device=device)
        positions = torch.arange(0, max_tokens, dtype=torch.int, device=device).expand(shape)

        # TODO use the zero pad collator?
        for i, tokens in enumerate(all_tokens):
            for j, token in enumerate(tokens):
                tokens_ids[i, j] = token
            attention_mask[i, :len(tokens)] = 0

        return {
            "tokens_ids": tokens_ids,
            "positions": positions,
            "attention_mask": attention_mask,
        }


if __name__ == "__main__":
    tokenizer = HamNoSysTokenizer()
    print(tokenizer([
        "\ue000\ue071",
        "\ue000\ue071\ue012\ue029\ue03f\ue089\ue0c6\ue0d8"
    ]))
