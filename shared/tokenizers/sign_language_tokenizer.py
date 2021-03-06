from typing import List

from .base_tokenizer import BaseTokenizer
from .hamnosys.hamnosys_tokenizer import HamNoSysTokenizer
from .signwriting.signwriting_tokenizer import SignWritingTokenizer


class SignLanguageTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        self.hamnosys_tokenizer = HamNoSysTokenizer()
        self.signwriting_tokenizer = SignWritingTokenizer(starting_index=len(self.hamnosys_tokenizer))

        super().__init__([])

        self.i2s = {**self.hamnosys_tokenizer.i2s, **self.signwriting_tokenizer.i2s}
        self.s2i = {**self.hamnosys_tokenizer.s2i, **self.signwriting_tokenizer.s2i}

    def tokenize(self, text: str) -> List[str]:
        if text.isascii():
            return self.signwriting_tokenizer.tokenize(text)

        return self.hamnosys_tokenizer.tokenize(text)

    def text_to_tokens(self, text: str) -> List[str]:
        if text.isascii():
            return self.signwriting_tokenizer.text_to_tokens(text)

        return self.hamnosys_tokenizer.text_to_tokens(text)

    def tokens_to_text(self, tokens: List[str]) -> str:
        if all(t.isascii() for t in tokens):
            return self.signwriting_tokenizer.tokens_to_text(tokens)

        return self.hamnosys_tokenizer.tokens_to_text(tokens)
