from pathlib import Path
from typing import List

from fontTools.ttLib import TTFont

from ..base_tokenizer import BaseTokenizer


class HamNoSysTokenizer(BaseTokenizer):

    def __init__(self, starting_index=None, **kwargs):
        self.font_path = Path(__file__).parent.joinpath("HamNoSysUnicode.ttf")

        with TTFont(self.font_path) as font:
            tokens = [chr(key) for key in font["cmap"].getBestCmap().keys()]

        super().__init__(tokens=tokens, starting_index=starting_index, **kwargs)

    def text_to_tokens(self, text: str) -> List[str]:
        return list(text)

    def tokens_to_text(self, tokens: List[str]) -> str:
        return "".join(tokens)


if __name__ == "__main__":
    tokenizer = HamNoSysTokenizer()
    print(tokenizer(["\ue000\ue071", "\ue000\ue071\ue012\ue029\ue03f\ue089\ue0c6\ue0d8"]))
