import json
from contextlib import contextmanager
from typing import List, Tuple, Optional

from transformers import PreTrainedTokenizer, PretrainedConfig

from ...shared.tokenizers import HamNoSysTokenizer, SignWritingTokenizer


class SignLanguageTokenizerConfig(PretrainedConfig):
    model_type = "sign-language"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SignLanguageTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.hamnosys_tokenizer = HamNoSysTokenizer()
        self.signwriting_tokenizer = SignWritingTokenizer(starting_index=len(self.hamnosys_tokenizer))

        self.i2s = {**self.hamnosys_tokenizer.i2s, **self.signwriting_tokenizer.i2s}
        self.s2i = {**self.hamnosys_tokenizer.s2i, **self.signwriting_tokenizer.s2i}

        self.pad_token = self.hamnosys_tokenizer.pad_token
        self.unk_token = self.hamnosys_tokenizer.unk_token
        self.bos_token = self.hamnosys_tokenizer.bos_token
        self.eos_token = self.hamnosys_tokenizer.eos_token

        self.is_encoder = True

    @property
    def vocab_size(self):
        return len(self.i2s)

    def get_vocab(self):
        return self.s2i

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def _tokenize(self, text: str) -> List[str]:
        if text.isascii():
            return list(self.signwriting_tokenizer.text_to_tokens(text))

        return self.hamnosys_tokenizer.text_to_tokens(text)

    def _convert_token_to_id(self, token):
        if token not in self.s2i:
            return self.unk_token_id
        return self.s2i[token]

    def _convert_id_to_token(self, index):
        return self.i2s[index]

    def convert_tokens_to_string(self, tokens):
        if len(tokens) == 0:
            return ""

        if tokens[0].isascii():
            return self.signwriting_tokenizer.tokens_to_text(tokens)

        return self.hamnosys_tokenizer.tokens_to_text(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        s2i_path = save_directory + "/" + (filename_prefix if filename_prefix else "") + "-s2i.json"
        with open(s2i_path, "w") as f:
            json.dump(self.s2i, f)
        return tuple([s2i_path])


    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
    #     return cls.from_prtrained_tokenizers(
    #         encoder_tokenizer_name_or_path=os.path.join(pretrained_model_name_or_path, 'encoder_tokenizer'),
    #         decoder_tokenizer_name_or_path=os.path.join(pretrained_model_name_or_path, 'decoder_tokenizer'),
    #     )
    #
    # @classmethod
    # def from_pretrained_tokenizers(
    #         cls,
    #         encoder_tokenizer: Union[str, os.PathLike, PreTrainedTokenizer],
    #         decoder_tokenizer: Union[str, os.PathLike, PreTrainedTokenizer],
    #         *init_inputs,
    #         **kwargs
    # ):
    #     encoder_tokenizer = encoder_tokenizer if isinstance(encoder_tokenizer, PreTrainedTokenizer) else \
    #         AutoTokenizer.from_pretrained(encoder_tokenizer, use_fast=False)
    #     decoder_tokenizer = decoder_tokenizer if isinstance(decoder_tokenizer, PreTrainedTokenizer) else \
    #         AutoTokenizer.from_pretrained(decoder_tokenizer, use_fast=False)
    #     return cls(encoder_tokenizer, decoder_tokenizer, *init_inputs, **kwargs)

    # def save_pretrained(
    #         self,
    #         save_directory: Union[str, os.PathLike],
    #         legacy_format: Optional[bool] = None,
    #         filename_prefix: Optional[str] = None,
    #         push_to_hub: bool = False,
    #         **kwargs,
    # ) -> Tuple[str]:
    #     encoder_tokenizer_save_directory = os.path.join(save_directory, 'encoder_tokenizer')
    #
    #     self.encoder_tokenizer.save_pretrained(
    #         save_directory=encoder_tokenizer_save_directory,
    #         lagacy_format=legacy_format,
    #         filename_prefix=filename_prefix,
    #         push_to_hub=push_to_hub,
    #         **kwargs
    #     )
    #
    #     decoder_tokenizer_save_directory = os.path.join(save_directory, 'decoder_tokenizer')
    #
    #     self.decoder_tokenizer.save_pretrained(
    #         save_directory=decoder_tokenizer_save_directory,
    #         lagacy_format=legacy_format,
    #         filename_prefix=filename_prefix,
    #         push_to_hub=push_to_hub,
    #         **kwargs
    #     )

    @contextmanager
    def as_target_tokenizer(self):
        self.is_encoder = False
        yield
        self.is_encoder = True
