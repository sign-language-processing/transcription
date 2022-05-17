import os

from contextlib import contextmanager
from typing import Tuple, Optional, Union, List

from transformers import AutoTokenizer, PreTrainedTokenizer

from text_to_text._tokenizers.sign_language_tokenizer import SignLanguageTokenizer


class EncoderDecoderTokenizer(PreTrainedTokenizer):
    def __init__(self, encoder_tokenizer, decoder_tokenizer, **kwargs) -> None:
        super().__init__(
            bos_token=encoder_tokenizer.bos_token,
            eos_token=encoder_tokenizer.eos_token,
            unk_token=encoder_tokenizer.unk_token,
            sep_token=encoder_tokenizer.sep_token,
            pad_token=encoder_tokenizer.pad_token,
            cls_token=encoder_tokenizer.cls_token,
            mask_token=encoder_tokenizer.mask_token,
            additional_special_tokens=encoder_tokenizer.additional_special_tokens,
            **kwargs,
        )
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.current_tokenizer = encoder_tokenizer

    @property
    def vocab_size(self):
        return self.current_tokenizer.vocab_size

    def __getstate__(self):
        return self.current_tokenizer.__getstate__()

    def __setstate__(self, d):
        self.current_tokenizer.__setstate__(d)

    def get_vocab(self):
        return self.current_tokenizer.get_vocab()

    def preprocess_text(self, inputs):
        return self.current_tokenizer.preprocess_text(inputs)

    def _tokenize(self, text: str) -> List[str]:
        # pylint: disable=protected-access
        return self.current_tokenizer._tokenize(text)

    def _convert_token_to_id(self, token):
        # pylint: disable=protected-access
        return self.current_tokenizer._convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        # pylint: disable=protected-access
        return self.current_tokenizer._convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens):
        return self.current_tokenizer.convert_tokens_to_string(tokens)

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return self.current_tokenizer.build_inputs_with_special_tokens(token_ids_0, token_ids_1)

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        return self.current_tokenizer.get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens)

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return self.current_tokenizer.create_token_type_ids_from_sequences(token_ids_0, token_ids_1)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return self.current_tokenizer.save_vocabulary(save_directory, filename_prefix)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
        return cls.from_pretrained_tokenizers(
            encoder_tokenizer=os.path.join(pretrained_model_name_or_path, 'encoder_tokenizer'),
            decoder_tokenizer=os.path.join(pretrained_model_name_or_path, 'decoder_tokenizer'),
        )

    @classmethod
    def from_pretrained_tokenizers(
            cls,
            encoder_tokenizer: Union[str, os.PathLike, PreTrainedTokenizer],
            decoder_tokenizer: Union[str, os.PathLike, PreTrainedTokenizer],
            *init_inputs,
            **kwargs
    ):
        encoder_tokenizer = encoder_tokenizer if isinstance(encoder_tokenizer, PreTrainedTokenizer) else \
            AutoTokenizer.from_pretrained(encoder_tokenizer, use_fast=False)
        # decoder_tokenizer = decoder_tokenizer if isinstance(decoder_tokenizer, PreTrainedTokenizer) else \
        #     AutoTokenizer.from_pretrained(decoder_tokenizer, use_fast=False)
        decoder_tokenizer = SignLanguageTokenizer()  # TODO load this dynamically by registring config
        return cls(encoder_tokenizer, decoder_tokenizer, *init_inputs, **kwargs)

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            legacy_format: Optional[bool] = None,
            filename_prefix: Optional[str] = None,
            push_to_hub: bool = False,
            **kwargs,
    ):
        encoder_tokenizer_save_directory = os.path.join(save_directory, 'encoder_tokenizer')

        self.encoder_tokenizer.save_pretrained(
            save_directory=encoder_tokenizer_save_directory,
            lagacy_format=legacy_format,
            filename_prefix=filename_prefix,
            push_to_hub=push_to_hub,
            **kwargs
        )

        decoder_tokenizer_save_directory = os.path.join(save_directory, 'decoder_tokenizer')

        self.decoder_tokenizer.save_pretrained(
            save_directory=decoder_tokenizer_save_directory,
            lagacy_format=legacy_format,
            filename_prefix=filename_prefix,
            push_to_hub=push_to_hub,
            **kwargs
        )

    def _set_current_tokenizer(self, tokenizer):
        assert tokenizer in [self.encoder_tokenizer, self.decoder_tokenizer]
        self.__dict__.update(tokenizer.__dict__)
        self.current_tokenizer = tokenizer

    @contextmanager
    def as_target_tokenizer(self):
        self._set_current_tokenizer(self.decoder_tokenizer)
        yield
        self._set_current_tokenizer(self.encoder_tokenizer)
