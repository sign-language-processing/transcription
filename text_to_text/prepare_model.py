import argparse
import os

from transformers import AutoTokenizer, EncoderDecoderModel

from text_to_text._tokenizers.encoder_decoder_tokenizer import EncoderDecoderTokenizer
from text_to_text._tokenizers.sign_language_tokenizer import SignLanguageTokenizer


def main():
    parser = argparse.ArgumentParser(description='My App.')
    parser.add_argument('--source_model', type=str, required=True, help='name or path of the source model')
    parser.add_argument('--target_model', type=str, required=True, help='name or path of the target model')
    parser.add_argument('--final_name', type=str, required=True, help='name of the final model')
    parser.add_argument('--outputs_dir', type=str, default='./')
    parser.add_argument('--model_max_length', type=int, default=256)
    args = parser.parse_args()

    # prepare source tokenizer
    os.makedirs(args.outputs_dir, exist_ok=True)
    src_tokenizer = AutoTokenizer.from_pretrained(args.source_model, use_fast=False)
    special_tokens = ["[MASK]", "<SW>", "<HNS>"]  # TODO add languages, countries, and signed languages
    src_tokenizer.add_tokens(special_tokens, special_tokens=True)

    # prepare target tokenizer
    trg_tokenizer = SignLanguageTokenizer()

    # create encoder-decoder tokenizer
    tokenizer = EncoderDecoderTokenizer.from_pretrained_tokenizers(src_tokenizer, trg_tokenizer)

    # prepare combined model
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.source_model, args.target_model)

    model.config.tokenizer_class = tokenizer.__class__.__name__
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = src_tokenizer.eos_token_id if src_tokenizer.eos_token_id is not None else src_tokenizer.sep_token_id
    model.config.max_length = args.model_max_length

    with tokenizer.as_target_tokenizer():
        model.decoder.config.pad_token_id = tokenizer.pad_token_id
        model.decoder.config.bos_token_id = tokenizer.bos_token_id
        model.decoder.config.eos_token_id = tokenizer.eos_token_id
        model.config.decoder_start_token_id = model.decoder.config.bos_token_id
        model.decoder.resize_token_embeddings(len(tokenizer))

    # save all
    model_dir = os.path.join(args.outputs_dir, args.final_name)
    os.makedirs(model_dir, exist_ok=True)
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)


if __name__ == '__main__':
    main()
