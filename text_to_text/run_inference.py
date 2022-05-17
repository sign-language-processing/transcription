from transformers import pipeline

from ._tokenizers.encoder_decoder_tokenizer import EncoderDecoderTokenizer

model_path = 'outputs/checkpoint-7000'

print("Loading model", model_path)
tokenizer = EncoderDecoderTokenizer.from_pretrained(model_path)
pipe = pipeline('translation', model_path, tokenizer=tokenizer)

print("Translating")
for text in ["In the beginning was the Word, and the Word was with God, and the Word was God.", "hello world", "Amit", "amit", "John", "joey", "String", "string", "raw", "123", "12345"]:
    # TO SignWriting-American-ASE

    # print(text, pipe('<SW> <us> <ase> ' + text)[0]['translation_text'])

    output = pipe('<SW> <us> <en> ' + text, return_tensors=True)
    ids = output[0]['translation_token_ids']['output_ids'][0][0][1:-1]
    with tokenizer.as_target_tokenizer():
        print(text, "\t", tokenizer.decode(ids))
