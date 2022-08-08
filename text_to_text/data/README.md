# Data

Scripts in this directory should create arbitrary directories under the `raw` subdirectory.

Each directory should include a `signed.X` and `spoken.X` entries. And can contain multiple files, as long as they are
many-to-many aligned.

### `signed.X`
 
1. Writing system `SW` (SignWriting) or `HNS` (HamNoSys)
2. Country code for that signed language
3. (optional) the sign language ISO code.
4. The rest of the tokens, if in SignWriting should be FSW, or in HamNoSys.

### `spoken.X`

1. Spoken language code 
2. The rest of the tokens are untokenized text in that spoken language.

## Usage

To create all data, run:

```bash
python fingerspelling.py # SW Fingerspelling in multiple languages
python dicta_sign.py # HamNoSys single words
python dgs_corpus.py # HamNoSys sentences
python sign_bank.py # SW words and sentences in multiple languages 
python swojs_glossario.py # SW sentences for Brazilian sign language
python bible.py # SignBank Bible + the multilingual bible corpus
python sign2mint.py # Sign2Mint - technical terms in German sign language
```