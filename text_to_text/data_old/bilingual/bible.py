import os
import re
from collections import defaultdict
from itertools import chain
from pathlib import Path

from tqdm import tqdm

BOOKS = {
    'Genesis': 'GEN',
    'Exodus': 'EXO',
    'Leviticus': 'LEV',
    'Numbers': 'NUM',
    'Deuteronomy': 'DEU',
    'Joshua': 'JOS',
    'Judges': 'JDG',
    'Ruth': 'RUT',
    '1 Samuel': '1SA',
    '2 Samuel': '2SA',
    '1 Kings': '1KI',
    '2 Kings': '2KI',
    '1 Chronicles': '1CH',
    '2 Chronicles': '2CH',
    'Ezra': 'EZR',
    'Nehemiah': 'NEH',
    'Esther': 'EST',
    'Job': 'JOB',
    'Psalms': 'PSA',
    'Psalm': 'PSA',
    'Proverbs': 'PRO',
    'Proverb': 'PRO',
    'Provrerbs': 'PRO',
    'Provertbs': 'PRO',
    'Ecclesiastes': 'ECC',
    'Song of Solomon': 'SON',
    'Isaiah': 'ISA',
    'Jeremiah': 'JER',
    'Lamentations': 'LAM',
    'Lamentaions': 'LAM',
    'Ezekiel': 'EZE',
    'Daniel': 'DAN',
    'Hosea': 'HOS',
    'Joel': 'JOE',
    'Amos': 'AMO',
    'Obadiah': 'OBA',
    'Jonah': 'JON',
    'Micah': 'MIC',
    'Nahum': 'NAH',
    'Habakkuk': 'HAB',
    'Zephaniah': 'ZEP',
    'Haggai': 'HAG',
    'Zechariah': 'ZEC',
    'Malachi': 'MAL',
    'Matthew': 'MAT',
    'Mark': 'MAR',
    'Luke': 'LUK',
    'John': 'JOH',
    'Acts': 'ACT',
    'Romans': 'ROM',
    '1 Corinthians': '1CO',
    '1Cor': '1CO',
    '1 Corinthinans': '1CO',
    '1 Corinthinan': '1CO',
    '2 Corinthians': '2CO',
    '2Co': '2CO',
    'Galatians': 'GAL',
    'Ephesians': 'EPH',
    'Philippians': 'PHI',
    'Colossians': 'COL',
    '1 Thessalonians': '1TH',
    '1 Thessolonians': '1TH',
    '2 Thessalonians': '2TH',
    '1 Timothy': '1TI',
    '1Timothy': '1TI',
    '2 Timothy': '2TI',
    'Titus': 'TIT',
    'Philemon': 'PHM',
    'Hebrews': 'HEB',
    'James': 'JAM',
    '1 Peter': '1PE',
    '1Pe': '1PE',
    '2 Peter': '2PE',
    '1 John': '1JO',
    '1 Joh': '1JO',
    '2 John': '2JO',
    '3 John': '3JO',
    'Jude': 'JUD',
    'Revelation': 'REV'
}

for b, c in list(BOOKS.items()):
    no_space = b.replace(" ", "")
    BOOKS[no_space] = c
    split = b.split(" ")

current_dir = Path(__file__).parent
bible_corpus_dir = current_dir.joinpath('bible_corpus')

raw_dir = current_dir.joinpath('raw').joinpath('bible')
raw_dir.mkdir(exist_ok=True, parents=True)

if not bible_corpus_dir.exists():
    os.system('git clone https://github.com/christos-c/bible-corpus.git ' + str(bible_corpus_dir.absolute()))

bibles = bible_corpus_dir.joinpath('bibles')

verse_dict = defaultdict(dict)

for bible_file in tqdm(bibles.iterdir()):
    with open(bible_file, "r", encoding="utf-8") as f:
        BIBLE_STR = str(f.read())
        languages = re.findall(r'language id=\"(.*?)\"', BIBLE_STR)
        if len(languages) > 0:
            language = languages[0]
            matches = re.findall(r'\<seg id=[\"\'](.*?)[\"\'] type=[\"\']verse[\"\']>([\s\S]*?)<\/seg>', BIBLE_STR)
            for match in matches:
                verse_dict[language][match[0]] = match[1].strip()

sign_verse_dict = defaultdict(lambda: defaultdict(lambda: ''))

verses = 0
for asl_bible_id in [151, 152]:
    errors = {"book": 0, "match": 0, "multi-verse": 0}

    corpus = current_dir.joinpath('raw').joinpath('sign_bank').joinpath(str(asl_bible_id))
    with open(corpus.joinpath('spoken.txt'), "r", encoding="utf-8") as f:
        spoken_texts = f.read().splitlines()
    with open(corpus.joinpath('signed.txt'), "r", encoding="utf-8") as f:
        signed_texts = f.read().splitlines()

    # Sort the verses because a verse may be split into multiple parts
    for spoken, signed in sorted(zip(spoken_texts, signed_texts), key=lambda x: x[0]):
        spoken = spoken[7:]
        if spoken.startswith("Title"):
            spoken = spoken[len("Title"):]
        spoken = spoken.strip()

        books = [book for book in sorted(BOOKS.keys(), key=len) if spoken.lower().startswith(book.lower())]
        if len(books) > 0:
            book = books[-1]
            spoken = spoken[len(book):].strip()
            # Pattern 1: Just chapter and verse
            p1 = re.match(r"^(\d+)v(\d+)", spoken)
            if p1 is None:
                p1 = re.match(r"^(\d+): ?(\d+)", spoken)
            # Pattern 2: Chapter, and possibly multiple verses
            p2 = re.match(r"^(\d+), Verse (\d+)-?(\d+)?", spoken)
            if p2 is None:
                p2 = re.match(r"^(\d+)_(\d+)-?(\d+)?", spoken)

            chapter = None
            verse = None
            if p1 is not None:
                chapter = p1[1]
                verse = p1[2]
            elif p2 is not None:
                if not p2[3]:  # Does not span multiple verses
                    chapter = p2[1]
                    verse = p2[2]
                else:
                    errors["multi-verse"] += 1
            else:
                errors["match"] += 1

            if chapter is not None and verse is not None:
                verses += 1
                _id = 'b.' + BOOKS[book] + '.' + str(int(chapter)) + '.' + str(int(verse))
                sign_verse_dict['us'][_id] = signed  # Some overlaps exist
        else:
            errors["book"] += 1

    print(errors, verses, len(sign_verse_dict['us'].keys()))

all_keys = sorted(
    set(chain.from_iterable([v.keys() for v in verse_dict.values()] + [v.keys() for v in sign_verse_dict.values()])))

for lang, verses in verse_dict.items():
    text = "\n".join([(f"${lang}$ | {verses[key]}") if key in verses else "" for key in all_keys])
    with open(raw_dir.joinpath("spoken." + lang), "w", encoding="utf-8") as f:
        f.write(text)

for lang, verses in sign_verse_dict.items():
    text = "\n".join([verses[key] if key in verses else "" for key in all_keys])
    with open(raw_dir.joinpath("signed." + lang), "w", encoding="utf-8") as f:
        f.write(text)
