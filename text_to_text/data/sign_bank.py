from pathlib import Path

import tensorflow_datasets as tfds
# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets
from tqdm import tqdm

signbank = tfds.load(name='sign_bank')

current_dir = Path(__file__).parent
raw_dir = current_dir.joinpath('raw').joinpath('sign_bank')
raw_dir.mkdir(exist_ok=True)

puddle_dirs = {}

for datum in tqdm(signbank["train"]):
    puddle = str(int(datum['puddle'].numpy()))
    if puddle not in puddle_dirs:
        puddle_dir = raw_dir.joinpath(puddle)
        puddle_dir.mkdir(exist_ok=True)
        puddle_dirs[puddle] = [
            open(puddle_dir.joinpath("spoken.txt"), "w", encoding="utf-8"),
            open(puddle_dir.joinpath("signed.txt"), "w", encoding="utf-8")
        ]

    spoken_language = datum['assumed_spoken_language_code'].numpy().decode('utf-8')
    country = datum['country_code'].numpy().decode('utf-8')
    sign_writing = datum['sign_writing'].numpy().decode('utf-8')
    terms = [t.numpy().decode('utf-8') for t in datum['terms']]
    spoken_text = " / ".join(terms)

    spoken_f, signed_f = puddle_dirs[puddle]
    spoken_f.write(spoken_language + " " + spoken_text.replace('\n', ' ') + "\n")
    signed_f.write("<SW> <" + country + "> | " + sign_writing + "\n")

for f1, f2 in puddle_dirs.values():
    f1.close()
    f2.close()
