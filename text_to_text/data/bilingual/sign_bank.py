import re
from pathlib import Path

# pylint: disable=unused-import
import sign_language_datasets.datasets
import tensorflow_datasets as tfds
from sign_language_datasets.datasets import SignDatasetConfig
from tqdm import tqdm

signbank = tfds.load(name='sign_bank', builder_kwargs=dict(config=SignDatasetConfig(name="annotations")))

current_dir = Path(__file__).parent
raw_dir = current_dir.joinpath('raw').joinpath('sign_bank')
raw_dir.mkdir(exist_ok=True, parents=True)

puddle_dirs = {}

disallowed_sign_pattern = re.compile("[ghijklmnopqrstuvwyzCDEFGHIJKNOPQTUVWXYZ,.]")

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
    sign_writing = " / ".join([t.numpy().decode('utf-8').replace("\n", " ") for t in datum['sign_writing']])

    # For some reason, sometimes the "signed" size does not have signwriting
    if bool(re.match(disallowed_sign_pattern, sign_writing)):
        print("Skipping", sign_writing)
        continue

    terms = [t.numpy().decode('utf-8').replace("\n", " ") for t in datum['terms']]
    spoken_text = " / ".join(terms)
    spoken_text = re.sub(r'<.*?>', '', spoken_text).strip()  # Remove tags like iframes and such

    spoken_f, signed_f = puddle_dirs[puddle]
    clean_spoken_text = spoken_text.replace('\n', ' ')

    if spoken_language != "":
        clean_spoken_text = f"${spoken_language}$ | {clean_spoken_text}"

    if country != "":
        sign_writing = f"$SW$ ${country}$ | {sign_writing}"
    else:
        sign_writing = f"$SW$ | {sign_writing}"

    spoken_f.write(f"{clean_spoken_text}\n")
    signed_f.write(f"{sign_writing}\n")

for f1, f2 in puddle_dirs.values():
    f1.close()
    f2.close()
