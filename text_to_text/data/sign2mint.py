from pathlib import Path

import tensorflow_datasets as tfds
# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig

import itertools

current_dir = Path(__file__).parent
raw_dir = current_dir.joinpath('raw').joinpath('sign2mint')
raw_dir.mkdir(exist_ok=True, parents=True)

config = SignDatasetConfig(name="only-annotations", version="1.0.0", include_video=False)
sign2mint = tfds.load(name='sign2_mint', builder_kwargs={"config": config})

spoken = open(raw_dir.joinpath("spoken.txt"), "w", encoding="utf-8")
signed = open(raw_dir.joinpath("signed.txt"), "w", encoding="utf-8")

for datum in sign2mint["train"]:
    spoken_text = datum['fachbegriff'].numpy().decode('utf-8')
    signed_text = datum['gebaerdenschrift']['fsw'].numpy().decode('utf-8')
    spoken.write("<de> " + spoken_text + "\n")
    signed.write("<SW> <de> <gsg> | " + signed_text)

spoken.close()
signed.close()
