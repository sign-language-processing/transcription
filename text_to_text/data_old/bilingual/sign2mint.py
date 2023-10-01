from pathlib import Path

# pylint: disable=unused-import
import sign_language_datasets.datasets
import tensorflow_datasets as tfds
from sign_language_datasets.datasets.config import SignDatasetConfig

current_dir = Path(__file__).parent
raw_dir = current_dir.joinpath('raw').joinpath('sign2mint')
raw_dir.mkdir(exist_ok=True, parents=True)

config = SignDatasetConfig(name="only-annotations", version="1.0.0", include_video=False)
sign2mint = tfds.load(name='sign2_mint', builder_kwargs={"config": config})

spoken = open(raw_dir.joinpath("spoken.txt"), "w", encoding="utf-8")
signed = open(raw_dir.joinpath("signed.txt"), "w", encoding="utf-8")

for datum in sign2mint["train"]:
    spoken_text = datum['fachbegriff'].numpy().decode('utf-8').strip()
    signed_text = datum['gebaerdenschrift']['fsw'].numpy().decode('utf-8').strip()
    spoken.write(f"$de$ | {spoken_text}\n")
    signed.write(f"$SW$ $de$ $gsg$ | {signed_text}\n")

spoken.close()
signed.close()
