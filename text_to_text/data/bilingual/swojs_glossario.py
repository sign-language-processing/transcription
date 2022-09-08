from pathlib import Path

# pylint: disable=unused-import
import sign_language_datasets.datasets
import tensorflow_datasets as tfds
from sign_language_datasets.datasets.config import SignDatasetConfig

config = SignDatasetConfig(name="only-annotations", version="1.0.0", include_video=False)
swojs_glossario = tfds.load(name='swojs_glossario', builder_kwargs={"config": config})

current_dir = Path(__file__).parent
raw_dir = current_dir.joinpath('raw').joinpath('swojs_glossario')
raw_dir.mkdir(exist_ok=True, parents=True)


def decode(tl):
    return list(map(lambda t: t.decode('utf-8'), tl.numpy()))


spoken_f = open(raw_dir.joinpath("spoken.txt"), "w", encoding="utf-8")
signed_f = open(raw_dir.joinpath("signed.txt"), "w", encoding="utf-8")

for datum in swojs_glossario["train"]:
    spoken_languages = decode(datum['spoken_language'])
    signed_languages = decode(datum['signed_language'])

    spoken_text = datum['title'].numpy().decode('utf-8').strip()
    signed_text = " ".join(decode(datum['sign_writing'])).strip()

    spoken_f.write(spoken_text + "\n")
    signed_f.write(f"$SW$ | {signed_text}\n")

spoken_f.close()
signed_f.close()
