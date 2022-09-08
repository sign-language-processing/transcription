from itertools import chain
from pathlib import Path

# pylint: disable=unused-import
import sign_language_datasets.datasets
import tensorflow_datasets as tfds
from sign_language_datasets.datasets.config import SignDatasetConfig

COUNTRIES = {"BSL": "uk", "DGS": "de", "LSF": "fr", "GSL": "gr"}
IANA = {"BSL": "bfi", "DGS": "gsg", "LSF": "fsl", "GSL": "gss"}

config = SignDatasetConfig(name="only-annotations", version="1.0.0", include_video=False, include_pose=None)
dicta_sign = tfds.load(name='dicta_sign', builder_kwargs={"config": config})

current_dir = Path(__file__).parent
raw_dir = current_dir.joinpath('raw').joinpath('dicta_sign')
raw_dir.mkdir(exist_ok=True, parents=True)

country_dirs = {}
country_dicts = {}
for country in COUNTRIES.values():
    country_dirs[country] = [
        open(raw_dir.joinpath("spoken." + country), "w", encoding="utf-8"),
        open(raw_dir.joinpath("signed." + country), "w", encoding="utf-8")
    ]
    country_dicts[country] = {}

for datum in dicta_sign["train"]:
    _id = datum['id'].numpy().decode('utf-8')
    spoken_language = datum['spoken_language'].numpy().decode('utf-8')
    signed_language = datum['signed_language'].numpy().decode('utf-8')
    country = COUNTRIES[signed_language]
    iana = IANA[signed_language]
    spoken_text = datum['text'].numpy().decode('utf-8')
    signed_text = datum['hamnosys'].numpy().decode('utf-8')

    spoken_f, signed_f = country_dirs[country]

    [sample_index, _] = _id.split('_')
    country_dicts[country][sample_index] = (f"${spoken_language}$ | {spoken_text}",
                                            f"$HNS$ ${country}$ ${iana}$ | {signed_text}")

all_ids = sorted(list(set(chain.from_iterable((c.keys() for c in country_dicts.values())))))

# Write to files
for k, (f1, f2) in country_dirs.items():
    for _id in all_ids:
        if _id in country_dicts[k]:
            spoken, signed = country_dicts[k][_id]
        else:
            spoken = signed = ''
        f1.write(spoken + '\n')
        f2.write(signed + '\n')

    f1.close()
    f2.close()
