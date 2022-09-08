import re
from collections import Counter
from pathlib import Path

# pylint: disable=unused-import
import sign_language_datasets.datasets
import tensorflow_datasets as tfds
from sign_language_datasets.datasets.config import SignDatasetConfig
from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences

current_dir = Path(__file__).parent
raw_dir = current_dir.joinpath('raw').joinpath('dgs_corpus')
raw_dir.mkdir(exist_ok=True, parents=True)

config = SignDatasetConfig(name="texts-new",
                           version="3.0.0",
                           include_video=False,
                           process_video=False,
                           include_pose=None)
dgs_types = tfds.load(name='dgs_types', builder_kwargs={"config": config})
dgs_corpus = tfds.load(name='dgs_corpus', builder_kwargs={"config": config})

# Map glosses to hamnosys
gloss_hamnosys_map = {}


def decode_str(s):
    return s.numpy().decode('utf-8')


for datum in dgs_types["train"]:
    hamnosys = decode_str(datum['hamnosys'])
    if len(hamnosys) > 0:
        glosses = [decode_str(g) for g in datum["glosses"]]
        for g in glosses:
            gloss_hamnosys_map[g] = hamnosys

spoken = open(raw_dir.joinpath("spoken.txt"), "w", encoding="utf-8")
signed = open(raw_dir.joinpath("signed.txt"), "w", encoding="utf-8")

# for start, treat this as a dictionary
for gloss, hamnosys in gloss_hamnosys_map.items():
    clean_text = re.compile(r'\d.*').sub('', gloss)
    spoken.write(f"$de$ | {clean_text.capitalize()}\n")
    signed.write(f"$HNS$ $de$ $gsg$ | {hamnosys}\n")

bad_glosses_counter = Counter()

for datum in dgs_corpus["train"]:
    elan_path = datum["paths"]["eaf"].numpy().decode('utf-8')
    sentences = get_elan_sentences(elan_path)

    for sentence in sentences:
        spoken_text = sentence["german"]
        glosses = [s["gloss"].replace("*", "") for s in sentence["glosses"]]

        bad_glosses = [g for g in glosses if g not in gloss_hamnosys_map]
        if len(bad_glosses) == 0:
            hamnosys = [gloss_hamnosys_map[g] for g in glosses]
            signed_text = " ".join(hamnosys)

            spoken.write(f"$de$ | {spoken_text}\n")
            signed.write(f"$HNS$ $de$ $gsg$ | {signed_text}\n")
        else:
            for bad_gloss in bad_glosses:
                bad_glosses_counter[bad_gloss] += 1

print(bad_glosses_counter.most_common(50))

spoken.close()
signed.close()
