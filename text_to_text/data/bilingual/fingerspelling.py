import os
import random
import re
import string
from pathlib import Path

import numpy as np

from shared.signwriting.signwriting import join_signs

current_dir = Path(__file__).parent
wordslist_dir = current_dir.joinpath('wordslist')
fingerspelling_dir = current_dir.joinpath('fingerspelling')

raw_dir = current_dir.joinpath('raw').joinpath('fingerspelling')
raw_dir.mkdir(exist_ok=True, parents=True)

if not wordslist_dir.exists():
    os.system('git clone https://github.com/imsky/wordlists.git ' + str(wordslist_dir.absolute()))

# name_files = wordslist_dir.joinpath('names').glob("**/*.txt")
name_files = wordslist_dir.glob("**/*.txt")
names = []
for name_file in name_files:
    with open(name_file, "r", encoding="utf-8") as f:
        names += f.read().splitlines()

names = list(string.ascii_lowercase + string.digits) + [n.strip() for n in names if len(n.strip()) > 0]

# Sample numbers
samples = {str(n) for n in np.power(10, np.random.exponential(3, 10000)).astype(np.int32)}
print("Sampled Numbers", len(samples))
print("Words:", len(names))
names += list(samples)

with open(raw_dir.joinpath("spoken.txt"), "w", encoding="utf-8") as spoken_f:
    for name in names:
        spoken_f.write(name + "\n")

for f_name in fingerspelling_dir.iterdir():
    if f_name.name == "README.md":
        continue

    spoken, country, iana, local_name = f_name.name.split(".")[0].split("-")
    with open(f_name, "r", encoding="utf-8") as f:
        content = re.sub(r'#.*$', '', f.read())  # Remove comments
        lines = [line.strip().split(",") for line in content.splitlines() if len(line.strip()) > 0]
        chars = {first.lower(): others for [first, *others] in lines}

    signed_f = open(raw_dir.joinpath(f"signed.{country}-{iana}"), "w", encoding="utf-8")

    counter = 0
    for name in names:
        sl = []
        caret = 0
        while caret < len(name):
            found = False
            for c, options in chars.items():
                if name[caret:caret + len(c)].lower() == c:
                    sl.append(random.choice(options))
                    caret += len(c)
                    found = True
                    break
            if not found:
                break

        if caret == len(name):
            counter += 1
            signed_f.write(f"$SW$ ${country}$ ${iana}$ | {join_signs(*sl, spacing=10)}\n")
        else:
            signed_f.write("\n")

    signed_f.close()

    print(f_name, counter)
