import itertools
import os
import pathlib

from tqdm import tqdm

import pandas as pd


def load_data():
    """
    Reads all raw files and returns an iterator of tuples (spoken, signed)
    """

    raw_dir = pathlib.Path(__file__).parent.joinpath('raw')

    for match in pathlib.Path(raw_dir).glob("**/*"):
        if "sign_bank" not in str(match):
            continue

        if match.is_dir():
            print(str(match))

            children = [match.joinpath(f) for f in os.listdir(match)]
            if all((c.is_file() for c in children)):
                spoken_files = [c for c in children if c.name.startswith('spoken')]
                signed_files = [c for c in children if c.name.startswith('signed')] #  and "us-ase" in c.name
                print("spoken", len(spoken_files), "signed", len(signed_files))

                for spoken in tqdm(spoken_files):
                    with open(spoken, "r", encoding="utf-8") as spoken_f:
                        spoken_lines = spoken_f.read().splitlines()
                    for signed in signed_files:
                        with open(signed, "r", encoding="utf-8") as signed_f:
                            signed_lines = signed_f.read().splitlines()
                        for sp, si in zip(spoken_lines, signed_lines):
                            if sp != "" and si != "":
                                yield sp, si


if __name__ == "__main__":
    total = 0
    all_data = []
    sw_data = ((sp, si) for sp, si in load_data() if si.startswith("<SW>"))
    for sp, si in sw_data:
        tgt_params, tgt = si.split(" | ")
        src = tgt_params + " " + sp

        if len(tgt.strip()) == 0:
            continue

        total += 1

        all_data.append({
            "src": src.strip(),
            "tgt": tgt.strip()
        })

    print("total", total)
    print(all_data[0])
    pd.DataFrame(all_data).to_csv('data.csv.gz', compression='gzip')

    #
    #
    # # O(1) memory total
    # print("total", sum(1 for x in load_data()))