import gzip
import json
import os
import pathlib
import random
import re
from collections import Counter, defaultdict
from random import shuffle
from typing import List


def tokenize(text):
    text = text.replace(' S', ' MS')
    text = re.sub('([SBMLR].*?)([0-9]{3})x([0-9]{3})', r'\1 \2 \3 ', text)
    text = text.replace('  ', ' ')
    return text


def detokenize(text):
    text = re.sub('(.*?) ([0-9]{3}) ([0-9]{3}) ?', r'\1\2x\3', text)
    text = re.sub(r'([BMLR])', r' \1', text)
    text = text.replace('  ', ' ')
    return text


def load_pair(spoken_files: List[str], signed_files: List[str]):
    for spoken in spoken_files:
        with open(spoken, "r", encoding="utf-8") as spoken_f:
            spoken_lines = spoken_f.read().splitlines()
        for signed in signed_files:
            with open(signed, "r", encoding="utf-8") as signed_f:
                signed_lines = signed_f.read().splitlines()

            for sp, si in zip(spoken_lines, signed_lines):
                if sp != "" and si != "":
                    yield sp, si


def load_data():
    raw_dir = pathlib.Path(__file__).parent.joinpath('bilingual', 'raw')

    for match in sorted(pathlib.Path(raw_dir).glob("**/*")):
        if match.is_dir():
            children = [match.joinpath(str(f_name)) for f_name in os.listdir(match)]
            if all((c.is_file() for c in children)):
                spoken_files = [c for c in children if c.name.startswith('spoken')]
                signed_files = [c for c in children if c.name.startswith('signed')]  # and "us-ase" in c.name
                dir_name = str(match).partition("/raw/")[2]
                yield dir_name, load_pair(spoken_files, signed_files)


CONTROL_WORDS = set()


def write_line(f, line):
    # We can't have "<" and ">" because bergamot (browser) thinks it is html
    line = line.replace('<', '$').replace('>', '$')

    # Extract control words
    for match in re.finditer(r'\$.*?\$', line):
        CONTROL_WORDS.add(match.group(0))

    f.write(line.strip() + "\n")


def build_bilingual():
    statistics = defaultdict(Counter)

    compressed_dir = pathlib.Path(__file__).parent.joinpath('compressed')

    sets = defaultdict(list)

    too_long = 0
    mono = 0
    total = 0
    data = load_data()
    for name, data in data:
        dataset_dir = compressed_dir.joinpath(name)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        data_list = list(data)
        shuffle(data_list)

        # At least 100 examples per dataset
        if len(data_list) < 100:
            continue

        for split, split_start, split_end in [('train', 0, 99.5), ('devtest', 99.5, 99.75), ('test', 99.75, 100)]:
            start_index, end_index = int(len(data_list) * split_start / 100), int(len(data_list) * split_end / 100)
            split_data = data_list[start_index:end_index]
            if len(split_data) == 0:
                continue

            sets[split].append(name)
            spoken_f = gzip.open(dataset_dir.joinpath(f'{split}.spoken.gz'), 'wt')
            signed_f = gzip.open(dataset_dir.joinpath(f'{split}.signed.gz'), 'wt')

            for sp, si in split_data:
                if sp[4:].strip() == "":
                    mono += 1
                    continue

                if sp[0] == '<' and "<" in si[1:]:
                    language = sp[1:3]
                    signed_second_tax = si[1:].index('<') + 2
                    country = si[signed_second_tax:signed_second_tax + 2]
                    statistics[country][language] += 1

                tgt_params, tgt = si.split(" | ")
                src = tgt_params + " " + sp

                if len(tgt.strip()) == 0:
                    continue

                total += 1

                # Unreasonable lengths
                if len(src) < 512 and len(tgt) < 2048:
                    write_line(spoken_f, src)
                    write_line(signed_f, tokenize(tgt))
                else:
                    too_long += 1

            spoken_f.close()
            signed_f.close()

    for split, set_names in sets.items():
        print(f"  {split}:")
        for name in set_names:
            if name != "swojs_glossario":
                print(f"    - custom-corpus_/custom_corpus/{name}/{split}")

    print("total", total)
    print("signed mono total", mono)
    print("toooooo long", too_long)

    with open("statistics.json", "w", encoding="utf-8") as f:
        json.dump(statistics, f)


def build_monolingual():
    compressed_dir = pathlib.Path(__file__).parent.joinpath('compressed')

    raw_dir = pathlib.Path(__file__).parent.joinpath('monolingual', 'raw')

    formats = ["SW"]  # , "HNS"]
    target_languages = [("us", "ase")]  # TODO add others

    for match in sorted(pathlib.Path(raw_dir).glob("**/*")):
        if match.is_dir():
            children = [match.joinpath(str(f_name)) for f_name in os.listdir(match)]
            if all((c.is_file() for c in children)):
                dataset_dir = compressed_dir.joinpath(match.name)
                dataset_dir.mkdir(parents=True, exist_ok=True)
                spoken_f = gzip.open(dataset_dir.joinpath('mono.spoken.gz'), 'wt')

                for c in children:
                    with open(c, "r", encoding="utf-8") as f:
                        for line in f.read().splitlines():
                            _format = random.choice(formats)
                            _target_language = random.choice(target_languages)
                            src = f"<{_format}> <{_target_language[0]}> <{_target_language[1]}> {line}"
                            write_line(spoken_f, src)

                print(f"- custom-mono_/custom_corpus/{match.name}/mono")


if __name__ == "__main__":
    build_bilingual()
    build_monolingual()

    print("CONTROL WORDS:")
    print(",".join(list(CONTROL_WORDS)))

# $de$,$az$,$hi$,$dsl$,$lv$,$ro$,$kbh$,$281$,$ak$,$psr$,$nsl$,$af$,$vn$,$zh-tw$,$bg$,$eu$,$cb$,$sn$,$mx$,$za$,$i$,$kr$,$SW$,$nh$,$po$,$di$,$sh$,$sgg$,$gv$,$th$,$kk$,$bfi$,$rm$,$hu$,$al$,$eg$,$gsg$,$np$,$he$,$sv$,$kb$,$gd$,$pe$,$es$,$cs$,$jk$,$cz$,$nl$,$ve$,$xh$,$tmh$,$pck$,$hds$,$tsq$,$ji$,$my$,$bn$,$eo$,$uy$,$gb$,$it$,$br$,$da$,$chr$,$gr$,$ru$,$py$,$sr$,$oj$,$cj$,$swl$,$ca$,$do$,$pp$,$cq$,$fsl$,$ise$,$ssp$,$is$,$gj$,$et$,$vi$,$cn$,$vgt$,$zu$,$tu$,$ss$,$pot$,$csl$,$HNS$,$ka$,$qc$,$uk$,$bzs$,$ac$,$ml$,$am$,$sk$,$lt$,$el$,$dk$,$mfs$,$pt$,$gss$,$jp$,$sy$,$ko$,$dj$,$en$,$tw$,$fil$,$jsl$,$ie$,$ja$,$ncs$,$mr$,$sa$,$mi$,$cp$,$mk$,$up$,$so$,$ag$,$fi$,$ni$,$pl$,$se$,$ph$,$isg$,$sq$,$tl$,$sg$,$us$,$mm$,$bs$,$ck$,$tr$,$cl$,$sfb$,$no$,$fa$,$te$,$co$,$ar$,$qu$,$mt$,$wa$,$fr$,$ew$,$id$,$ase$,$tn$,$be$,$hy$,$sl$,$wo$,$ke$,$ch$,$gn$,$zh$,$hn$
