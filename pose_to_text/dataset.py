import logging
from functools import partial
from itertools import chain
from typing import List, Tuple

import torch
from joeynmt.constants import BOS_TOKEN, EOS_TOKEN, PAD_ID, PAD_TOKEN, UNK_TOKEN
from joeynmt.datasets import make_data_iter
from joeynmt.vocabulary import Vocabulary
from torch.utils.data import DataLoader, Dataset

from pose_to_text.batch import SignBatch
from shared.collator import collate_tensors
from shared.tokenizers import SignLanguageTokenizer
from text_to_pose.data import TextPoseDataset
from text_to_pose.data import get_dataset as get_single_dataset

logger = logging.getLogger(__name__)
CPU_DEVICE = torch.device("cpu")


class PoseTextDataset(Dataset):

    def __init__(self, dataset: TextPoseDataset, split: str):
        self.dataset = dataset
        self.split = split

        special_tokens = {
            "init_token": BOS_TOKEN,
            "eos_token": EOS_TOKEN,
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN
        }
        self.trg_lang = "signed"
        self.src_lang = "poses"

        self.tokenizer = {
            self.src_lang: None,
            self.trg_lang: SignLanguageTokenizer(**special_tokens),
        }
        self.trg_vocab = Vocabulary(self.tokenizer[self.trg_lang].vocab())

        self.sequence_encoder = {
            self.src_lang: lambda x: x,
            self.trg_lang: lambda x: x,
        }
        self.has_trg = True
        self.random_subset = 0

        # For bleu calculation
        self.trg = [
            " ".join(self.tokenizer[self.trg_lang].text_to_tokens(datum["text"])) for datum in self.dataset.data
        ]
        # For compatibility with Joey
        self.src = ["" for _ in self.dataset.data]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        datum = self.dataset[idx]
        src = datum["pose"]["data"]
        trg = self.tokenizer[self.trg_lang].tokenize(datum["text"], bos=True, eos=True)
        trg = torch.tensor(trg, dtype=torch.long)
        return src, trg

    def get_item(self, idx: int, lang: str) -> List[str]:
        """
        seek one src/trg item of given index.
            - tokenization is applied here.
            - length-filtering, bpe-dropout etc also triggered if self.split == "train"
        """
        if lang == self.src_lang:
            return []

        return self.trg[idx].split(" ")

    def collate_fn(self,
                   batch: List[Tuple[torch.Tensor, torch.Tensor]],
                   pad_index: int = PAD_ID,
                   device: torch.device = CPU_DEVICE,
                   has_trg: bool = True,
                   is_train: bool = True) -> SignBatch:
        src, trg = zip(*batch)
        src_length = [len(s) for s in src]
        trg_length = [len(s) for s in trg]

        return SignBatch(
            src=collate_tensors(src),
            src_length=collate_tensors(src_length),
            trg=collate_tensors(trg, pad_value=pad_index),
            trg_length=collate_tensors(trg_length),
            device=device,
            pad_index=pad_index,
            has_trg=has_trg,
            is_train=is_train,
        )

    # TODO remove once this is the default in JoeyNMT
    def make_iter(self, pad_index: int = PAD_ID, device: torch.device = CPU_DEVICE, **kwargs) -> DataLoader:
        data_loader = make_data_iter(self, pad_index=pad_index, device=device, **kwargs)
        data_loader.collate_fn = partial(
            self.collate_fn,
            pad_index=pad_index,
            device=device,
            has_trg=True,
            is_train=self.split == "train",
        )
        return data_loader


def get_dataset(split_name="train", **kwargs):
    datasets = [
        get_single_dataset(name="dicta_sign", **kwargs),
        # get_single_dataset(name="sign2mint", **kwargs)
    ]

    all_data = list(chain.from_iterable([d.data for d in datasets]))
    return PoseTextDataset(TextPoseDataset(all_data), split=split_name)
