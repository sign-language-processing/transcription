import logging
from itertools import chain
from typing import List, Tuple

import torch
from joeynmt.constants import BOS_TOKEN, EOS_TOKEN, PAD_ID, PAD_TOKEN, UNK_TOKEN
from joeynmt.datasets import BaseDataset
from joeynmt.vocabulary import Vocabulary

from pose_to_text.batch import SignBatch
from text_to_pose.data import TextPoseDataset
from text_to_pose.data import get_dataset as get_single_dataset

from .._shared.collator import collate_tensors
from .._shared.tokenizers import SignLanguageTokenizer

logger = logging.getLogger(__name__)
CPU_DEVICE = torch.device("cpu")


class PoseTextDataset(BaseDataset):

    #     def __init__(
    #         self,
    #         path: str,
    #         src_lang: str,
    #         trg_lang: str,
    #         split: int = "train",
    #         has_trg: bool = True,
    #         tokenizer: Dict[str, BasicTokenizer] = None,
    #         sequence_encoder: Dict[str, Callable] = None,
    #         random_subset: int = -1,
    #     ):
    #         self.path = path
    #         self.src_lang = src_lang
    #         self.trg_lang = trg_lang
    #         self.has_trg = has_trg
    #         self.split = split
    #         if self.split == "train":
    #             assert self.has_trg
    #
    #         _place_holder = {self.src_lang: None, self.trg_lang: None}
    #         self.tokenizer = _place_holder if tokenizer is None else tokenizer
    #         self.sequence_encoder = (_place_holder
    #                                  if sequence_encoder is None else sequence_encoder)
    #
    #         # for ransom subsampling
    #         self.random_subset =

    def __init__(self, dataset: TextPoseDataset, split: str, has_trg: bool = True, random_subset=0):
        trg_lang = "signed"
        src_lang = "poses"

        special_tokens = {
            "init_token": BOS_TOKEN,
            "eos_token": EOS_TOKEN,
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN
        }

        super().__init__(path=None,
                         src_lang=src_lang,
                         trg_lang=trg_lang,
                         has_trg=has_trg,
                         split=split,
                         random_subset=random_subset,
                         tokenizer={
                             src_lang: None,
                             trg_lang: SignLanguageTokenizer(**special_tokens),
                         },
                         sequence_encoder={
                             src_lang: lambda x: x,
                             trg_lang: lambda x: x,
                         })

        self.dataset = dataset

        # Model needs to know how many classes for softmax
        self.trg_vocab = Vocabulary(self.tokenizer[self.trg_lang].vocab())

    def __len__(self):
        return len(self.dataset)

    @property
    def src(self) -> List[str]:
        """get detokenized preprocessed data in src language."""
        # For compatibility with Joey
        return ["" for _ in self.dataset.data]

    @property
    def trg(self) -> List[str]:
        """get detokenized preprocessed data in trg language."""
        # For bleu calculation
        return [" ".join(self.tokenizer[self.trg_lang].text_to_tokens(datum["text"])) for datum in self.dataset.data]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        datum = self.dataset[idx]
        src = datum["pose"]["data"]
        trg = self.tokenizer[self.trg_lang].tokenize(datum["text"], bos=True, eos=True)
        trg = torch.tensor(trg, dtype=torch.long)
        return src, trg

    # def get_item(self, idx: int, lang: str) -> List[str]:
    #     """
    #     seek one src/trg item of given index.
    #         - tokenization is applied here.
    #         - length-filtering, bpe-dropout etc also triggered if self.split == "train"
    #     """
    #     if lang == self.src_lang:
    #         return []
    #
    #     return self.trg[idx].split(" ")

    def collate_fn(
        self,
        batch: List[Tuple],
        pad_index: int = PAD_ID,
        device: torch.device = CPU_DEVICE,
    ) -> SignBatch:
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
            has_trg=self.has_trg,
            is_train=self.split == "train",
        )


def get_dataset(split_name="train", **kwargs):
    datasets = [
        get_single_dataset(name="dicta_sign", **kwargs),
        # get_single_dataset(name="sign2mint", **kwargs)
    ]

    all_data = list(chain.from_iterable([d.data for d in datasets]))
    return PoseTextDataset(TextPoseDataset(all_data), split=split_name)
