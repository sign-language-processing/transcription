from itertools import chain

from joeynmt.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from joeynmt.vocabulary import Vocabulary
from torch.utils.data import Dataset

from shared.tokenizers import SignLanguageTokenizer
from text_to_pose.data import TextPoseDataset
from text_to_pose.data import get_dataset as get_single_dataset


class PoseTextDataset(Dataset):
    def __init__(self, dataset: TextPoseDataset, split: str):
        self.dataset = dataset
        self.split = split

        special_tokens = {"init_token": BOS_TOKEN, "eos_token": EOS_TOKEN,
                          "pad_token": PAD_TOKEN, "unk_token": UNK_TOKEN}
        self.tokenizer = SignLanguageTokenizer(**special_tokens)
        self.trg_vocab = Vocabulary(self.tokenizer.vocab())

    def __getitem__(self, idx) -> dict:
        datum = self.dataset[idx]
        trg_tokens = self.tokenizer.tokenize(datum["text"], bos=True, eos=True)
        return {
            "src": datum["pose"]["data"],
            "src_length": datum["pose"]["length"],
            "trg": trg_tokens,
            "trg_length": len(trg_tokens)
        }


def get_dataset(split_name="train", **kwargs):
    datasets = [
        get_single_dataset(name="dicta_sign", **kwargs),
        # get_single_dataset(name="sign2mint", **kwargs)
    ]

    all_data = list(chain.from_iterable([d.data for d in datasets]))
    return PoseTextDataset(TextPoseDataset(all_data), split=split_name)
