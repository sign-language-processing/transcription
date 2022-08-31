import logging
from functools import partial
from typing import Any, Dict, List

import torch
from joeynmt.batch import Batch
from joeynmt.constants import PAD_ID
from joeynmt.training import make_data_iter
from torch.utils.data import DataLoader

from pose_to_text.batch import SignBatch
from shared.collator.collator import zero_pad_collator

logger = logging.getLogger(__name__)
CPU_DEVICE = torch.device("cpu")


def collate_fn(
        batch: List[Dict[str, Any]],
        pad_index: int = PAD_ID,
        device: torch.device = CPU_DEVICE,
        has_trg: bool = True,
        is_train: bool = True) -> Batch:
    collated = zero_pad_collator(batch)

    return SignBatch(
        src=collated["src"],
        src_length=collated["src_length"],
        trg=collated["trg"],
        trg_length=collated["trg_length"],
        device=device,
        pad_index=pad_index,
        has_trg=has_trg,
        is_train=is_train,
    )


def get_data_iter(pad_index: int = PAD_ID,
                  device: torch.device = CPU_DEVICE,
                  **kwargs) -> DataLoader:
    data_loader = make_data_iter(**kwargs)
    dataset = data_loader.dataset

    # data iterator
    return DataLoader(
        dataset=dataset,
        batch_sampler=data_loader.batch_sampler,
        collate_fn=partial(
            collate_fn,
            pad_index=pad_index,
            device=device,
            has_trg=True,
            is_train=dataset.split == "train",
        ),
        num_workers=data_loader.num_workers,
    )
