import torch
from joeynmt.batch import Batch


class SignBatch(Batch):
    def __init__(self, src_length, **kwargs):
        super().__init__(src_length=src_length, **kwargs)
        self.src_mask = self._pose_mask(src_length)

    def _pose_mask(self, pose_length: torch.Tensor) -> torch.Tensor:
        max_len = pose_length.max().item()
        return torch.arange(max_len)[None, :] < pose_length[:, None]
