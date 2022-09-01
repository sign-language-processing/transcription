import torch
from joeynmt.batch import Batch


class SignBatch(Batch):

    def __init__(self, device: torch.device, **kwargs):
        super().__init__(device=device, **kwargs)
        self.src_mask = self._pose_mask(device)

    def _pose_mask(self, device: torch.device) -> torch.Tensor:
        max_len = self.src_length.max().item()
        mask = torch.arange(max_len, device=device)[None, :] < self.src_length[:, None]
        mask = torch.unsqueeze(mask, dim=1)
        return mask
