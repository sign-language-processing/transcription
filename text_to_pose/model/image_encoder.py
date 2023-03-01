import torch
from torch import nn


class ImageEncoderModel(nn.Module):

    def __init__(self):
        super().__init__()
        raise NotImplementedError()

    def forward(self, images: torch.Tensor):
        raise NotImplementedError()
