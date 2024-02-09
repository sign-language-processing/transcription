import numpy as np
import torch


def masked_loss(loss_type: str,
                tensor1: torch.Tensor,
                tensor2: torch.Tensor,
                confidence: torch.Tensor,
                model_num_steps: int = 10):
    # Loss by confidence. If missing data, no loss. If less likely data, fewer gradients.
    difference = tensor1 - tensor2

    if loss_type == 'l1':
        error = torch.abs(difference).sum(-1)
    elif loss_type == 'l2':
        error = torch.pow(difference, 2).sum(-1)
    else:
        raise NotImplementedError()

    # normalization of the loss (Section 5.4)
    num_steps_norm = np.log(model_num_steps)**2 if model_num_steps != 1 else 1

    return (error * confidence).mean() * num_steps_norm
