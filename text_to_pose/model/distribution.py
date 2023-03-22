import torch
from torch import nn


class DistributionPredictionModel(nn.Module):

    def __init__(self, input_size: int):
        super().__init__()

        self.fc_mu = nn.Linear(input_size, 1)
        self.fc_var = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor):
        mu = self.fc_mu(x)
        if not self.training:  # In test time, just predict the mean
            return mu

        log_var = self.fc_var(x)
        # sample z from q
        std = torch.exp(log_var / 2)

        q = torch.distributions.Normal(mu, std)
        return q.rsample()
