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

        print('X has nan', torch.isnan(x).any().item())
        print('mu has nan', torch.isnan(mu).any().item())
        print('mu.weight has nan', torch.isnan(mu.weight.data).any().item())
        print('mu.bias has nan', torch.isnan(mu.bias.data).any().item())
        print('log_var has nan', torch.isnan(log_var).any().item())
        print('log_var.weight has nan', torch.isnan(log_var.weight.data).any().item())
        print('log_var.bias has nan', torch.isnan(log_var.bias.data).any().item())
        print('std has nan', torch.isnan(std).any().item())

        q = torch.distributions.Normal(mu, std)
        return q.rsample()
