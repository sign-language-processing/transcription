"""Taken from https://huggingface.co/blog/annotated-diffusion"""

import torch


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 1)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps)**2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def get_alphas(betas: torch.Tensor):
    """
    In train time at step n, we try to predict y from x_n = ((1-b_n)*x + b_n*y)
    In test time, we want to know how much to add to x for it to represent the next step's input.
    x_{n-1} + a_n*(y - a_{n-1}) = x_n
    (1-b_{n-1})*x + b_{n-1}*y + a_n*(y - ((1-b_{n-1})*x + b_{n-1}*y)) = (1-b_n)*x + b_n*y
    x - b_{n-1}*x + a_n*(1-b_{n-1})*(y-x) = x - b_n*x + b_n*y - b_{n-1}*y
    a_n*(1-b_{n-1})*(y-x) = - b_n*x + b_n*y - b_{n-1}*y + b_{n-1}*x
    a_n*(1-b_{n-1})*(y-x) = (b_n - b_{n-1})*(y-x)
    a_n = (b_n - b_{n-1}) / (1-b_{n-1})
    """
    alphas = []
    prev_beta = 0
    for beta in betas:
        alpha = (beta - prev_beta) / (1 - prev_beta)
        alphas.append(alpha)
        prev_beta = beta
    return torch.tensor(alphas)
