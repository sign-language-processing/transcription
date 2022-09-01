from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from ..shared.models.pose_encoder import PoseEncoderModel


def masked_loss(loss_type, pose: torch.Tensor, pose_hat: torch.Tensor, confidence: torch.Tensor):
    # Loss by confidence. If missing joint, no loss. If less likely joint, less gradients.
    if loss_type == 'l1':
        error = torch.abs(pose - pose_hat).sum(-1)
    elif loss_type == 'l2':
        error = torch.pow(pose - pose_hat, 2).sum(-1)
    else:
        raise NotImplementedError()
    return (error * confidence).mean()


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


class IterativeTextGuidedPoseGenerationModel(pl.LightningModule):

    def __init__(self,
                 tokenizer,
                 pose_dims: (int, int) = (137, 2),
                 hidden_dim: int = 128,
                 text_encoder_depth=2,
                 pose_encoder_depth=4,
                 encoder_heads=2,
                 encoder_dim_feedforward=2048,
                 max_seq_size: int = 1000,
                 loss_type='l1'):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_seq_size = max_seq_size

        # Embedding layers
        self.positional_embeddings = nn.Embedding(num_embeddings=max_seq_size, embedding_dim=hidden_dim)

        self.embedding = nn.Embedding(
            num_embeddings=len(tokenizer),
            embedding_dim=hidden_dim,
            padding_idx=tokenizer.pad_token_id,
        )

        self.pose_encoder = PoseEncoderModel(pose_dims=pose_dims,
                                             hidden_dim=hidden_dim,
                                             encoder_depth=pose_encoder_depth,
                                             encoder_heads=encoder_heads,
                                             encoder_dim_feedforward=encoder_dim_feedforward,
                                             max_seq_size=max_seq_size,
                                             dropout=0)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                   nhead=encoder_heads,
                                                   dim_feedforward=encoder_dim_feedforward,
                                                   batch_first=True)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=text_encoder_depth)

        # Predict sequence length
        self.seq_length = DistributionPredictionModel(hidden_dim)

        # Predict pose difference
        self.pose_diff_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.pose_encoder.pose_dim),
        )

        # Loss
        self.loss_type = loss_type

    def encode_text(self, texts: List[str]):
        tokenized = self.tokenizer(texts, device=self.device)
        positional_embedding = self.positional_embeddings(tokenized["positions"])
        embedding = self.embedding(tokenized["tokens_ids"]) + positional_embedding
        encoded = self.text_encoder(embedding, src_key_padding_mask=tokenized["attention_mask"])
        seq_length = self.seq_length(torch.mean(encoded, dim=1))
        return {"data": encoded, "mask": tokenized["attention_mask"]}, seq_length

    def refine_pose_sequence(self, pose_sequence, text_encoding):
        batch_size, seq_length, _, _ = pose_sequence["data"].shape
        pose_encoding = self.pose_encoder(pose=pose_sequence, additional_sequence=text_encoding)
        pose_encoding = pose_encoding[:, :seq_length, :]

        # Predict desired change
        flat_pose_projection = self.pose_diff_projection(pose_encoding)
        return flat_pose_projection.reshape(batch_size, seq_length, *self.pose_encoder.pose_dims)

    def forward(self, text: str, first_pose: torch.Tensor, step_size: float = 0.5):
        text_encoding, sequence_length = self.encode_text([text])
        sequence_length = round(float(sequence_length))

        pose_sequence = {
            "data": first_pose.expand(1, sequence_length, *self.pose_encoder.pose_dims),
            "mask": torch.zeros([1, sequence_length], dtype=torch.bool),
        }
        while True:
            yield pose_sequence["data"][0]

            step = self.refine_pose_sequence(pose_sequence, text_encoding)
            pose_sequence["data"] = pose_sequence["data"] + step_size * step

    def training_step(self, batch, *unused_args, steps=100):
        return self.step(batch, *unused_args, steps=steps, name="train")

    def validation_step(self, batch, *unused_args, steps=100):
        return self.step(batch, *unused_args, steps=steps, name="validation")

    def step(self, batch, *unused_args, steps: int, name: str):
        text_encoding, sequence_length = self.encode_text(batch["text"])
        pose = batch["pose"]

        # Calculate sequence length loss
        sequence_length_loss = F.mse_loss(sequence_length, pose["length"]) / 10000

        # Repeat the first frame for initial prediction
        batch_size, pose_seq_length, _, _ = pose["data"].shape
        pose_sequence = {
            "data": torch.stack([pose["data"][:, 0]] * pose_seq_length, dim=1),
            "mask": torch.logical_not(pose["inverse_mask"])
        }

        refinement_loss = 0
        for _ in range(steps):
            pose_sequence["data"] = pose_sequence["data"].detach()  # Detach from graph
            l1_gold = pose["data"] - pose_sequence["data"]
            l1_predicted = self.refine_pose_sequence(pose_sequence, text_encoding)
            refinement_loss += masked_loss(self.loss_type, l1_gold, l1_predicted, confidence=pose["confidence"])

            step_size = 1 / steps
            l1_step = l1_gold if name == "validation" else l1_predicted
            pose_sequence["data"] = pose_sequence["data"] + step_size * l1_step

            if name == "train":  # add just a little noise while training
                pose_sequence["data"] = pose_sequence["data"] + torch.randn_like(pose_sequence["data"]) * 1e-4

        self.log(name + "_seq_length_loss", sequence_length_loss, batch_size=batch_size)
        self.log(name + "_refinement_loss", refinement_loss, batch_size=batch_size)
        loss = refinement_loss + sequence_length_loss
        self.log(name + "_loss", loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
