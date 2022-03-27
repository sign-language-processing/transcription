from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np


def masked_mse_loss(pose: torch.Tensor, pose_hat: torch.Tensor, confidence: torch.Tensor):
    # Loss by confidence. If missing joint, no loss. If less likely joint, less gradients.
    sq_error = torch.pow(pose - pose_hat, 2).sum(-1)
    return (sq_error * confidence).mean()


class IterativeTextGuidedPoseGenerationModel(pl.LightningModule):
    def __init__(
            self,
            tokenizer,
            pose_dims: (int, int) = (137, 2),
            hidden_dim: int = 128,
            text_encoder_depth=2,
            pose_encoder_depth=4,
            encoder_heads=2,
            encoder_dim_feedforward=2048,
            max_seq_size: int = 1000):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_seq_size = max_seq_size
        self.pose_dims = pose_dims
        pose_dim = int(np.prod(pose_dims))

        # Embedding layers
        self.positional_embeddings = nn.Embedding(
            num_embeddings=max_seq_size, embedding_dim=hidden_dim
        )

        self.embedding = nn.Embedding(
            num_embeddings=len(tokenizer),
            embedding_dim=hidden_dim,
            padding_idx=tokenizer.pad_token_id,
        )
        self.pose_projection = nn.Linear(pose_dim, hidden_dim)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=encoder_heads,
                                                   dim_feedforward=encoder_dim_feedforward, batch_first=True)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=text_encoder_depth)
        self.pose_encoder = nn.TransformerEncoder(encoder_layer, num_layers=pose_encoder_depth)

        # Predict sequence length
        self.seq_length = nn.Linear(hidden_dim, 1)

        # Predict pose difference
        self.pose_diff_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pose_dim),
        )

    def encode_text(self, texts: List[str]):
        tokenized = self.tokenizer(texts, device=self.device)
        positional_embedding = self.positional_embeddings(tokenized["positions"])
        embedding = self.embedding(tokenized["tokens_ids"]) + positional_embedding
        encoded = self.text_encoder(embedding, src_key_padding_mask=tokenized["attention_mask"])
        bos = encoded[:, 0, :]
        seq_length = self.seq_length(bos)
        return {"data": encoded, "mask": tokenized["attention_mask"]}, seq_length

    def refine_pose_sequence(self, pose_sequence, text_encoding, positional_embedding):
        batch_size, seq_length, _, _ = pose_sequence["data"].shape
        flat_pose_data = pose_sequence["data"].reshape(batch_size, seq_length, -1)
        # Encode pose sequence
        pose_embedding = self.pose_projection(flat_pose_data) + positional_embedding
        pose_text_sequence = torch.cat([pose_embedding, text_encoding["data"]], dim=1)
        pose_text_mask = torch.cat(
            [pose_sequence["mask"], text_encoding["mask"]], dim=1
        )
        pose_encoding = self.pose_encoder(
            pose_text_sequence, src_key_padding_mask=pose_text_mask
        )[:, :seq_length, :]
        # Predict desired change
        flat_pose_projection = self.pose_diff_projection(pose_encoding)
        return flat_pose_projection.reshape(batch_size, seq_length, *self.pose_dims)

    def forward(self, text: str, first_pose: torch.Tensor, step_size: float = 0.5):
        text_encoding, sequence_length = self.encode_text([text])
        sequence_length = round(float(sequence_length))

        pose_sequence = {
            "data": first_pose.expand(1, sequence_length, *self.pose_dims),
            "mask": torch.zeros([1, sequence_length], dtype=torch.bool),
        }
        positions = torch.arange(0, min(sequence_length, self.max_seq_size), dtype=torch.int, device=self.device)
        positional_embedding = self.positional_embeddings(positions)
        while True:
            yield pose_sequence["data"][0]

            step = self.refine_pose_sequence(pose_sequence, text_encoding, positional_embedding)
            pose_sequence["data"] = pose_sequence["data"] + step_size * step

    def training_step(self, batch, *unused_args, steps=10):
        return self.step(batch, *unused_args, steps=steps, name="train")

    def validation_step(self, batch, *unused_args, steps=10):
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

        positions = torch.arange(0, pose_seq_length, dtype=torch.int, device=self.device)
        positional_embedding = self.positional_embeddings(positions)

        refinement_loss = 0
        for _ in range(steps):
            pose_sequence["data"] = pose_sequence["data"].detach()  # Detach from graph
            l1_gold = pose["data"] - pose_sequence["data"]
            l1_predicted = self.refine_pose_sequence(pose_sequence, text_encoding, positional_embedding)
            refinement_loss += masked_mse_loss(l1_gold, l1_predicted, confidence=pose["confidence"])

            step_shape = [batch_size, 1, 1, 1]
            step_size = 1 + torch.randn(step_shape, device=self.device) / 10
            l1_step = l1_gold if name == "validation" else l1_predicted
            pose_sequence["data"] = pose_sequence["data"] + step_size * l1_step

            if name == "train":  # add just a little noise while training
                pose_sequence["data"] = pose_sequence["data"] + torch.randn_like(pose_sequence["data"]) * 1e-4

        self.log(name + "_seq_length_loss", sequence_length_loss, batch_size=batch_size)
        self.log(name + "_refinement_loss", refinement_loss, batch_size=batch_size)
        train_loss = refinement_loss + sequence_length_loss
        self.log(name + "_loss", train_loss, batch_size=batch_size)
        return train_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
