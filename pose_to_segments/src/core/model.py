from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class PoseTaggingModel(pl.LightningModule):

    def __init__(self,
                 sign_class_weights: List[float],
                 sentence_class_weights: List[float],
                 pose_dims: (int, int) = (137, 2),
                 hidden_dim: int = 128,
                 encoder_depth=2,
                 encoder_bidirectional=True,
                 learning_rate=1e-3):
        super().__init__()

        self.learning_rate = learning_rate

        self.pose_dims = pose_dims
        pose_dim = int(np.prod(pose_dims))

        self.pose_projection = nn.Linear(pose_dim, hidden_dim)

        if encoder_bidirectional:
            assert hidden_dim / 2 == hidden_dim // 2, "Hidden dimensions must be even, not odd"
            lstm_hidden_dim = hidden_dim // 2
        else:
            lstm_hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.LSTM(hidden_dim,
                               lstm_hidden_dim,
                               num_layers=encoder_depth,
                               batch_first=True,
                               bidirectional=encoder_bidirectional)

        # tag sequence for sign bio
        self.sign_bio_head = nn.Linear(hidden_dim, 3)
        sign_loss_weight = torch.tensor(sign_class_weights, dtype=torch.float)
        self.sign_loss_function = nn.NLLLoss(reduction='none', weight=sign_loss_weight)

        # tag sequence for sentence bio
        self.sentence_bio_head = nn.Linear(hidden_dim, 3)
        sentence_loss_weight = torch.tensor(sentence_class_weights, dtype=torch.float)
        self.sentence_loss_function = nn.NLLLoss(reduction='none', weight=sentence_loss_weight)

    def forward(self, pose_data: torch.Tensor):
        batch_size, seq_length, _, _ = pose_data.shape
        flat_pose_data = pose_data.reshape(batch_size, seq_length, -1)

        pose_projection = self.pose_projection(flat_pose_data)
        pose_encoding, _ = self.encoder(pose_projection)

        sign_bio_logits = self.sign_bio_head(pose_encoding)
        sentence_bio_logits = self.sentence_bio_head(pose_encoding)

        return {"sign": F.log_softmax(sign_bio_logits, dim=-1), "sentence": F.log_softmax(sentence_bio_logits, dim=-1)}

    def training_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, name="train")

    def validation_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, name="validation")

    def step(self, batch, *unused_args, name: str):
        pose_data = batch["pose"]["data"]
        batch_size = len(pose_data)

        log_probs = self.forward(pose_data)

        loss_mask = batch["mask"].reshape(-1)

        sign_losses = self.sign_loss_function(log_probs["sign"].reshape(-1, 3), batch["bio"]["sign"].reshape(-1))
        sign_loss = (sign_losses * loss_mask).mean()

        sentence_losses = self.sentence_loss_function(log_probs["sentence"].reshape(-1, 3),
                                                      batch["bio"]["sentence"].reshape(-1))
        sentence_loss = (sentence_losses * loss_mask).mean()

        loss = sign_loss + sentence_loss

        self.log(f"{name}_sign_loss", sign_loss, batch_size=batch_size)
        self.log(f"{name}_sentence_loss", sentence_loss, batch_size=batch_size)
        self.log(f"{name}_loss", loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
