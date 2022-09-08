import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class PoseTaggingModel(pl.LightningModule):

    def __init__(self, pose_dims: (int, int) = (137, 2), hidden_dim: int = 128, encoder_depth=2):
        super().__init__()

        self.pose_dims = pose_dims
        pose_dim = int(np.prod(pose_dims))

        self.pose_projection = nn.Linear(pose_dim, hidden_dim)

        assert hidden_dim / 2 == hidden_dim // 2, "Hidden dimensions must be even, not odd"

        # Encoder
        self.encoder = nn.LSTM(hidden_dim,
                               hidden_dim // 2,
                               num_layers=encoder_depth,
                               batch_first=True,
                               bidirectional=True)

        # tag sequence for sign bio / sentence bio
        self.sign_bio_head = nn.Linear(hidden_dim, 3)
        self.sentence_bio_head = nn.Linear(hidden_dim, 3)

        self.loss_function = nn.NLLLoss(reduction='none',
                                        weight=torch.tensor([1, 25, 1], dtype=torch.float))  # B is important

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

        sign_losses = self.loss_function(log_probs["sign"].reshape(-1, 3), batch["sign_bio"].reshape(-1))
        sign_loss = (sign_losses * loss_mask).mean()
        sentence_losses = self.loss_function(log_probs["sentence"].reshape(-1, 3), batch["sentence_bio"].reshape(-1))
        sentence_loss = (sentence_losses * loss_mask).mean()
        loss = sign_loss + sentence_loss

        self.log(name + "_sign_loss", sign_loss, batch_size=batch_size)
        self.log(name + "_sentence_loss", sentence_loss, batch_size=batch_size)
        self.log(name + "_loss", loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
