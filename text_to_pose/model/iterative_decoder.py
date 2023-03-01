import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from _shared.models.pose_encoder import PoseEncoderModel

from .distribution import DistributionPredictionModel
from .image_encoder import ImageEncoderModel
from .masked_loss import masked_loss
from .schedule import cosine_beta_schedule, get_alphas
from .text_encoder import TextEncoderModel


class IterativeGuidedPoseGenerationModel(pl.LightningModule):

    def __init__(self,
                 pose_encoder: PoseEncoderModel,
                 text_encoder: TextEncoderModel = None,
                 image_encoder: ImageEncoderModel = None,
                 hidden_dim: int = 128,
                 max_seq_size: int = 1000,
                 learning_rate: float = 0.003,
                 num_steps: int = 10,
                 seq_len_loss_weight: float = 2e-5,
                 noise_epsilon: float = 1e-4,
                 loss_type='l2'):
        super().__init__()

        self.noise_epsilon = noise_epsilon
        self.max_seq_size = max_seq_size
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.seq_len_loss_weight = seq_len_loss_weight

        # Encoders
        self.pose_encoder = pose_encoder
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        # Embedding layers
        self.step_embedding = nn.Embedding(num_embeddings=num_steps, embedding_dim=hidden_dim)

        # Predict sequence length
        self.seq_length = DistributionPredictionModel(hidden_dim)

        # Predict pose difference
        self.pose_diff_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.pose_encoder.pose_dim),
        )

        # Diffusion parameters
        self.betas = cosine_beta_schedule(self.num_steps)  # Training time propotions
        self.alphas = get_alphas(self.betas)  # Inference time steps

    def refine_pose_sequence(self, pose_sequence, text_encoding, batch_step: torch.LongTensor):
        batch_size, seq_length, _, _ = pose_sequence["data"].shape

        step_embedding = self.step_embedding(batch_step).unsqueeze(1)
        step_mask = torch.zeros([step_embedding.shape[0], 1], dtype=torch.bool, device=self.device)

        additional_sequence = {
            "data": torch.cat([step_embedding, text_encoding["data"]], dim=1),
            "mask": torch.cat([step_mask, text_encoding["mask"]], dim=1)
        }

        pose_encoding = self.pose_encoder(pose=pose_sequence, additional_sequence=additional_sequence)
        pose_encoding = pose_encoding[:, :seq_length, :]

        # Predict desired change
        flat_pose_projection = self.pose_diff_projection(pose_encoding)
        return flat_pose_projection.reshape(batch_size, seq_length, *self.pose_encoder.pose_dims)

    def refinement_step(self, pose_sequence, text_encoding, batch_step: torch.LongTensor):
        change_pred = self.refine_pose_sequence(pose_sequence, text_encoding, batch_step)
        step_size = self.batch_step_size(batch_step)
        # Probably possible to perform multiplication without changing the shape
        while len(step_size.shape) < len(change_pred.shape):
            step_size = step_size.unsqueeze(-1)
        return change_pred, pose_sequence["data"] + torch.mul(step_size, change_pred)

    def get_step_proportion(self, step_num: int):
        # At the first step, n-1, we get 1 for noise and 0 for gold
        # At the last step, 0, we get a small number for noise, and a large one for gold
        return self.betas[step_num]

    def get_batch_step_proportion(self, batch_step: torch.LongTensor):
        steps = batch_step.tolist()
        sizes = [self.get_step_proportion(step) for step in steps]
        return torch.tensor(sizes, device=self.device, dtype=torch.float)

    def step_size(self, step_num: int):
        # Alphas in ascending order, but step size should be 1 when step_num=0
        return self.alphas[self.num_steps - step_num - 1]

    def batch_step_size(self, batch_step: torch.LongTensor):
        steps = batch_step.tolist()
        sizes = [self.step_size(step) for step in steps]
        return torch.tensor(sizes, device=self.device, dtype=torch.float)

    def forward(self, text: str, first_pose: torch.FloatTensor, force_sequence_length: int = None):
        text_encoding = self.text_encoder([text])
        sequence_length = self.seq_length(torch.mean(text_encoding["data"], dim=1))
        sequence_length = min(round(float(sequence_length)), self.max_seq_size)
        if force_sequence_length is not None:
            sequence_length = force_sequence_length

        pose_sequence = {
            "data": first_pose.expand(1, sequence_length, *self.pose_encoder.pose_dims),
            "mask": torch.zeros([1, sequence_length], dtype=torch.bool, device=self.device),
        }

        yield pose_sequence["data"][0]

        steps = torch.arange(self.num_steps, dtype=torch.long, device=self.device).unsqueeze(-1)
        steps_descending = (self.num_steps - 1) - steps

        for step_num in steps_descending:
            _, pred = self.refinement_step(pose_sequence, text_encoding, step_num)
            pose_sequence["data"] = pred
            yield pose_sequence["data"][0]

    def training_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, phase="train")

    def validation_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, phase="validation")

    def step(self, batch, *unused_args, phase: str):
        text_encoding = self.text_encoder(batch["text"])
        pose = batch["pose"]

        # Calculate sequence length loss
        sequence_length = self.seq_length(torch.mean(text_encoding["data"], dim=1))
        sequence_length_loss = F.mse_loss(sequence_length, pose["length"])

        # Repeat the first frame for initial prediction
        batch_size, pose_seq_length, _, _ = pose["data"].shape
        pose_sequence = {
            "data": torch.stack([pose["data"][:, 0]] * pose_seq_length, dim=1),
            "mask": torch.logical_not(pose["inverse_mask"])
        }

        # Similar to diffusion, we will choose a random step number for every sample from the batch
        batch_step = torch.randint(low=0, high=self.num_steps, size=[batch_size], dtype=torch.long, device=self.device)

        # Let's randomly add noise based on the step
        noise_proportion = self.get_batch_step_proportion(batch_step).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        noise_data = noise_proportion * pose_sequence["data"]
        gold_data = (1 - noise_proportion) * pose["data"]
        blend = noise_data + gold_data
        pose_sequence["data"] = blend

        if phase == "train":  # add just a little noise while training
            noise = torch.randn_like(pose_sequence["data"]) * self.noise_epsilon
            pose_sequence["data"] += noise

        # At every step, we apply loss to the predicted sequence to be exactly a full reproduction
        change_pred, _ = self.refinement_step(pose_sequence, text_encoding, batch_step)
        gold_difference = pose["data"] - pose_sequence["data"]
        refinement_loss = masked_loss(self.loss_type,
                                      gold_difference,
                                      change_pred,
                                      confidence=pose["confidence"],
                                      model_num_steps=self.num_steps)

        self.log(phase + "_seq_length_loss", sequence_length_loss, batch_size=batch_size)
        self.log(phase + "_refinement_loss", refinement_loss, batch_size=batch_size)
        loss = refinement_loss + self.seq_len_loss_weight * sequence_length_loss
        self.log(phase + "_loss", loss, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
