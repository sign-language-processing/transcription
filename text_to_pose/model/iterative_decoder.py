import random
from typing import List

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
                 smoothness_loss_weight: float = 1e-2,
                 noise_epsilon: float = 1e-4,
                 loss_type='l2'):
        super().__init__()

        self.noise_epsilon = noise_epsilon
        self.max_seq_size = max_seq_size
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.seq_len_loss_weight = seq_len_loss_weight
        self.smoothness_loss_weight = smoothness_loss_weight

        # Encoders
        self.pose_encoder = pose_encoder
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        # Pose correction layer to fix missing joints
        self.pose_correction = nn.Sequential(
            nn.Linear(self.pose_encoder.pose_dim, self.pose_encoder.pose_dim),
            nn.SiLU(),
            nn.Linear(self.pose_encoder.pose_dim, self.pose_encoder.pose_dim),
        )

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

    def correct_pose(self, data: torch.FloatTensor):
        if self.training:
            _, keypoints, _ = data.shape
            # mask a block of keypoints
            keypoint_start = random.randint(0, keypoints - 1)
            keypoint_end = random.randint(keypoint_start, min(keypoints, keypoint_start + 21))
            data[:, keypoint_start:keypoint_end, :] = 0

        flat_pose = data.reshape(-1, self.pose_encoder.pose_dim)
        corrected_pose = self.pose_correction(flat_pose)
        if not self.training:
            flat_conf = (flat_pose != 0).float()
            corrected_pose = (1 - flat_conf) * corrected_pose + flat_conf * flat_pose

        return corrected_pose.reshape(data.shape)

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
        predicted_diff = flat_pose_projection.reshape(batch_size, seq_length, *self.pose_encoder.pose_dims)
        return predicted_diff

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

    def forward(self,
                text: str,
                first_pose: torch.FloatTensor,
                force_sequence_length: int = None,
                classifier_free_guidance=None):
        empty_text_encoding = self.text_encoder([""]) if classifier_free_guidance is not None else None
        text_encoding = self.text_encoder([text])
        sequence_length = self.seq_length(torch.mean(text_encoding["data"], dim=1))
        sequence_length = min(round(float(sequence_length)), self.max_seq_size)
        if force_sequence_length is not None:
            sequence_length = force_sequence_length

        # Add missing keypoints
        first_pose = self.correct_pose(first_pose)

        x_T = first_pose.expand(1, sequence_length, *self.pose_encoder.pose_dims)
        mask = torch.zeros([1, sequence_length], dtype=torch.bool, device=self.device)

        yield x_T[0]

        steps = torch.arange(self.num_steps, dtype=torch.long, device=self.device).unsqueeze(-1)
        steps_descending = (self.num_steps - 1) - steps

        x_t = x_T
        for step_num in steps_descending:
            pose_t = {"data": x_t, "mask": mask}
            conditional = self.refine_pose_sequence(pose_t, text_encoding, step_num)

            if classifier_free_guidance is not None:
                unconditional = self.refine_pose_sequence(pose_t, empty_text_encoding, step_num)
                x_0 = unconditional + classifier_free_guidance * (conditional - unconditional)
                print('cfg', (conditional - unconditional).abs().sum())
            else:
                print('conditional')
                x_0 = conditional

            yield x_0[0]

            if step_num > 0:
                # Now we need to noise the predicted sequence "back" to time t
                x_t = self.noise_pose_sequence(x_0, x_T, step_num - 1)

    def noise_pose_sequence(self,
                            x_0: torch.FloatTensor,
                            x_T: torch.FloatTensor,
                            batch_step: torch.LongTensor,
                            deviation=0):
        noise_proportion = self.get_batch_step_proportion(batch_step).view(-1, 1, 1, 1)
        if deviation > 0:
            noise_proportion *= 1 + deviation * torch.randn_like(noise_proportion)
        noise_data = noise_proportion * x_T
        gold_data = (1 - noise_proportion) * x_0
        blend = noise_data + gold_data
        return blend

    def training_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, steps=[-1])

    def validation_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, steps=list(range(self.num_steps)))

    def smoothness_loss(self, pose_sequence: torch.Tensor, confidence: torch.Tensor):
        shifted_pose = torch.roll(pose_sequence, 1, dims=1)
        shifted_confidence = torch.roll(confidence, 1, dims=1)
        confidence = confidence * shifted_confidence
        return masked_loss('l1', pose_sequence, shifted_pose, confidence=confidence, model_num_steps=self.num_steps)

    def step(self, batch, *unused_args, steps: List[int]):
        if self.training:
            # Randomly remove some text during training
            for i, text in enumerate(batch["text"]):
                if random.random() < 0.1:
                    batch["text"][i] = ""

        text_encoding = self.text_encoder(batch["text"])
        pose = batch["pose"]

        # Calculate sequence length loss
        sequence_length = self.seq_length(torch.mean(text_encoding["data"], dim=1))
        sequence_length_loss = F.mse_loss(sequence_length, pose["length"])

        # # Reconstruct missing keypoints from the first pose
        first_pose = pose["data"][:, 0]
        first_conf = pose["confidence"][:, 0]
        fixed_pose = self.correct_pose(first_pose)
        pose_reconstruction_loss = masked_loss(self.loss_type,
                                               first_pose,
                                               fixed_pose,
                                               confidence=first_conf,
                                               model_num_steps=1)

        # Repeat the first frame for initial prediction
        batch_size, pose_seq_length, _, _ = pose["data"].shape
        pose_sequence = {
            "data": torch.stack([first_pose] * pose_seq_length, dim=1),
            "mask": torch.logical_not(pose["inverse_mask"])
        }

        # In training, only one step is used. For validation, we use all steps
        refinement_loss = 0
        smoothness_loss = 0
        for step in steps:
            # Similar to diffusion, we will choose a random step number for every sample from the batch
            if step == -1:
                batch_step = torch.randint(low=0,
                                           high=self.num_steps,
                                           size=[batch_size],
                                           dtype=torch.long,
                                           device=self.device)
            else:
                # We want to make sure that we always use the same step number for validation loss calculation
                batch_step = torch.full([batch_size], fill_value=step, dtype=torch.long, device=self.device)

            # Let's randomly add noise based on the step
            deviation = self.noise_epsilon if self.training else 0
            pose_sequence["data"] = self.noise_pose_sequence(pose["data"],
                                                             pose_sequence["data"],
                                                             batch_step,
                                                             deviation=deviation)

            if self.training:  # multiply by just a little noise while training
                noise = 1 + torch.randn_like(pose_sequence["data"]) * self.noise_epsilon
                first_frame = pose_sequence["data"][:, 0]
                pose_sequence["data"] *= noise
                pose_sequence["data"][:, 0] = first_frame  # First frame should never change

            # At every step, we apply loss to the predicted sequence to be exactly a full reproduction
            predicted_sequence = self.refine_pose_sequence(pose_sequence, text_encoding, batch_step)

            refinement_loss += masked_loss(self.loss_type,
                                           pose["data"],
                                           predicted_sequence,
                                           confidence=pose["confidence"],
                                           model_num_steps=self.num_steps)

            smoothness_loss += self.smoothness_loss(predicted_sequence, confidence=pose["confidence"])

        phase = "train" if self.training else "validation"
        self.log(phase + "_seq_length_loss", sequence_length_loss, batch_size=batch_size)
        self.log(phase + "_refinement_loss", refinement_loss, batch_size=batch_size)
        self.log(phase + "_smoothness_loss", smoothness_loss, batch_size=batch_size)
        self.log(phase + "_reconstruction_loss", pose_reconstruction_loss, batch_size=batch_size)

        loss = pose_reconstruction_loss + refinement_loss + \
               self.seq_len_loss_weight * sequence_length_loss + \
               self.smoothness_loss_weight * smoothness_loss

        # loss = refinement_loss + self.seq_len_loss_weight * sequence_length_loss
        self.log(phase + "_loss", loss, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
