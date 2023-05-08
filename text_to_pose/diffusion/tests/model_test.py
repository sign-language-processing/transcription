import unittest
from unittest.mock import MagicMock

import torch

from _shared.models import PoseEncoderModel
from text_to_pose.diffusion.src.model import TextEncoderModel

from _shared.tokenizers.dummy_tokenizer import DummyTokenizer
from text_to_pose.diffusion.src.model.iterative_decoder import IterativeGuidedPoseGenerationModel


class ModelTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_dim = (2, 2)
        self.seq_length = 5
        self.hidden_dim = 2

    def test_encode_text(self):
        text_encoder = TextEncoderModel(tokenizer=DummyTokenizer(), hidden_dim=self.hidden_dim)
        encoded_text = text_encoder(["test"])
        self.assertTrue(torch.all(torch.isfinite(encoded_text["data"])))
        self.assertTrue(torch.all(torch.eq(torch.zeros_like(encoded_text["mask"]), encoded_text["mask"])))

    def model_setup(self):
        pose_encoder = PoseEncoderModel(pose_dims=self.pose_dim,
                                        hidden_dim=self.hidden_dim,
                                        max_seq_size=self.seq_length)

        text_encoder = MagicMock(return_value={
            "data": torch.ones([1, 2, self.hidden_dim]),
            "mask": torch.zeros([1, 2], dtype=torch.bool),
        })
        model = IterativeGuidedPoseGenerationModel(text_encoder=text_encoder,
                                                   pose_encoder=pose_encoder,
                                                   hidden_dim=self.hidden_dim,
                                                   max_seq_size=self.seq_length)
        model.log = MagicMock(return_value=True)
        return model

    def model_forward(self):
        model = self.model_setup()
        model.eval()
        with torch.no_grad():
            first_pose = torch.full(self.pose_dim, fill_value=2, dtype=torch.float)
            return model.forward("", first_pose, force_sequence_length=self.seq_length)

    def test_forward_yields_initial_pose_sequence(self):
        model_forward = self.model_forward()

        pose_sequence = next(model_forward)
        self.assertEqual(pose_sequence.shape, (self.seq_length, *self.pose_dim))
        self.assertTrue(torch.all(pose_sequence == 2))

    def test_forward_yields_many_pose_sequences(self):
        model_forward = self.model_forward()

        next(model_forward)
        pose_sequence = next(model_forward)
        self.assertEqual(pose_sequence.shape, (self.seq_length, *self.pose_dim))
        self.assertTrue(torch.all(torch.isfinite(pose_sequence)))

    def get_batch(self, confidence=1):
        return {
            "text": ["text1"],
            "pose": {
                "length": torch.tensor([self.seq_length], dtype=torch.float),
                "data": torch.ones([1, self.seq_length, *self.pose_dim], dtype=torch.float),
                "confidence": torch.full([1, self.seq_length, self.pose_dim[0]], fill_value=confidence),
                "inverse_mask": torch.ones([1, self.seq_length]),
            },
        }

    def test_training_step_expected_loss_zero(self):
        model = self.model_setup()
        model.seq_len_loss_weight = 0
        batch = self.get_batch(confidence=0)

        loss = float(model.training_step(batch))
        self.assertEqual(0, loss)

    def test_training_step_expected_loss_finite(self):
        model = self.model_setup()
        batch = self.get_batch(confidence=1)

        loss = model.training_step(batch)
        self.assertNotEqual(0, float(loss))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
