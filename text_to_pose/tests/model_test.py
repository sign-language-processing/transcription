import unittest
from unittest.mock import MagicMock

import torch

from ...shared.tokenizers.dummy_tokenizer import DummyTokenizer
from ..model import IterativeTextGuidedPoseGenerationModel


class ModelTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_dim = (2, 2)
        self.seq_length = 5
        self.hidden_dim = 2

    def test_encode_text(self):
        model = IterativeTextGuidedPoseGenerationModel(
            tokenizer=DummyTokenizer(),
            hidden_dim=self.hidden_dim,
            pose_dims=self.pose_dim,
        )
        encoded_text, predicted_length = model.encode_text(["test"])
        self.assertTrue(torch.all(torch.isfinite(predicted_length)))
        self.assertTrue(torch.all(torch.isfinite(encoded_text["data"])))
        self.assertTrue(torch.all(torch.eq(torch.zeros_like(encoded_text["mask"]), encoded_text["mask"])))

    def model_setup(self):
        model = IterativeTextGuidedPoseGenerationModel(
            tokenizer=DummyTokenizer(),
            hidden_dim=self.hidden_dim,
            pose_dims=self.pose_dim,
        )
        model.encode_text = MagicMock(return_value=(
            {
                "data": torch.ones([1, 2, self.hidden_dim]),
                "mask": torch.zeros([1, 2], dtype=torch.bool),
            },
            torch.tensor([self.seq_length]),
        ))
        model.log = MagicMock(return_value=True)
        return model

    def model_forward(self):
        model = self.model_setup()
        model.eval()
        with torch.no_grad():
            first_pose = torch.full(self.pose_dim, fill_value=2, dtype=torch.float)
            return model.forward("", first_pose)

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
        batch = self.get_batch(confidence=0)

        loss = float(model.training_step(batch, None, steps=1))
        self.assertEqual(loss, 0)

    def test_training_step_expected_loss_finite(self):
        model = self.model_setup()
        batch = self.get_batch(confidence=1)

        loss = model.training_step(batch, None, steps=1)
        self.assertNotEqual(float(loss), 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
