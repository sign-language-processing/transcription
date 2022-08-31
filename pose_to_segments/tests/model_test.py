import unittest
from unittest.mock import MagicMock

import torch

from ..model import PoseTaggingModel


class ModelTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_dim = (2, 2)
        self.seq_length = 5
        self.hidden_dim = 4

    def model_setup(self):
        model = PoseTaggingModel(hidden_dim=self.hidden_dim, pose_dims=self.pose_dim, encoder_depth=2)
        model.log = MagicMock(return_value=True)
        return model

    def get_batch(self):
        return {
            "pose": {
                "data": torch.ones([2, self.seq_length, *self.pose_dim], dtype=torch.float),
            },
            "mask": torch.ones([2, self.seq_length], dtype=torch.float),
            "sign_bio": torch.zeros((2, self.seq_length), dtype=torch.long),
            "sentence_bio": torch.zeros((2, self.seq_length), dtype=torch.long)
        }

    def test_forward_yields_bio_probs(self):
        model = self.model_setup()
        batch = self.get_batch()
        log_probs = model.forward(batch["pose"]["data"])

        # shape check
        self.assertEqual(log_probs["sign"].shape, (len(batch["sign_bio"]), self.seq_length, 3))
        self.assertEqual(log_probs["sentence"].shape, (len(batch["sign_bio"]), self.seq_length, 3))

        # nan / inf check
        self.assertTrue(torch.all(torch.isfinite(log_probs["sign"])))
        self.assertTrue(torch.all(torch.isfinite(log_probs["sentence"])))

        # softmax probs check
        sum_sign = torch.exp(log_probs["sign"]).sum(-1)
        self.assertTrue(torch.allclose(sum_sign, torch.ones_like(sum_sign)))
        sum_sentence = torch.exp(log_probs["sentence"]).sum(-1)
        self.assertTrue(torch.allclose(sum_sentence, torch.ones_like(sum_sentence)))

    def test_training_step_expected_loss_finite(self):
        model = self.model_setup()
        batch = self.get_batch()

        loss = model.training_step(batch)
        self.assertNotEqual(float(loss), 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
