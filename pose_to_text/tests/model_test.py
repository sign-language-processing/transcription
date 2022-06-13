import unittest
from unittest.mock import MagicMock

import torch

from ...shared.tokenizers.dummy_tokenizer import DummyTokenizer
from ..model import PoseToTextModel


class ModelTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_dim = (2, 2)
        self.seq_length = 5
        self.hidden_dim = 4

    def test_embed_text(self):
        model = PoseToTextModel(
            tokenizer=DummyTokenizer(),
            hidden_dim=self.hidden_dim,
            pose_dims=self.pose_dim
        )
        embedded_text = model.embed_text(["test"])
        self.assertTrue(torch.all(torch.isfinite(embedded_text["data"])))
        self.assertTrue(torch.all(torch.eq(torch.zeros_like(embedded_text["mask"]), embedded_text["mask"])))
        self.assertEqual(embedded_text["tokens_ids"].dtype, torch.int64)

    def model_setup(self):
        model = PoseToTextModel(
            tokenizer=DummyTokenizer(),
            hidden_dim=self.hidden_dim,
            pose_dims=self.pose_dim,
            max_seq_size=10,
            pose_encoder_depth=1,
            text_encoder_depth=1,
            encoder_heads=2,
            encoder_dim_feedforward=16,
        )
        model.embed_text = MagicMock(
            return_value=({
                "tokens_ids": torch.tensor([[2, 3]], dtype=torch.long),
                "data": torch.ones([1, 2, self.hidden_dim]),
                "mask": torch.zeros([1, 2], dtype=torch.bool),
            })
        )
        model.log = MagicMock(return_value=True)
        return model

    def get_batch(self):
        return {
            "text": ["text1"],
            "pose": {
                "length": torch.tensor([self.seq_length], dtype=torch.float),
                "data": torch.ones([1, self.seq_length, *self.pose_dim], dtype=torch.float),
                "inverse_mask": torch.ones([1, self.seq_length]),
            },
        }

    def test_training_step_expected_loss_finite(self):
        model = self.model_setup()
        batch = self.get_batch()

        loss = model.training_step(batch)
        self.assertNotEqual(float(loss), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_forward_returns_texts(self):
        model = self.model_setup()
        batch = self.get_batch()

        with torch.no_grad():
            model.eval()
            pred = model.forward(batch)

        self.assertEqual(len(pred), 1)
        self.assertTrue(isinstance(pred[0], str))



if __name__ == "__main__":
    unittest.main()
