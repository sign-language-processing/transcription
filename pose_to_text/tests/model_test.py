import unittest

import torch
from joeynmt.vocabulary import Vocabulary

from pose_to_text.batch import SignBatch
from shared.collator.collator import collate_tensors

from ..model import build_model


class ModelTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_dim = (2, 2)
        self.seq_length = 5

    def model_setup(self):
        transformer_cfg = {
            "num_layers": 2,
            "num_heads": 2,
            "hidden_size": 10,
            "ff_size": 20,
            "dropout": 0.1,
            "emb_dropout": 0.1
        }
        cfg = {
            "decoder": {
                **transformer_cfg, "embeddings": {
                    "embedding_dim": 10
                }
            },
            "encoder": transformer_cfg,
            "pose_encoder": transformer_cfg,
        }
        model = build_model(pose_dims=self.pose_dim, cfg=cfg, trg_vocab=Vocabulary([]))
        model.log_parameters_list()
        model.loss_function = ("crossentropy", 0.0)

        return model

    def get_batch(self):
        return SignBatch(src=torch.rand(1, self.seq_length, *self.pose_dim),
                         src_length=collate_tensors([self.seq_length]),
                         trg=torch.zeros(1, self.seq_length, dtype=torch.long),
                         trg_length=collate_tensors([self.seq_length]),
                         device=torch.device("cpu"))

    def test_forward_expected_loss_finite(self):
        model = self.model_setup()
        batch = self.get_batch()

        loss = model(return_type="loss", **batch.__dict__)[0]
        self.assertNotEqual(float(loss), 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
