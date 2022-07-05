import random
import unittest

import torch

from ..model import PoseToTextModel
from ...shared.tokenizers.sign_language_tokenizer import SignLanguageTokenizer

# It is important t test overfitting on more than one string,
# to make sure the model doesn't always generate the same output.
TEXTS = ["M518x533S1870a489x515", '']  # 10 tokens, 17 tokens


def get_batch():
    bsz = len(TEXTS)
    return {
        "text": TEXTS,
        "pose": {
            "length": torch.tensor([3], dtype=torch.float).expand(bsz, 1),
            "data": torch.randn([bsz, 3, 10, 2], dtype=torch.float),
            "confidence": torch.ones([bsz, 3, 1]),
            "inverse_mask": torch.ones([bsz, 3], dtype=torch.int8),
        },
    }


class ModelOverfitTestCase(unittest.TestCase):
    def test_model_should_overfit(self):
        torch.manual_seed(42)
        random.seed(42)

        batch = get_batch()

        model = PoseToTextModel(
            tokenizer=SignLanguageTokenizer(),
            hidden_dim=10,
            encoder_dim_feedforward=10,
            max_seq_size=20,
            pose_dims=(10, 2),
        )
        optimizer = model.configure_optimizers(lr=1e-2)

        model.train()
        torch.set_grad_enabled(True)

        # Training loop
        losses = []
        for _ in range(120):
            loss = model.training_step(batch)
            loss_float = float(loss.detach())
            losses.append(loss_float)

            optimizer.zero_grad()  # clear gradients
            loss.backward()  # backward
            optimizer.step()  # update parameters

        print("losses", losses)

        with torch.no_grad():
            model.eval()
            pred = model.forward(batch)
        self.assertEqual(TEXTS, pred)


if __name__ == '__main__':
    unittest.main()
