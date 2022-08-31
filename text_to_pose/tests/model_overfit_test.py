import random
import unittest

import torch

from ...shared.tokenizers.dummy_tokenizer import DummyTokenizer
from ..model import IterativeTextGuidedPoseGenerationModel


def get_batch(bsz=4):
    data_tensor = torch.tensor([[[1, 1]], [[2, 2]], [[3, 3]]], dtype=torch.float)
    return {
        "text": ["text1"] * bsz,
        "pose": {
            "length": torch.tensor([3], dtype=torch.float).expand(bsz, 1),
            "data": data_tensor.expand(bsz, *data_tensor.shape),
            "confidence": torch.ones([bsz, 3, 1]),
            "inverse_mask": torch.ones([bsz, 3]),
        },
    }


class ModelOverfitTestCase(unittest.TestCase):

    def overfit_in_steps(self, steps: int):
        torch.manual_seed(42)
        random.seed(42)

        batch = get_batch()

        model = IterativeTextGuidedPoseGenerationModel(
            tokenizer=DummyTokenizer(),
            hidden_dim=10,
            encoder_dim_feedforward=10,
            pose_dims=(1, 2),
        )
        optimizer = model.configure_optimizers()

        model.train()
        torch.set_grad_enabled(True)

        # Training loop
        losses = []
        for _ in range(500):
            loss = model.training_step(batch, steps=steps)
            loss_float = float(loss.detach())
            losses.append(loss_float)

            optimizer.zero_grad()  # clear gradients
            loss.backward()  # backward
            optimizer.step()  # update parameters

        print("losses", losses)

        model.eval()

        first_pose = batch["pose"]["data"][0, 0, :, :]
        with torch.no_grad():
            prediction = model.forward("text1", first_pose=first_pose, step_size=1)
        next(prediction)
        for _ in range(steps):
            seq = next(prediction)
            print("seq predicted", seq)

            self.assertEqual(seq.shape, (3, 1, 2))
            self.assertTrue(torch.all(torch.eq(torch.round(seq), batch["pose"]["data"][0])))

    def test_model_should_overfit_single_step(self):
        # Here, in training the model always sees the same sequence
        self.overfit_in_steps(steps=1)

    def test_model_should_overfit_multiple_steps(self):
        # Here, in training the model sees different sequences
        self.overfit_in_steps(steps=3)


if __name__ == '__main__':
    unittest.main()
