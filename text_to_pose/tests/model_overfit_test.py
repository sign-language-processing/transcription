import random
import unittest

import torch

from _shared.models import PoseEncoderModel

from ..._shared.tokenizers.dummy_tokenizer import DummyTokenizer
from ..model.iterative_decoder import IterativeGuidedPoseGenerationModel
from ..model.text_encoder import TextEncoderModel


def get_batch(bsz=4):
    data_tensor = torch.tensor([[[1, 1]], [[2, 2]], [[3, 3]]], dtype=torch.float32)
    return {
        "text": ["text1"] * bsz,
        "pose": {
            "length": torch.tensor([3], dtype=torch.float32).expand(bsz, 1),
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

        hidden_dim = 10
        max_seq_size = 10

        text_encoder = TextEncoderModel(tokenizer=DummyTokenizer(), hidden_dim=hidden_dim, dim_feedforward=10)

        pose_encoder = PoseEncoderModel(pose_dims=(1, 2),
                                        encoder_dim_feedforward=10,
                                        hidden_dim=hidden_dim,
                                        max_seq_size=max_seq_size)

        model = IterativeGuidedPoseGenerationModel(
            text_encoder=text_encoder,
            pose_encoder=pose_encoder,
            hidden_dim=hidden_dim,
            max_seq_size=max_seq_size,
            num_steps=steps,
            seq_len_loss_weight=1  # Make sure sequence length is well predicted
        )

        optimizer = model.configure_optimizers()

        model.train()
        torch.set_grad_enabled(True)

        # Simple training loop
        losses = []
        for _ in range(50):
            loss = model.training_step(batch)
            loss_float = float(loss.detach())
            losses.append(loss_float)

            optimizer.zero_grad()  # clear gradients
            loss.backward()  # backward
            optimizer.step()  # update parameters

        print("losses", losses)

        model.eval()

        first_pose = batch["pose"]["data"][0, 0, :, :]
        with torch.no_grad():
            prediction = model.forward("text1", first_pose=first_pose)

        # Exhaust sequence
        final_seq = None
        for seq in prediction:
            final_seq = seq
            print("seq predicted", seq)

        self.assertEqual(final_seq.shape, (3, 1, 2))
        self.assertTrue(torch.all(torch.eq(torch.round(final_seq), batch["pose"]["data"][0])))

    def test_model_should_overfit_single_step(self):
        # Here, in training the model always sees the same sequence
        self.overfit_in_steps(steps=1)

    def test_model_should_overfit_multiple_steps(self):
        # Here, in training the model sees different sequences
        self.overfit_in_steps(steps=3)


if __name__ == '__main__':
    unittest.main()
