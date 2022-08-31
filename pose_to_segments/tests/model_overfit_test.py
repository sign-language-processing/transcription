import random
import unittest

import torch

from ..model import PoseTaggingModel


def get_batch(bsz=4):
    data_tensor = torch.tensor([[[1, 1]], [[2, 2]], [[3, 3]]], dtype=torch.float)
    return {
        "pose": {
            "data": data_tensor.expand(bsz, *data_tensor.shape),
        },
        "mask": torch.ones([bsz, 3], dtype=torch.float),
        "sign_bio": torch.stack([torch.tensor([0, 2, 1], dtype=torch.long)] * bsz),
        "sentence_bio": torch.stack([torch.tensor([1, 0, 2], dtype=torch.long)] * bsz),
    }


class ModelOverfitTestCase(unittest.TestCase):

    def test_model_should_overfit(self):
        torch.manual_seed(42)
        random.seed(42)

        batch = get_batch(bsz=1)

        model = PoseTaggingModel(
            hidden_dim=10,
            pose_dims=(1, 2),
        )
        model.loss_function.weight = torch.ones(3, dtype=torch.float)  # Override the loss weight
        optimizer = model.configure_optimizers()

        model.train()
        torch.set_grad_enabled(True)

        # Training loop
        losses = []
        for _ in range(200):
            loss = model.training_step(batch)
            loss_float = float(loss.detach())
            losses.append(loss_float)

            optimizer.zero_grad()  # clear gradients
            loss.backward()  # backward
            optimizer.step()  # update parameters

        print("losses", losses)

        pose_data = batch["pose"]["data"][0].unsqueeze(0)
        prob = model(pose_data)

        sign_argmax = torch.argmax(prob["sign"], dim=-1)
        print("sign_argmax", sign_argmax)
        self.assertTrue(torch.all(torch.eq(sign_argmax, batch["sign_bio"][0])))

        sentence_argmax = torch.argmax(prob["sentence"], dim=-1)
        print("sentence_argmax", sentence_argmax)
        self.assertTrue(torch.all(torch.eq(sentence_argmax, batch["sentence_bio"][0])))


if __name__ == '__main__':
    unittest.main()
