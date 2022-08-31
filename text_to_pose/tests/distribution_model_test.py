import unittest

import torch

from ..model import DistributionPredictionModel


class DistributionModelTestCase(unittest.TestCase):

    def test_prediction_in_eval_should_be_consistent(self):
        model = DistributionPredictionModel(input_size=10)
        model.eval()
        tensor = torch.randn(size=[10])
        pred_1 = float(model(tensor))
        pred_2 = float(model(tensor))

        self.assertEqual(pred_1, pred_2)

    def test_prediction_in_eval_should_be_inconsistent(self):
        model = DistributionPredictionModel(input_size=10)
        model.train()
        tensor = torch.randn(size=[10])
        pred_1 = float(model(tensor))
        pred_2 = float(model(tensor))

        self.assertNotEqual(pred_1, pred_2)


if __name__ == "__main__":
    unittest.main()
