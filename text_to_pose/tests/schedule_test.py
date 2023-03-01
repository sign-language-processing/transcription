import unittest

import torch

from ..model.schedule import get_alphas


class ScheduleTestCase(unittest.TestCase):

    def test_alphas(self):
        betas = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        alphas = get_alphas(betas).tolist()

        self.assertAlmostEqual(1 / 10, alphas[0], delta=1e-6)
        self.assertAlmostEqual(1 / 9, alphas[1], delta=1e-6)
        self.assertAlmostEqual(1 / 8, alphas[2], delta=1e-6)
        self.assertAlmostEqual(1 / 7, alphas[3], delta=1e-6)
        self.assertAlmostEqual(1 / 6, alphas[4], delta=1e-6)
        self.assertAlmostEqual(1 / 5, alphas[5], delta=1e-6)
        self.assertAlmostEqual(1 / 4, alphas[6], delta=1e-6)
        self.assertAlmostEqual(1 / 3, alphas[7], delta=1e-6)
        self.assertAlmostEqual(1 / 2, alphas[8], delta=1e-6)
        self.assertAlmostEqual(1 / 1, alphas[9], delta=1e-6)


if __name__ == "__main__":
    unittest.main()
