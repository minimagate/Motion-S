import unittest

from motions.metrics import (
    ade,
    fde,
    gaussian_mixture_calibration,
    gaussian_mixture_nll,
    inference_latency,
    parameter_count,
)


class MetricsTests(unittest.TestCase):
    def test_ade_fde(self):
        pred = [[0, 0], [2, 0]]
        target = [[0, 1], [0, 0]]
        self.assertEqual(ade(pred, target), 1.5)
        self.assertEqual(fde(pred, target), 2.0)

    def test_mixture_nll_and_calibration(self):
        means = [[0.0]]
        covs = [[[1.0]]]
        weights = [1.0]
        self.assertAlmostEqual(
            gaussian_mixture_nll([0.0], means, covs, weights),
            0.5 * __import__("math").log(2 * __import__("math").pi),
        )
        self.assertTrue(gaussian_mixture_calibration([0.0], means, covs, weights)[0.5])

    def test_latency_and_parameter_count(self):
        self.assertGreater(inference_latency(lambda: None, repeats=2)["calls_per_second"], 0)
        self.assertEqual(parameter_count([[1, 2], [3]]), 3)


if __name__ == "__main__":
    unittest.main()
