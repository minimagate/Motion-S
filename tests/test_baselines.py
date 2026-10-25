import unittest

from motions.baselines import constant_acceleration, constant_velocity, linear_kalman_filter


class BaselineTests(unittest.TestCase):
    def test_irregular_time_velocity_2d(self):
        self.assertEqual(
            constant_velocity([[0, 0], [2, 4]], [0, 2], [3, 4]), [[3.0, 6.0], [4.0, 8.0]]
        )

    def test_acceleration_and_short_history(self):
        result = constant_acceleration([[0], [1], [4]], [0, 1, 2], [3])
        self.assertAlmostEqual(result[0][0], 9.0)
        self.assertEqual(constant_acceleration([[1, 2]], [0], [1]), [[1.0, 2.0]])

    def test_3d_kalman_covariance(self):
        forecast = linear_kalman_filter([[0, 0, 0], [1, 2, 3]], [0, 1], [2, 3])
        self.assertEqual(len(forecast.means[0]), 3)
        self.assertEqual(len(forecast.covariances[0]), 3)
        self.assertTrue(
            all(forecast.covariances[1][i][i] >= forecast.covariances[0][i][i] for i in range(3))
        )

    def test_constant_velocity_one_point(self):
        self.assertEqual(constant_velocity([[5, 6, 7]], [4], [5]), [[5.0, 6.0, 7.0]])


if __name__ == "__main__":
    unittest.main()
