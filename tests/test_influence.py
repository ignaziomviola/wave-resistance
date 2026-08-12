import unittest

import numpy as np

from wave_resistance.influence import (
    panel_normal_influence,
    rankine_potential,
    rankine_velocity,
)


class InfluenceTests(unittest.TestCase):
    def test_rankine_kernel_and_gradient(self):
        field = np.array([[1.0, 0.0, 0.0]])
        source = np.array([[0.0, 0.0, 0.0]])
        self.assertAlmostEqual(rankine_potential(field, source)[0, 0], 1.0 / (4.0 * np.pi))
        expected = np.array([-1.0 / (4.0 * np.pi), 0.0, 0.0])
        np.testing.assert_allclose(rankine_velocity(field, source)[0, 0], expected)

    def test_panel_operator_has_analytic_jump(self):
        points = np.array([[0.0, 1.0, -0.2], [1.0, 1.0, -0.2]])
        normals = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        matrix = panel_normal_influence(
            points, normals, points, np.array([0.1, 0.1]), mirror_y=False
        )
        np.testing.assert_allclose(np.diag(matrix), -0.5)
        self.assertTrue(np.all(np.isfinite(matrix)))

    def test_singular_point_sources_are_rejected(self):
        point = np.zeros((1, 3))
        with self.assertRaises(ValueError):
            rankine_potential(point, point)


if __name__ == "__main__":
    unittest.main()
