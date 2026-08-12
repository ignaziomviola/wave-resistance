import unittest

import numpy as np

from wave_resistance import (
    LinearFreeSurfaceSolver,
    MeshSettings,
    SolverSettings,
    wigley_hull,
    wigley_michell_coefficient,
)
from wave_resistance.kochin import (
    kochin_amplitude,
    kochin_wave_coefficient,
    kochin_wave_pattern,
)


class KochinTests(unittest.TestCase):
    def test_linear_source_wigley_matches_michell_over_declared_benchmark(self):
        hull = wigley_hull(beam_to_length=0.05)
        solver = LinearFreeSurfaceSolver(
            MeshSettings(
                hull_stations=41,
                hull_vertical_points=15,
                free_surface_x_points=13,
                free_surface_y_points=5,
            ),
            SolverSettings(wave_cut_points=64, kochin_integration_points=3001),
        )
        geometry = solver._prepare(hull)
        strengths = -geometry.normals[:, 0]
        for froude in (0.20, 0.25, 0.30, 0.35, 0.40):
            calculated, tail = kochin_wave_coefficient(
                geometry.collocation,
                geometry.areas,
                strengths,
                froude,
                geometry.wetted_area_over_l2,
                integration_points=3001,
            )
            reference = wigley_michell_coefficient(
                froude,
                hull.hydrostatics.wetted_area_m2,
                beam_over_length=0.05,
                integration_points=10001,
            )
            self.assertLess(abs(calculated - reference) / reference, 0.06)
            self.assertLess(tail, 1.0e-3)

    def test_zero_sources_have_zero_resistance(self):
        coefficient, tail = kochin_wave_coefficient(
            np.array([[0.0, 0.1, -0.1]]),
            np.array([0.01]),
            np.array([0.0]),
            0.3,
            0.2,
            integration_points=1001,
            t_max=20.0,
        )
        self.assertEqual(coefficient, 0.0)
        self.assertEqual(tail, 0.0)

    def test_amplitude_and_wave_pattern_are_symmetric(self):
        points = np.array([[0.1, 0.08, -0.06], [-0.1, 0.04, -0.03]])
        areas = np.array([0.02, 0.015])
        strengths = np.array([0.4, -0.2])
        t, lam, amplitude = kochin_amplitude(
            points,
            areas,
            strengths,
            0.3,
            integration_points=1001,
            t_max=8.0,
        )
        self.assertEqual(t.shape, lam.shape)
        self.assertEqual(t.shape, amplitude.shape)
        self.assertTrue(np.all(np.isfinite(amplitude)))

        x = np.linspace(0.6, 1.6, 24)
        y = np.linspace(-0.8, 0.8, 31)
        pattern = kochin_wave_pattern(
            points,
            areas,
            strengths,
            0.3,
            x,
            y,
            integration_points=1001,
            t_max=8.0,
        )
        self.assertEqual(pattern.shape, (y.size, x.size))
        self.assertTrue(np.allclose(pattern, pattern[::-1], atol=1.0e-12))
        self.assertAlmostEqual(float(np.max(np.abs(pattern))), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
