import math
import unittest

import numpy as np

from wave_resistance import (
    MichellOperator,
    MichellSolver,
    SolverSettings,
    wigley_hull,
)
from wave_resistance.quadrature import PhaseAwareMichellIntegrator
from wave_resistance.reference import wigley_amplitude


class MichellSolverTests(unittest.TestCase):
    def test_default_wigley_matches_closed_form_amplitude_curve(self):
        hull = wigley_hull()
        solver = MichellSolver()
        result = solver.solve(hull, [0.30])

        def analytic(lambda_):
            return wigley_amplitude(lambda_, 0.30), np.ones_like(lambda_), False

        reference = PhaseAwareMichellIntegrator(
            SolverSettings(amplitude_crosscheck=False)
        ).integrate(0.30, 1.0, analytic)
        relative_error = abs(result.integral[0] - reference.integral) / reference.integral
        self.assertLess(relative_error, 0.005)
        self.assertTrue(result.converged[0])
        self.assertGreaterEqual(result.wave_resistance_N[0], 0.0)

    def test_geometric_scaling_preserves_coefficient_and_cubes_force(self):
        settings = SolverSettings(
            relative_tolerance=1e-4,
            amplitude_crosscheck=False,
            tail_consecutive_cycles=3,
        )
        solver = MichellSolver(settings)
        small = solver.solve(wigley_hull(length_m=1.0, nx=11, nz=5), [0.30])
        large = solver.solve(wigley_hull(length_m=2.0, nx=11, nz=5), [0.30])
        np.testing.assert_allclose(
            small.c_wave_resistance, large.c_wave_resistance, rtol=2e-12, atol=0.0
        )
        self.assertAlmostEqual(
            large.wave_resistance_N[0] / small.wave_resistance_N[0], 8.0, places=11
        )

    def test_offset_grid_refinement_changes_curve_by_less_than_one_percent(self):
        settings = SolverSettings(
            relative_tolerance=1e-4,
            amplitude_crosscheck=False,
            tail_consecutive_cycles=3,
        )
        solver = MichellSolver(settings)
        coarse = solver.solve(wigley_hull(nx=21, nz=9), [0.30])
        refined = solver.solve(wigley_hull(nx=41, nz=17), [0.30])
        relative_change = abs(
            refined.c_wave_resistance[0] - coarse.c_wave_resistance[0]
        ) / refined.c_wave_resistance[0]
        self.assertLess(relative_change, 0.01)

    def test_froude_envelope_requires_explicit_override(self):
        hull = wigley_hull(nx=11, nz=5)
        with self.assertRaises(ValueError):
            MichellSolver().solve(hull, [0.08])
        settings = SolverSettings(
            allow_outside_envelope=True,
            relative_tolerance=1e-4,
            amplitude_crosscheck=False,
            tail_consecutive_cycles=3,
        )
        result = MichellSolver(settings).solve(hull, [0.08])
        self.assertTrue(any("outside declared" in flag for flag in result.validity_flags[0]))

    def test_spectrum_and_batch_operator(self):
        settings = SolverSettings(
            relative_tolerance=1e-4,
            amplitude_crosscheck=False,
            tail_consecutive_cycles=3,
            spectrum_points=32,
        )
        solver = MichellSolver(settings)
        hull = wigley_hull(nx=11, nz=5)
        result = solver.solve(hull, [0.30], include_spectrum=True)
        self.assertIn(0.30, result.spectral_density)
        spectrum = result.spectral_density[0.30]
        self.assertEqual(spectrum.lambda_values.shape, (32,))
        self.assertTrue(np.all(spectrum.d_integral_d_lambda >= 0.0))

        operator = MichellOperator.from_hull(hull, [0.30], solver)
        batch = np.stack((hull.half_breadths, 1.02 * hull.half_breadths))
        batch_result = operator.solve_batch(batch)
        self.assertEqual(len(batch_result.results), 2)
        self.assertTrue(all(item.converged[0] for item in batch_result.results))
        self.assertGreater(batch_result.results[1].wave_resistance_N[0], 0.0)


if __name__ == "__main__":
    unittest.main()
