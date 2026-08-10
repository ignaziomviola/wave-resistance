import unittest

import numpy as np

from wave_resistance.models import SolverSettings
from wave_resistance.quadrature import PhaseAwareMichellIntegrator


def evaluator(amplitude_function):
    def evaluate(lambda_):
        amplitude = amplitude_function(lambda_)
        return amplitude, np.ones_like(lambda_), False

    return evaluate


class QuadratureTests(unittest.TestCase):
    def test_zero_amplitude_is_nonnegative_and_converges(self):
        integrator = PhaseAwareMichellIntegrator(SolverSettings())
        result = integrator.integrate(0.3, 1.0, evaluator(lambda x: np.zeros_like(x)))
        self.assertTrue(result.converged)
        self.assertEqual(result.integral, 0.0)

    def test_phase_aware_integral_agrees_under_refinement(self):
        def oscillatory(lambda_):
            return (1.0 + np.exp(90.0j * lambda_)) * np.exp(-lambda_ / 3.0) / lambda_**2

        base = SolverSettings(
            relative_tolerance=2e-6,
            min_panels_per_cycle=8,
            lambda_cap=300.0,
        )
        refined = SolverSettings(
            relative_tolerance=2e-7,
            min_panels_per_cycle=16,
            lambda_cap=300.0,
        )
        first = PhaseAwareMichellIntegrator(base).integrate(
            0.25, 1.0, evaluator(oscillatory)
        )
        second = PhaseAwareMichellIntegrator(refined).integrate(
            0.25, 1.0, evaluator(oscillatory)
        )
        self.assertTrue(first.converged)
        self.assertTrue(second.converged)
        self.assertLess(abs(first.integral - second.integral), 2e-5 * second.integral)


if __name__ == "__main__":
    unittest.main()
