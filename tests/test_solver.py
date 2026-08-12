import tempfile
import unittest
from pathlib import Path

import numpy as np

from wave_resistance import (
    LinearFreeSurfaceSolver,
    MeshSettings,
    SolverSettings,
    image_inspired_yacht,
)


class SolverTests(unittest.TestCase):
    def _solver(self):
        return LinearFreeSurfaceSolver(
            MeshSettings(
                hull_stations=13,
                hull_vertical_points=7,
                free_surface_x_points=17,
                free_surface_y_points=6,
                upstream_lengths=0.5,
                downstream_lengths=1.0,
                lateral_lengths=0.7,
            ),
            SolverSettings(
                wave_cut_points=64,
                kochin_integration_points=1501,
                kochin_t_max=25.0,
            ),
        )

    def test_curve_is_nonnegative_and_reports_residuals(self):
        result = self._solver().solve(image_inspired_yacht(), [0.25, 0.30])
        self.assertTrue(np.all(result.wave_resistance_coefficient >= 0.0))
        self.assertTrue(np.all(result.wave_resistance_N >= 0.0))
        self.assertTrue(all(item.relative_residual < 1.0e-8 for item in result.diagnostics))
        self.assertEqual(result.froude_number.shape, (2,))

    def test_scale_invariance_of_coefficient_and_force_scaling(self):
        solver = self._solver()
        small = solver.solve(image_inspired_yacht(length_ref_m=1.0), [0.30])
        large = solver.solve(image_inspired_yacht(length_ref_m=2.0), [0.30])
        np.testing.assert_allclose(
            small.wave_resistance_coefficient,
            large.wave_resistance_coefficient,
            rtol=2.0e-12,
        )
        self.assertAlmostEqual(large.wave_resistance_N[0] / small.wave_resistance_N[0], 8.0)

    def test_envelope_and_exports(self):
        solver = self._solver()
        with self.assertRaises(ValueError):
            solver.solve(image_inspired_yacht(), [0.10])
        result = solver.solve(image_inspired_yacht(), [0.30], retain_wave_fields=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result.to_csv(root / "curve.csv")
            result.diagnostics_to_json(root / "diagnostics.json")
            result.wave_fields_to_npz(root / "fields.npz")
            self.assertTrue((root / "curve.csv").is_file())
            self.assertTrue((root / "diagnostics.json").is_file())
            self.assertTrue((root / "fields.npz").is_file())


if __name__ == "__main__":
    unittest.main()
