import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wave_resistance.models import SolverSettings, WaterProperties, WaveResistanceResult


class ModelTests(unittest.TestCase):
    def test_water_properties_reject_nonpositive_values(self):
        with self.assertRaises(ValueError):
            WaterProperties(rho_kg_m3=0.0)
        with self.assertRaises(ValueError):
            WaterProperties(g_m_s2=-1.0)

    def test_solver_settings_enforce_phase_resolution(self):
        with self.assertRaises(ValueError):
            SolverSettings(min_panels_per_cycle=7)

    def test_result_exports_csv(self):
        result = WaveResistanceResult(
            fn=np.array([0.2]),
            speed_m_s=np.array([1.0]),
            wave_resistance_N=np.array([2.0]),
            c_wave_resistance=np.array([0.003]),
            integral=np.array([1.0e-5]),
            quadrature_error=np.array([1.0e-10]),
            tail_fraction=np.array([1.0e-7]),
            cancellation_ratio=np.array([2.0]),
            converged=np.array([True]),
            validity_flags=((),),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curve.csv"
            result.to_csv(path)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["converged"], "True")


if __name__ == "__main__":
    unittest.main()
