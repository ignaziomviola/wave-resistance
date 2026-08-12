import tempfile
import unittest
from pathlib import Path

from wave_resistance import MeshSettings, SolverSettings, WaterProperties


class ModelTests(unittest.TestCase):
    def test_invalid_physical_and_numerical_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            WaterProperties(density_kg_m3=0.0)
        with self.assertRaises(ValueError):
            MeshSettings(hull_stations=3)
        with self.assertRaises(ValueError):
            SolverSettings(fn_min=0.5, fn_max=0.4)


if __name__ == "__main__":
    unittest.main()
