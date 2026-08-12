import unittest

import numpy as np

from wave_resistance import wigley_michell_amplitude, wigley_michell_coefficient


class ReferenceTests(unittest.TestCase):
    def test_amplitude_and_coefficient_are_finite_positive(self):
        amplitude = wigley_michell_amplitude(np.array([1.0, 10.0, 100.0]), 0.3)
        self.assertTrue(np.all(np.isfinite(amplitude)))
        self.assertGreater(wigley_michell_coefficient(0.3, 0.15), 0.0)

    def test_invalid_froude_is_rejected(self):
        with self.assertRaises(ValueError):
            wigley_michell_amplitude(np.array([1.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
