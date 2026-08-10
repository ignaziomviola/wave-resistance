import unittest

import numpy as np

from wave_resistance.reference import wigley_amplitude


class WigleyReferenceTests(unittest.TestCase):
    def test_translation_phase_does_not_change_magnitude(self):
        lam = np.linspace(1.0, 8.0, 20)
        amplitude = wigley_amplitude(lam, 0.3)
        shifted = amplitude * np.exp(-1.7j * lam / 0.3**2)
        np.testing.assert_allclose(np.abs(amplitude), np.abs(shifted), rtol=1e-14)

    def test_amplitude_is_finite(self):
        amplitude = wigley_amplitude(np.array([1.0, 10.0, 100.0]), 0.2)
        self.assertTrue(np.all(np.isfinite(amplitude)))

    def test_invalid_froude_rejected(self):
        with self.assertRaises(ValueError):
            wigley_amplitude(1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
