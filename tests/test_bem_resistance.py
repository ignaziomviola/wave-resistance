import unittest

import numpy as np

from wave_resistance.bem.resistance import force_balance


class ResistanceDiagnosticsTests(unittest.TestCase):
    def test_missing_independent_force_cannot_pass_balance(self):
        discrepancy,accepted=force_balance(1.,np.nan,.02,1e-5)
        self.assertTrue(np.isnan(discrepancy))
        self.assertFalse(accepted)


if __name__=="__main__": unittest.main()
