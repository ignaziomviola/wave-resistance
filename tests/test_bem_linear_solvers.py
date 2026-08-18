import unittest

import numpy as np

from wave_resistance.bem.linear_solvers import solve_dense


class DenseLinearSolverTests(unittest.TestCase):
    def test_rank_deficiency_is_rejected_unless_explicitly_allowed(self):
        matrix=np.array([[1.,1.],[2.,2.]])
        rhs=np.array([1.,2.])
        self.assertFalse(solve_dense(matrix,rhs).converged)
        self.assertTrue(solve_dense(matrix,rhs,allowed_nullity=1).converged)

    def test_condition_limit_is_an_acceptance_gate(self):
        matrix=np.diag([1.,1e-10])
        rhs=np.array([1.,1e-10])
        self.assertFalse(solve_dense(matrix,rhs,rtol=1e-12,
                                     condition_limit=1e8).converged)


if __name__=="__main__": unittest.main()
