"""Deterministic dense solves with explicit residual histories."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class LinearSolve:
    solution: np.ndarray
    residual_history: tuple
    rank: int
    condition_number: float
    converged: bool

def solve_dense(matrix, rhs, *, rtol=1e-9, allowed_nullity=0,
                condition_limit=float("inf")) -> LinearSolve:
    """Solve a dense system and apply explicit numerical acceptance tests.

    ``allowed_nullity`` is deliberately opt-in.  It is one only for the
    compatible closed-body Neumann problem; coupled free-surface systems must
    have full numerical rank.  A small residual alone is not evidence that a
    rank-deficient least-squares solution is unique.
    """
    a=np.asarray(matrix,float); b=np.asarray(rhs,float)
    if allowed_nullity < 0 or allowed_nullity >= min(a.shape):
        raise ValueError("allowed_nullity must be in [0, min(matrix.shape))")
    if condition_limit <= 0:
        raise ValueError("condition_limit must be positive")
    try:
        # Rank detection uses NumPy's dimension-scaled machine threshold.
        # ``rtol`` controls the equation residual only; using it as ``rcond``
        # would silently discard modes well below the separately configured
        # condition-number limit.
        x,residuals,rank,singular=np.linalg.lstsq(a,b,rcond=None)
    except np.linalg.LinAlgError:
        return LinearSolve(np.full(a.shape[1],np.nan),(float("inf"),),0,float("inf"),False)
    required_rank=min(a.shape)-int(allowed_nullity)
    residual=float(np.linalg.norm(a@x-b)/max(np.linalg.norm(b),1.0))
    # For an explicitly permitted null mode, condition the nonsingular
    # subspace rather than dividing by the discarded singular value.
    condition=(float(singular[0]/singular[required_rank-1])
               if len(singular)>=required_rank and singular[required_rank-1]>0
               else float("inf"))
    return LinearSolve(x,(residual,),int(rank),condition,
                       bool(np.isfinite(residual) and residual<=max(rtol,1e-12)*10
                            and rank>=required_rank
                            and condition<=condition_limit))
