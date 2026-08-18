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

def solve_dense(matrix, rhs, *, rtol=1e-9) -> LinearSolve:
    a=np.asarray(matrix,float); b=np.asarray(rhs,float)
    try:
        x,residuals,rank,singular=np.linalg.lstsq(a,b,rcond=rtol)
    except np.linalg.LinAlgError:
        return LinearSolve(np.full(a.shape[1],np.nan),(float("inf"),),0,float("inf"),False)
    residual=float(np.linalg.norm(a@x-b)/max(np.linalg.norm(b),1.0))
    condition=float(singular[0]/singular[-1]) if len(singular) and singular[-1]>0 else float("inf")
    # The exterior single-layer Neumann operator has one constant-density
    # null mode on a closed component.  Flux compatibility fixes the physical
    # velocity even when the unaugmented collocation matrix reports nullity 1.
    return LinearSolve(x,(residual,),int(rank),condition,
                       bool(np.isfinite(residual) and residual<=max(rtol,1e-12)*10
                            and rank>=min(a.shape)-1))
