"""Gauge-aware deterministic dense linear solve."""
import numpy as np

def constrained_solve(A,b,areas=None):
    """Least-squares solve, enforcing zero integrated source flux."""
    if areas is None: areas=np.ones(A.shape[1])
    augmented=np.block([[A, areas[:,None]],[areas[None,:],np.zeros((1,1))]])
    rhs=np.r_[b,0.]
    x=np.linalg.lstsq(augmented,rhs,rcond=None)[0][:-1]
    return x, float(np.linalg.norm(A@x-b)/max(np.linalg.norm(b),1.))
