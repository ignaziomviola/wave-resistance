from dataclasses import dataclass

class UnsupportedPotentialFlow(ValueError): pass
class PotentialFlowFailure(RuntimeError): pass

@dataclass(frozen=True)
class ConvergenceStatus:
    algebraic_converged: bool=False
    free_surface_converged: bool=False
    force_balance_converged: bool=False
    mesh_converged: bool=False
    domain_converged: bool=False
    accepted: bool=False
