"""Neumann-Kelvin wave resistance of a bare sailing-yacht canoe body.

Scope: steady, deep-water, inviscid, irrotational flow with a linearised free-surface
condition and the exact body condition on the actual wetted hull at a prescribed
attitude.  Viscous and form resistance, circulation, lift, leeway, appendages, induced
resistance, nonlinear free-surface effects and finite depth are outside the model.
"""

from .hull import Attitude, Hull, PlaneSection, TriMesh, tessellate
from .hydrostatics import GRAVITY, RHO_FRESH, Hydrostatics, hydrostatics, solve_reference_heave
from .iges import IgesFile, NurbsSurface, read_iges

__all__ = [
    "Attitude", "Hull", "PlaneSection", "TriMesh", "tessellate",
    "Hydrostatics", "hydrostatics", "solve_reference_heave", "RHO_FRESH", "GRAVITY",
    "IgesFile", "NurbsSurface", "read_iges",
]
__version__ = "0.2.0.dev0"
