"""Neumann-Kelvin wave resistance of a bare sailing-yacht canoe body.

Scope: steady, deep-water, inviscid, irrotational flow with a linearised free-surface
condition and the exact body condition on the actual wetted hull at a prescribed
attitude.  Viscous and form resistance, circulation, lift, leeway, appendages, induced
resistance, nonlinear free-surface effects and finite depth are outside the model.
"""

from .greens import envelope_ok, green, oscillation_count, wave_part, wave_part_gradient
from .hull import Attitude, Hull, PlaneSection, TriMesh, tessellate
from .hydrostatics import GRAVITY, RHO_FRESH, Hydrostatics, hydrostatics, solve_reference_heave
from .iges import IgesFile, NurbsSurface, read_iges
from .measurements import Run, ittc57_friction_coefficient, sysser01_runs
from .nk import NKResult, check_envelope, influence_matrix, pressure_resistance, solve_nk
from .reference import sysser01_reference
from .spectrum import mesh_resolved_lambda, resistance_constant, wave_resistance

__all__ = [
    "Attitude", "Hull", "PlaneSection", "TriMesh", "tessellate",
    "Hydrostatics", "hydrostatics", "solve_reference_heave", "RHO_FRESH", "GRAVITY",
    "IgesFile", "NurbsSurface", "read_iges",
    "green", "wave_part", "wave_part_gradient", "oscillation_count", "envelope_ok",
    "NKResult", "solve_nk", "influence_matrix", "pressure_resistance", "check_envelope",
    "wave_resistance", "resistance_constant", "mesh_resolved_lambda",
    "sysser01_reference", "sysser01_runs", "Run", "ittc57_friction_coefficient",
]
__version__ = "0.2.0.dev0"
