"""Dense, deterministic Rankine-panel reference implementation."""

from .mesh import MeshDiagnostics, SurfaceMesh, free_surface_mesh, offset_hull_mesh
from .kernels import source_panel_potential, source_panel_velocity, solid_angle

__all__ = [
    "MeshDiagnostics",
    "SurfaceMesh",
    "free_surface_mesh",
    "offset_hull_mesh",
    "solid_angle",
    "source_panel_potential",
    "source_panel_velocity",
]
