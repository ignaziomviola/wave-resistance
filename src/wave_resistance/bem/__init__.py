"""Transparent dense Rankine-panel reference implementation."""

from .mesh import SurfaceMesh, offset_hull_mesh, rectangular_free_surface
from .kernels import source_panel_potential, source_panel_velocity

__all__ = ["SurfaceMesh", "offset_hull_mesh", "rectangular_free_surface",
           "source_panel_potential", "source_panel_velocity"]
