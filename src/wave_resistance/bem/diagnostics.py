"""Convergence history containers and graph validity checks."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from .mesh import SurfaceMesh

@dataclass
class ResidualHistory:
    bem: List[float]=field(default_factory=list)
    hull_impermeability: List[float]=field(default_factory=list)
    free_surface_kinematic: List[float]=field(default_factory=list)
    free_surface_dynamic: List[float]=field(default_factory=list)
    nonlinear_update: List[float]=field(default_factory=list)
    waterline: List[float]=field(default_factory=list)
    geometry: List[float]=field(default_factory=list)
    force_balance: List[float]=field(default_factory=list)

    def as_dict(self) -> Dict[str,np.ndarray]:
        return {key:np.asarray(value,float) for key,value in vars(self).items()}

def validate_graph(mesh: SurfaceMesh, max_slope: float) -> Tuple[bool,Tuple[str,...],float]:
    if not np.all(np.isfinite(mesh.vertices)):
        return False,("free surface contains non-finite coordinates",),float("inf")
    tri=mesh.triangles
    a=tri[:,1,:2]-tri[:,0,:2]; b=tri[:,2,:2]-tri[:,0,:2]
    projected=0.5*(a[:,0]*b[:,1]-a[:,1]*b[:,0])
    if np.any(np.abs(projected)<1e-15):
        return False,("free surface has a folded or degenerate xy projection",),float("inf")
    normals=mesh.normals
    slopes=np.sqrt(normals[:,0]**2+normals[:,1]**2)/np.maximum(np.abs(normals[:,2]),1e-30)
    maximum=float(np.max(slopes))
    if maximum>max_slope:
        return False,("free-surface slope exceeds the mesh-resolution limit",),maximum
    return True,(),maximum
