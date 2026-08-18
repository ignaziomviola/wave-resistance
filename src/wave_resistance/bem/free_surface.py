"""Discrete free-surface convection, damping, and graph movement."""
from __future__ import annotations
import numpy as np
from .mesh import SurfaceMesh


def boundary_vertices(mesh: SurfaceMesh) -> np.ndarray:
    """Return vertices on the exterior boundary of a triangular surface."""
    owners = {}
    for face in mesh.faces:
        for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
            edge=tuple(sorted((int(a),int(b))))
            owners[edge]=owners.get(edge,0)+1
    return np.unique([vertex for edge,count in owners.items() if count==1
                      for vertex in edge]).astype(int)


def project_panel_elevation(mesh: SurfaceMesh, panel_elevation: np.ndarray,
                            fixed_vertices: np.ndarray) -> np.ndarray:
    """Least-squares project face-centred elevations onto graph vertices.

    The returned vector is the vertex elevation.  Fixed vertices are imposed
    exactly, so the design waterline and outer truncation boundary cannot
    drift during nonlinear iteration.
    """
    values=np.asarray(panel_elevation,float)
    if values.shape!=(len(mesh.faces),):
        raise ValueError("panel_elevation must contain one value per face")
    fixed=np.unique(np.asarray(fixed_vertices,int))
    if np.any(fixed<0) or np.any(fixed>=len(mesh.vertices)):
        raise ValueError("fixed vertex index is outside the mesh")
    interpolation=np.zeros((len(mesh.faces),len(mesh.vertices)))
    rows=np.arange(len(mesh.faces))[:,None]
    interpolation[rows,mesh.faces]=1/3
    free=np.setdiff1d(np.arange(len(mesh.vertices)),fixed)
    result=np.zeros(len(mesh.vertices))
    if len(free):
        result[free]=np.linalg.lstsq(interpolation[:,free],values,rcond=None)[0]
    return result

def least_squares_derivative(points: np.ndarray, *, axis: int=0, neighbours: int=8,
                             upwind: bool=True) -> np.ndarray:
    points=np.asarray(points,float); n=len(points); d=np.zeros((n,n))
    for i,p in enumerate(points):
        delta=points[:,:2]-p[:2]; distance=np.linalg.norm(delta,axis=1)
        candidates=np.arange(n)
        if upwind:
            upstream=candidates[delta[:,0]>=-1e-12] # flow is toward decreasing x
            if len(upstream)>=3: candidates=upstream
        order=candidates[np.argsort(distance[candidates])[:max(3,min(neighbours,len(candidates)))]]
        a=np.column_stack((np.ones(len(order)),delta[order,0],delta[order,1]))
        pseudo=np.linalg.pinv(a)
        d[i,order]=pseudo[axis+1]
    return d

def sponge_strength(points: np.ndarray, length: float, *, upstream: float,
                    downstream: float, lateral: float, fraction: float,
                    maximum: float) -> np.ndarray:
    p=np.asarray(points,float); value=np.zeros(len(p)); x=p[:,0]/length; y=np.abs(p[:,1])/length
    down_start=1+downstream*(1-fraction)
    lat_start=lateral*(1-fraction)
    value=np.maximum(value,np.clip((x-down_start)/max(downstream*fraction,1e-12),0,1)**2)
    value=np.maximum(value,np.clip((y-lat_start)/max(lateral*fraction,1e-12),0,1)**2)
    value=np.maximum(value,np.clip((-upstream-x)/max(upstream*0.15,1e-12),0,1)**2)
    return maximum*value

def move_graph(mesh: SurfaceMesh, panel_elevation: np.ndarray,
               fixed_vertices: np.ndarray) -> SurfaceMesh:
    vertex_eta=project_panel_elevation(mesh,panel_elevation,fixed_vertices)
    vertices=np.array(mesh.vertices,copy=True); vertices[:,2]=vertex_eta
    return mesh.with_vertices(vertices,name="nonlinear_free_surface")
