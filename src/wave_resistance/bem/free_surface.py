"""Discrete free-surface convection, damping, and graph movement."""
from __future__ import annotations
import numpy as np
from .mesh import SurfaceMesh

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
    values=np.asarray(panel_elevation,float); accum=np.zeros(len(mesh.vertices)); count=np.zeros(len(mesh.vertices))
    for value,face in zip(values,mesh.faces):
        accum[face]+=value; count[face]+=1
    vertex_eta=np.divide(accum,count,out=np.zeros_like(accum),where=count>0)
    vertex_eta[np.asarray(fixed_vertices,dtype=int)]=0.0
    vertices=np.array(mesh.vertices,copy=True); vertices[:,2]=vertex_eta
    return mesh.with_vertices(vertices,name="nonlinear_free_surface")
