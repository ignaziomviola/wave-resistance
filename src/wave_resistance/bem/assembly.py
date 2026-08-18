"""Dense single-layer potential and normal-derivative assembly."""

from __future__ import annotations
import numpy as np
from .kernels import (point_source_potential, point_source_velocity,
                      source_panel_potential, source_panel_velocity)
from .mesh import SurfaceMesh


def influence_matrices(source: SurfaceMesh, target_points: np.ndarray,
                       target_normals: np.ndarray, *, far_field_ratio: float=4.0,
                       principal_diagonal: bool=False):
    points=np.asarray(target_points,float); normals=np.asarray(target_normals,float)
    triangles=source.triangles; centroids=source.centroids; areas=source.areas
    radii=np.max(np.linalg.norm(triangles-centroids[:,None,:],axis=2),axis=1)
    s=np.empty((len(points),len(triangles))); h=np.empty_like(s)
    for i,(point,normal) in enumerate(zip(points,normals)):
        distance=np.linalg.norm(centroids-point,axis=1)
        far=distance>far_field_ratio*radii if np.isfinite(far_field_ratio) else np.zeros(len(areas),bool)
        if np.any(far):
            delta=point-centroids[far]; r=np.linalg.norm(delta,axis=1)
            s[i,far]=areas[far]/(4*np.pi*r)
            velocity=-areas[far,None]*delta/(4*np.pi*r[:,None]**3)
            h[i,far]=velocity@normal
        for j in np.flatnonzero(~far):
            on_self=(distance[j]<1e-11*max(radii[j],1.0))
            s[i,j]=source_panel_potential(point,triangles[j],rtol=2e-7,max_depth=4)
            velocity=source_panel_velocity(point,triangles[j],
                principal_value=bool(principal_diagonal and on_self))
            h[i,j]=np.dot(velocity,normal)
    return s,h


def collocation_matrices(mesh: SurfaceMesh, *, far_field_ratio: float=4.0):
    s,h=influence_matrices(mesh,mesh.centroids,mesh.normals,
                           far_field_ratio=far_field_ratio)
    # The fluid is on +normal for body panels and on -normal for the upper
    # free-surface boundary.  Select the corresponding one-sided jump.
    diagonal=np.where(mesh.tags=="free_surface",0.5,-0.5)
    np.fill_diagonal(h,diagonal)
    return s,h

def collocation_normal_matrix(mesh: SurfaceMesh, *, far_field_ratio: float=4.0):
    """Assemble only the normal derivative, avoiding potential quadrature."""
    points=mesh.centroids; normals=mesh.normals; triangles=mesh.triangles
    centroids=mesh.centroids; areas=mesh.areas
    radii=np.max(np.linalg.norm(triangles-centroids[:,None,:],axis=2),axis=1)
    h=np.empty((len(points),len(triangles)))
    for i,(point,normal) in enumerate(zip(points,normals)):
        distance=np.linalg.norm(centroids-point,axis=1)
        far=distance>far_field_ratio*radii if np.isfinite(far_field_ratio) else np.zeros(len(areas),bool)
        if np.any(far):
            delta=point-centroids[far]; r=np.linalg.norm(delta,axis=1)
            velocity=-areas[far,None]*delta/(4*np.pi*r[:,None]**3)
            h[i,far]=velocity@normal
        for j in np.flatnonzero(~far):
            h[i,j]=np.dot(source_panel_velocity(point,triangles[j]),normal)
    np.fill_diagonal(h,np.where(mesh.tags=="free_surface",0.5,-0.5))
    return h


def evaluate_velocity(mesh: SurfaceMesh, strengths: np.ndarray,
                      points: np.ndarray, *, far_field_ratio: float=4.0) -> np.ndarray:
    points=np.asarray(points,float); strengths=np.asarray(strengths,float)
    triangles=mesh.triangles; centroids=mesh.centroids; areas=mesh.areas
    radii=np.max(np.linalg.norm(triangles-centroids[:,None,:],axis=2),axis=1)
    result=np.zeros_like(points,dtype=float)
    for i,point in enumerate(points):
        distance=np.linalg.norm(centroids-point,axis=1)
        far=distance>far_field_ratio*radii if np.isfinite(far_field_ratio) else np.zeros(len(areas),bool)
        if np.any(far):
            delta=point-centroids[far]; r=np.linalg.norm(delta,axis=1)
            velocity=-areas[far,None]*delta/(4*np.pi*r[:,None]**3)
            result[i]+=strengths[far]@velocity
        for j in np.flatnonzero(~far):
            result[i]+=strengths[j]*source_panel_velocity(point,triangles[j])
    return result
