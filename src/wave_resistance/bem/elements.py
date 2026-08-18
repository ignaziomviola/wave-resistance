"""Low-order triangular element utilities."""

from __future__ import annotations
import numpy as np


def barycentric_coordinates(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    a,b,c=np.asarray(triangle,float); p=np.asarray(point,float)
    v0=b-a; v1=c-a; v2=p-a
    d00=np.dot(v0,v0); d01=np.dot(v0,v1); d11=np.dot(v1,v1)
    d20=np.dot(v2,v0); d21=np.dot(v2,v1)
    denominator=d00*d11-d01*d01
    if abs(denominator)<1e-30: raise ValueError("degenerate triangle")
    v=(d11*d20-d01*d21)/denominator
    w=(d00*d21-d01*d20)/denominator
    return np.array((1-v-w,v,w))


def triangle_quality(triangle: np.ndarray) -> tuple:
    t=np.asarray(triangle,float)
    edges=np.array([np.linalg.norm(t[1]-t[0]),np.linalg.norm(t[2]-t[1]),
                    np.linalg.norm(t[0]-t[2])])
    area=0.5*np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))
    aspect=float(np.max(edges)/max(2*area/np.max(edges),1e-30))
    cos=(edges[0]**2+edges[2]**2-edges[1]**2)/(2*edges[0]*edges[2])
    return area,aspect,float(np.degrees(np.arccos(np.clip(cos,-1,1))))
