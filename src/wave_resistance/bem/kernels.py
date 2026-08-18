"""Rankine source integrals over planar triangles.

The kernel is ``G=1/(4*pi*r)``.  Velocity uses the exact edge/solid-angle
formula.  Potential uses Duffy quadrature for a target on the panel and
adaptive symmetric quadrature otherwise.
"""

from __future__ import annotations
import math
from typing import Tuple
import numpy as np
from numpy.polynomial.legendre import leggauss

FOUR_PI=4.0*math.pi
_BARY=np.array([[1/3,1/3,1/3],
 [0.059715871789770,0.470142064105115,0.470142064105115],
 [0.470142064105115,0.059715871789770,0.470142064105115],
 [0.470142064105115,0.470142064105115,0.059715871789770],
 [0.797426985353087,0.101286507323456,0.101286507323456],
 [0.101286507323456,0.797426985353087,0.101286507323456],
 [0.101286507323456,0.101286507323456,0.797426985353087]])
_WEIGHTS=np.array([0.225,0.132394152788506,0.132394152788506,
                   0.132394152788506,0.125939180544827,
                   0.125939180544827,0.125939180544827])


def _geometry(triangle: np.ndarray) -> Tuple[np.ndarray,float]:
    t=np.asarray(triangle,float)
    cross=np.cross(t[1]-t[0],t[2]-t[0]); twice=np.linalg.norm(cross)
    if twice<=1e-15: raise ValueError("degenerate source panel")
    return cross/twice,0.5*twice


def solid_angle(point: np.ndarray, triangle: np.ndarray) -> float:
    """Signed solid angle; positive on the side selected by panel normal."""
    p=np.asarray(point,float); t=np.asarray(triangle,float); normal,_=_geometry(t)
    r=t-p; lengths=np.linalg.norm(r,axis=1)
    scale=max(np.max(np.linalg.norm(t-t[0],axis=1)),1.0)
    h=float(np.dot(p-t[0],normal))
    if abs(h)<1e-12*scale:
        # Boundary limit.  Callers use centroids/interior collocation points.
        return 2*math.pi
    numerator=float(np.dot(r[0],np.cross(r[1],r[2])))
    denominator=(np.prod(lengths)+np.dot(r[0],r[1])*lengths[2]
                 +np.dot(r[1],r[2])*lengths[0]+np.dot(r[2],r[0])*lengths[1])
    raw=2*math.atan2(numerator,denominator)
    # With CCW vertices, vectors from the field point give the opposite sign.
    omega=-raw
    if h*omega<0: omega=-omega
    return omega


def source_panel_velocity(point: np.ndarray, triangle: np.ndarray,
                          *, principal_value: bool=False) -> np.ndarray:
    """Exact integral of ``grad G`` including the selected one-sided jump."""
    p=np.asarray(point,float); t=np.asarray(triangle,float); normal,_=_geometry(t)
    gradient=np.zeros(3)
    for i,j in ((0,1),(1,2),(2,0)):
        edge=t[j]-t[i]; length=np.linalg.norm(edge); tangent=edge/length
        outward=np.cross(tangent,normal)
        r1=np.linalg.norm(p-t[i]); r2=np.linalg.norm(p-t[j])
        denominator=max(r1+r2-length,1e-300)
        edge_integral=math.log(max((r1+r2+length)/denominator,1.0))
        gradient -= outward*edge_integral
    if not principal_value:
        gradient -= normal*solid_angle(p,t)
    return gradient/FOUR_PI


def _rule(point: np.ndarray, triangle: np.ndarray) -> float:
    _,area=_geometry(triangle)
    samples=_BARY@np.asarray(triangle,float)
    return area*float(np.dot(_WEIGHTS,1/np.linalg.norm(samples-point,axis=1)))/FOUR_PI


def _subdivide(t: np.ndarray):
    a,b,c=t; ab=(a+b)/2; bc=(b+c)/2; ca=(c+a)/2
    return (np.array((a,ab,ca)),np.array((ab,b,bc)),
            np.array((ca,bc,c)),np.array((ab,bc,ca)))


def _adaptive(point: np.ndarray, triangle: np.ndarray, rtol: float,
              atol: float, depth: int) -> float:
    coarse=_rule(point,triangle)
    children=_subdivide(triangle)
    fine=sum(_rule(point,child) for child in children)
    if depth<=0 or abs(fine-coarse)<=max(atol,rtol*abs(fine)):
        return fine
    return sum(_adaptive(point,child,rtol,atol/4,depth-1) for child in children)


def _duffy_at_point(point: np.ndarray, triangle: np.ndarray, order: int=64) -> float:
    nodes,weights=leggauss(order); u=(nodes+1)/2; wu=weights/2
    result=0.0
    t=np.asarray(triangle,float); p=np.asarray(point,float)
    for b,c in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
        cross=np.linalg.norm(np.cross(b-p,c-p))
        if cross<1e-30: continue
        # y=p+u*((1-v)(b-p)+v(c-p)); Jacobian=cross*u.
        direction=(1-u[:,None])*(b-p)+u[:,None]*(c-p)
        angular=float(np.dot(wu,1/np.linalg.norm(direction,axis=1)))
        result += cross*angular # integral over u is exactly one
    return result/FOUR_PI


def source_panel_potential(point: np.ndarray, triangle: np.ndarray, *,
                           rtol: float=1e-9, atol: float=1e-12,
                           max_depth: int=7) -> float:
    p=np.asarray(point,float); t=np.asarray(triangle,float); normal,_=_geometry(t)
    projected=p-np.dot(p-t[0],normal)*normal
    # Barycentric containment of the projection.
    v0=t[1]-t[0]; v1=t[2]-t[0]; v2=projected-t[0]
    den=np.dot(v0,v0)*np.dot(v1,v1)-np.dot(v0,v1)**2
    b=(np.dot(v1,v1)*np.dot(v2,v0)-np.dot(v0,v1)*np.dot(v2,v1))/den
    c=(np.dot(v0,v0)*np.dot(v2,v1)-np.dot(v0,v1)*np.dot(v2,v0))/den
    scale=max(np.linalg.norm(v0),np.linalg.norm(v1),1.0)
    if abs(np.dot(p-t[0],normal))<1e-12*scale and b>=-1e-12 and c>=-1e-12 and b+c<=1+1e-12:
        return _duffy_at_point(p,t)
    return _adaptive(p,t,rtol,atol,max_depth)


def point_source_potential(point: np.ndarray, centroid: np.ndarray, area: float) -> float:
    return area/(FOUR_PI*np.linalg.norm(np.asarray(point)-centroid))


def point_source_velocity(point: np.ndarray, centroid: np.ndarray, area: float) -> np.ndarray:
    delta=np.asarray(point)-centroid; r=np.linalg.norm(delta)
    return -area*delta/(FOUR_PI*r**3)
