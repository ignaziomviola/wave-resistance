"""Rankine constant-triangle integrals evaluated by Duffy quadrature.

The transformation resolves a collocation singularity at a triangle vertex;
an observation point on a panel is handled by splitting about that point.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from functools import lru_cache

FOUR_PI=4*np.pi

@lru_cache(maxsize=None)
def _unit_legendre(order):
    q,w=leggauss(order)
    return (q+1)/2, w/2

def _duffy_vertex(obs, a,b,c, order):
    q,w=_unit_legendre(order)
    total=0.; grad=np.zeros(3)
    ab=b-a; ac=c-a; jac=np.linalg.norm(np.cross(ab,ac))
    if jac == 0.0:
        return 0.0, np.zeros(3)
    for i,u in enumerate(q):
        pts=a+u*ab+u*q[:,None]*ac
        r=obs-pts; rn=np.linalg.norm(r,axis=1)
        weights=w[i]*w*u*jac
        total += np.sum(weights/rn)
        grad += np.sum(weights[:,None]*(-r/rn[:,None]**3),axis=0)
    return total/FOUR_PI, grad/FOUR_PI

def _integral(obs, tri, order):
    obs=np.asarray(obs,float); tri=np.asarray(tri,float)
    # Split around the orthogonal projection for self and near-singular
    # interactions. Duffy's radial coordinate then resolves the sharp peak.
    n=np.cross(tri[1]-tri[0],tri[2]-tri[0]); unit=n/np.linalg.norm(n)
    signed_distance=np.dot(obs-tri[0],unit); distance=abs(signed_distance)
    projection=obs-signed_distance*unit
    v0=tri[1]-tri[0]; v1=tri[2]-tri[0]; v2=projection-tri[0]
    d00=np.dot(v0,v0); d01=np.dot(v0,v1); d11=np.dot(v1,v1)
    d20=np.dot(v2,v0); d21=np.dot(v2,v1); denominator=d00*d11-d01*d01
    beta=(d11*d20-d01*d21)/denominator
    gamma=(d00*d21-d01*d20)/denominator
    alpha=1.0-beta-gamma
    inside=min(alpha,beta,gamma)>=-1e-13
    scale=max(np.linalg.norm(tri[1]-tri[0]),np.linalg.norm(tri[2]-tri[0]))
    if inside and distance < 0.5*scale:
        vals=[_duffy_vertex(obs,projection,tri[i],tri[(i+1)%3],order) for i in range(3)]
        return sum(x[0] for x in vals),sum((x[1] for x in vals),np.zeros(3))
    return _duffy_vertex(obs,tri[0],tri[1],tri[2],order)

def source_panel_potential(observation, triangle, order=16): return _integral(observation,triangle,order)[0]
def source_panel_velocity(observation, triangle, order=16): return _integral(observation,triangle,order)[1]
