"""Pressure and solved-source far-field resistance estimates."""
from __future__ import annotations
import math
import numpy as np
from .mesh import SurfaceMesh
from .assembly import evaluate_velocity

def pressure_resistance(mesh: SurfaceMesh, velocity: np.ndarray, rho: float, speed: float):
    pressure=0.5*rho*(speed**2-np.sum(velocity**2,axis=1))
    resistance=float(np.sum(pressure*mesh.normals[:,0]*mesh.areas))
    return resistance,pressure

def kochin_resistance(mesh: SurfaceMesh, strengths: np.ndarray, rho: float,
                      speed: float, gravity: float, *, order: int=48,
                      t_max: float=6.0) -> float:
    """Deep-water wave resistance of the computed source distribution.

    This is the classical symmetric-source Kochin integral in the transformed
    variable ``t=tan(theta)``.  It is deliberately independent of hull pressure.
    """
    nodes,weights=np.polynomial.legendre.leggauss(order); t=0.5*t_max*(nodes+1); w=0.5*t_max*weights
    lam=np.sqrt(1+t*t); k0=gravity/speed**2; c=mesh.centroids; q=strengths*mesh.areas
    amplitude=[]
    for l,tt in zip(lam,t):
        phase=k0*l*c[:,0] + k0*l*tt*c[:,1]
        decay=np.exp(-np.clip(k0*l*l*np.maximum(c[:,2],0),0,700))
        amplitude.append(np.sum(q*decay*np.exp(1j*phase)))
    amplitude=np.asarray(amplitude)
    integral=float(np.dot(w,lam**3*np.abs(amplitude)**2))
    return rho*k0**2*integral/math.pi

def force_balance(pressure: float, far_field: float, relative_tolerance: float,
                  absolute_tolerance: float):
    discrepancy=abs(pressure-far_field)
    tolerance=max(relative_tolerance*max(abs(pressure),abs(far_field)),absolute_tolerance)
    return discrepancy,discrepancy<=tolerance

def momentum_flux_resistance(mesh: SurfaceMesh, strengths: np.ndarray, rho: float,
                             speed: float, *, order: int=6,
                             depth: float=None) -> float:
    """Independent outer-control-box momentum-flux estimate.

    Upstream, downstream, lateral, bottom, and the just-submerged top face are
    integrated; the uniform-stream tensor is subtracted analytically pointwise.
    """
    vertices=mesh.vertices; xmin,xmax=np.min(vertices[:,0]),np.max(vertices[:,0])
    ymax=float(np.max(np.abs(vertices[:,1]))); length=max(xmax-xmin,1e-12)
    depth=float(depth or max(np.max(vertices[:,2]),0.5*length))
    nodes,weights=np.polynomial.legendre.leggauss(order)
    uniform=np.array((-speed,0.,0.)); total=0.0
    def face(origin,a,b,normal):
        nonlocal total
        uv=np.array([(u,v) for u in nodes for v in nodes]); ww=np.outer(weights,weights).ravel()
        points=origin+(uv[:,0,None]+1)/2*a+(uv[:,1,None]+1)/2*b
        velocity=evaluate_velocity(mesh,strengths,points,far_field_ratio=2.5)+uniform
        pressure=.5*rho*(speed**2-np.sum(velocity**2,axis=1))
        flux=rho*velocity[:,0]*(velocity@normal)+pressure*normal[0]
        uniform_flux=rho*uniform[0]*np.dot(uniform,normal)
        jac=np.linalg.norm(np.cross(a,b))/4
        total += float(np.dot(ww,flux-uniform_flux)*jac)
    face(np.array((xmin,-ymax,0.)),np.array((0.,2*ymax,0.)),np.array((0.,0.,depth)),np.array((-1.,0.,0.)))
    face(np.array((xmax,-ymax,0.)),np.array((0.,2*ymax,0.)),np.array((0.,0.,depth)),np.array((1.,0.,0.)))
    face(np.array((xmin,-ymax,0.)),np.array((length,0.,0.)),np.array((0.,0.,depth)),np.array((0.,-1.,0.)))
    face(np.array((xmin,ymax,0.)),np.array((length,0.,0.)),np.array((0.,0.,depth)),np.array((0.,1.,0.)))
    face(np.array((xmin,-ymax,depth)),np.array((length,0.,0.)),np.array((0.,2*ymax,0.)),np.array((0.,0.,1.)))
    # A small inward offset selects the water side of the panel sheet.  The
    # displacement-hull footprint is measure-small on this far-field box.
    face(np.array((xmin,-ymax,1e-8*length)),np.array((length,0.,0.)),np.array((0.,2*ymax,0.)),np.array((0.,0.,-1.)))
    return total
